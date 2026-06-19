"""Landsat 8/9 TOA adapter — ``B2_landsat..B7_landsat`` + ``QA_PIXEL`` (TASK-012).

Replaces the Landsat placeholders with **top-of-atmosphere reflectance** on the cell
grid, reading the clipped L1TP ``.tar`` scenes (raw DN + ``_MTL.json``). It reproduces
the GEE ``LANDSAT/LC0{8,9}/C02/T1_TOA`` value domain:

    ρ = (REFLECTANCE_MULT_BAND_n · DN + REFLECTANCE_ADD_BAND_n) / sin(SUN_ELEVATION)

giving reflectance in ``[0, ~1]`` (snow/specular pixels may exceed 1) — **not** scaled
by 10000; the downstream normalization divides by 10000 itself. ``DN == 0`` is the
scene's no-data and maps to ``-9999`` (valid threshold ``>= 1e-7``; zero is invalid).

Three behaviours the contract (``base.py``) mandates for scene sources:

- **L9→L8 fallback.** Try Landsat 9 for the day first; fall back to Landsat 8 only if L9
  has no covering scene; both absent → all-``-9999`` placeholder (renamed band names).
- **Same-(tile, date) coalesce.** A WRS-2 path/row may be delivered as several products
  on one date (e.g. L9 ``044024`` twice on ``20250425``). Gather **all** of them and
  coalesce per pixel — first valid (non-nodata, in-threshold) value wins, deterministic
  order = latest processing time first — a valid-pixel **union**, never an average.
- **Cross-tile mosaic before crop.** Multiple WRS-2 tiles covering one cell are merged in
  their native UTM zone before the reproject.

**Resample = NEAREST (bit-exact parity).** GEE's ``export_from_csv_utm`` upsamples the
30 m Landsat to the 10 m cell grid as constant blocks, so **nearest reproduces GEE
bit-exactly** (median 0 across the three TASK-012b reference patches; bilinear smears
~0.003-0.012 over high-gradient cloud/snow edges). This is the same coarse-source rule
MODIS uses — anything coarser than the 10 m cell is nearest-upsampled by GEE. Nearest also
cannot blend a valid reflectance with the ``-9999`` no-data, so no edge-bleed guard is
needed. (Supersedes the spec's "bilinear for reflectance" — the archive disproved it.)

**Mixed UTM zone (load-bearing).** The archive is **mixed-zone per scene**: paths 043/044
are EPSG:32611 (same zone as the cell grid), 042024 is 32612, 042025 is 32611. USGS
assigns the zone by scene-center longitude, so we **read each band's native CRS** and pass
it to :func:`~src.data.local_sources.base.reproject_to_cell` (zone-agnostic). We never
hardcode 32612. ``QA_PIXEL`` rides the same fallback/coalesce path, also nearest.
"""

from __future__ import annotations

import datetime
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog

from src.data.config import NO_DATA_VALUE
from src.data.local_sources._scene_ops import (
    BandRead,
    cell_window,
    coalesce_tile,
    mosaic_to_cell,
)
from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
)

logger = structlog.get_logger(__name__)

#: Renamed optical bands → the underlying Landsat band number (B2..B7 → 2..7).
_LANDSAT_BANDS: dict[str, int] = {f"B{n}_landsat": n for n in range(2, 8)}

#: The valid-reflectance floor; ``< 1e-7`` (incl. exactly 0 = scene no-data) is invalid.
_VALID_MIN: float = 1e-7

#: Clipped scene tar stem: ``LC0{8,9}_L1TP_{path}{row}_{acq}_{proc}_02_T1``.
_SCENE_RE = re.compile(r"^LC0[89]_L1TP_(?P<pathrow>\d{6})_(?P<acq>\d{8})_(?P<proc>\d{8})_02_T1$")


@dataclass(frozen=True)
class _SceneInfo:
    """Parsed identity of one clipped Landsat scene tar."""

    path: Path
    pathrow: str
    acq: datetime.date
    proc: datetime.date


def _parse_scene(tar_path: Path) -> _SceneInfo | None:
    """Parse a scene tar's stem; return ``None`` if it is not an L1TP T1 product."""
    m = _SCENE_RE.match(tar_path.stem)
    if m is None:
        return None
    return _SceneInfo(
        path=tar_path,
        pathrow=m.group("pathrow"),
        acq=datetime.datetime.strptime(m.group("acq"), "%Y%m%d").date(),
        proc=datetime.datetime.strptime(m.group("proc"), "%Y%m%d").date(),
    )


def _toa_coefficients(mtl: dict, band_num: int) -> tuple[float, float, float]:
    """Return ``(mult, add, sun_elevation_deg)`` for ``band_num`` from MTL JSON."""
    root = mtl["LANDSAT_METADATA_FILE"]
    rescale = root["LEVEL1_RADIOMETRIC_RESCALING"]
    mult = float(rescale[f"REFLECTANCE_MULT_BAND_{band_num}"])
    add = float(rescale[f"REFLECTANCE_ADD_BAND_{band_num}"])
    sun_elev = float(root["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"])
    return mult, add, sun_elev


def _band_filename(mtl: dict, band_num: int) -> str:
    """Return the ``_B{n}.TIF`` member name for ``band_num`` from MTL JSON."""
    return mtl["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"][f"FILE_NAME_BAND_{band_num}"]


class _LandsatBase(LocalSourceAdapter):
    """Shared Landsat scene discovery, MTL/DN→TOA decode, coalesce + mosaic."""

    def __init__(self, *, landsat9_root: Path, landsat8_root: Path) -> None:
        self.landsat9_root = landsat9_root
        self.landsat8_root = landsat8_root

    def _scenes_for_day(self, root: Path, day: datetime.date) -> list[_SceneInfo]:
        """All clipped scenes under ``root`` acquired on ``day`` (any path/row)."""
        scenes = [_parse_scene(p) for p in sorted(root.glob("*.tar"))]
        return [s for s in scenes if s is not None and s.acq == day]

    def _read_band_toa(self, scene: _SceneInfo, band_num: int, cell: GridCell) -> BandRead | None:
        """Read one band's DN over the cell footprint and convert to TOA reflectance.

        Windowed to the cell neighbourhood (see :func:`cell_window`) so a full Landsat
        scene band is never materialized whole. Returns ``None`` if the scene tar lacks the
        MTL or band member, or the cell does not intersect the scene. ``DN == 0`` (scene
        no-data) and out-of-threshold values become ``-9999``.
        """
        with tarfile.open(scene.path, "r") as tar:
            mtl_name = next(
                (n.name for n in tar.getmembers() if n.name.upper().endswith("_MTL.JSON")),
                None,
            )
            if mtl_name is None:
                return None
            mtl = json.loads(tar.extractfile(mtl_name).read())  # type: ignore[union-attr]
            band_member = _band_filename(mtl, band_num)
            if band_member not in tar.getnames():
                return None
            with tempfile.TemporaryDirectory() as tmp:
                tar.extract(band_member, path=tmp)
                with rasterio.open(Path(tmp) / band_member) as ds:
                    window = cell_window(ds, cell)
                    if window is None:
                        return None
                    dn = ds.read(1, window=window).astype(np.float64)
                    transform = ds.window_transform(window)
                    crs = str(ds.crs)

        mult, add, sun_elev = _toa_coefficients(mtl, band_num)
        toa = (mult * dn + add) / np.sin(np.deg2rad(sun_elev))
        # DN == 0 is the scene fill; anything below the valid floor is no-data.
        toa[(dn == 0) | (toa < _VALID_MIN)] = float(NO_DATA_VALUE)
        return BandRead(values=toa, transform=transform, crs=crs)

    def _band_on_cell(
        self,
        scenes: list[_SceneInfo],
        band_num: int,
        cell: GridCell,
        *,
        categorical: bool,
    ) -> npt.NDArray[np.float32] | None:
        """Coalesce per tile, mosaic across tiles, reproject one band to the cell grid.

        Returns ``None`` if no scene yields a readable band (caller falls back / fills).
        """
        # Group scenes by tile (path/row); within a tile, latest processing first.
        by_tile: dict[str, list[_SceneInfo]] = {}
        for scene in scenes:
            by_tile.setdefault(scene.pathrow, []).append(scene)

        tile_reads: list[BandRead] = []
        for pathrow, tile_scenes in by_tile.items():
            ordered = sorted(tile_scenes, key=lambda s: s.proc, reverse=True)
            reads = [r for s in ordered if (r := self._read_band_toa(s, band_num, cell))]
            if reads:
                tile_reads.append(coalesce_tile(reads))

        if not tile_reads:
            return None

        # Mixed-UTM-zone safe (paths 043/044 → 32611, 042024 → 32612): mosaic within each
        # zone, reproject each to the cell grid, then first-valid-combine. ``mosaic_tiles``
        # alone would raise ``CRS mismatch`` on a cross-zone day. See ``mosaic_to_cell``.
        return mosaic_to_cell(tile_reads, cell, categorical=categorical)

    def _select_scenes(self, day: datetime.date) -> list[_SceneInfo]:
        """L9→L8 fallback: L9 scenes for the day if any, else L8 scenes."""
        l9 = self._scenes_for_day(self.landsat9_root, day)
        if l9:
            return l9
        return self._scenes_for_day(self.landsat8_root, day)


class LandsatAdapter(_LandsatBase):
    """Landsat 8/9 ``B2_landsat..B7_landsat`` TOA adapter (``high`` tier).

    Args:
        landsat9_root: Clipped Landsat 9 archive root (tried first).
        landsat8_root: Clipped Landsat 8 archive root (fallback).
    """

    bands_out = list(_LANDSAT_BANDS)
    spatial_kind = "high"
    native_fill = None

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the six optical TOA bands on the cell grid (``-9999`` where missing)."""
        if day is None:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
        scenes = self._select_scenes(day)
        if not scenes:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.float32]] = []
        for band_name in self.bands_out:
            # NEAREST (categorical path): GEE's export upsamples the 30 m Landsat to the
            # 10 m cell as constant blocks, so nearest is bit-exact (median 0 across the
            # three reference patches; bilinear smears ~0.003-0.012 over snow/cloud edges).
            # Same coarse-source rule as MODIS. Nearest also cannot blend valid + -9999.
            arr = self._band_on_cell(scenes, _LANDSAT_BANDS[band_name], cell, categorical=True)
            if arr is None:
                return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
            bands.append(arr)

        logger.info(
            "landsat_fetch",
            cell_id=cell.cell_id,
            day=day.isoformat(),
            scenes=len(scenes),
            sat="L9" if self._scenes_for_day(self.landsat9_root, day) else "L8",
        )
        return np.stack(bands, axis=0).astype(np.float32)


class LandsatCloudAdapter(_LandsatBase):
    """Landsat ``QA_PIXEL`` cloud-flag adapter (categorical/NN, cloud slot).

    Args:
        landsat9_root: Clipped Landsat 9 archive root (tried first).
        landsat8_root: Clipped Landsat 8 archive root (fallback).
    """

    bands_out = ["QA_PIXEL"]
    spatial_kind = "time"
    native_fill = None

    def _qa_member(self, scene: _SceneInfo) -> str | None:
        """The ``_QA_PIXEL.TIF`` member name in the scene tar, or ``None``."""
        with tarfile.open(scene.path, "r") as tar:
            return next((n for n in tar.getnames() if n.upper().endswith("_QA_PIXEL.TIF")), None)

    def _read_qa(self, scene: _SceneInfo, cell: GridCell) -> BandRead | None:
        """Read the scene's ``QA_PIXEL`` bit-flag band over the cell footprint (no TOA).

        Windowed to the cell neighbourhood (see :func:`cell_window`); ``None`` if the band
        is absent or the cell does not intersect the scene.
        """
        member = self._qa_member(scene)
        if member is None:
            return None
        with tarfile.open(scene.path, "r") as tar, tempfile.TemporaryDirectory() as tmp:
            tar.extract(member, path=tmp)
            with rasterio.open(Path(tmp) / member) as ds:
                window = cell_window(ds, cell)
                if window is None:
                    return None
                return BandRead(
                    values=ds.read(1, window=window).astype(np.float64),
                    transform=ds.window_transform(window),
                    crs=str(ds.crs),
                )

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the ``QA_PIXEL`` bit-flag on the cell grid (NN; L9→L8 fallback)."""
        if day is None:
            return create_placeholder(n_bands=1, shape=cell.shape)
        scenes = self._select_scenes(day)
        if not scenes:
            return create_placeholder(n_bands=1, shape=cell.shape)

        # QA_PIXEL has no DN==0 fill semantics; coalesce/mosaic mirror the optical path
        # via the same tile grouping, but read through _read_qa (categorical NN).
        by_tile: dict[str, list[_SceneInfo]] = {}
        for scene in scenes:
            by_tile.setdefault(scene.pathrow, []).append(scene)

        tile_reads: list[BandRead] = []
        for tile_scenes in by_tile.values():
            ordered = sorted(tile_scenes, key=lambda s: s.proc, reverse=True)
            reads = [r for s in ordered if (r := self._read_qa(s, cell))]
            if reads:
                tile_reads.append(reads[0])  # latest processing time wins

        if not tile_reads:
            return create_placeholder(n_bands=1, shape=cell.shape)

        # Mixed-UTM-zone safe (same as the optical path): mosaic within each zone, reproject
        # to the cell grid, first-valid-combine. ``mosaic_tiles`` alone raises ``CRS
        # mismatch`` on a cross-zone day (042024=32612 alongside 043/044=32611).
        combined = mosaic_to_cell(tile_reads, cell, categorical=True)
        logger.info("landsat_cloud_fetch", cell_id=cell.cell_id, day=day.isoformat())
        return combined[np.newaxis, :, :].astype(np.float32)
