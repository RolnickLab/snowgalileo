"""Sentinel-2 L1C adapter — ``[B2,B3,B4,B8,B11,B12]`` harmonized reflectance (TASK-013).

Replaces the S2 placeholders with the value domain GEE's ``COPERNICUS/S2_HARMONIZED``
produces, reading the clipped L1C SAFE ``.zip`` granules (JP2 bands). For baseline ≥ N0400
(all 116 archive granules are N0511) GEE subtracts a **−1000 DN harmonization offset**;
we reproduce it exactly (the 10 m bands match the reference patches bit-for-bit).

    harmonized_DN = raw_DN − 1000   (baseline ≥ 04.00 / N0511)

``DN == 0`` is the L1C no-data and maps to ``-9999``. The downstream normalization divides
by 10000; valid threshold is ``>= -1`` (post-offset).

**Resample = NEAREST (bit-exact parity).** GEE's ``export_from_csv_utm`` upsamples to the
10 m cell grid as constant blocks. The 10 m bands (B2/B3/B4/B8) are already on a 10 m grid;
the **20 m SWIR** bands (B11/B12) are coarser than the 10 m cell, so nearest reproduces GEE
(signed-median 0 on the reference patches) while bilinear smears ~20 DN. Same coarse-source
rule as MODIS/Landsat. (Supersedes the spec/spike "bilinear for reflectance".)

**Single UTM zone (no mixed-zone trap).** Unlike Landsat's WRS-2 scenes, Sentinel-2 MGRS
tiles encode the zone: every archive tile is ``T11U**`` = EPSG:32611 (the cell grid's
zone). We still read each band's native CRS and pass it to ``reproject_to_cell``
(zone-agnostic), so a future cross-zone tile would Just Work.

**Same-(tile, date) coalesce + cross-tile mosaic** mirror Landsat (shared via
``_scene_ops``): a tile may be delivered as several products on one date (e.g. ``20250420
T11UNT`` R113 vs R070); gather all, first-valid-wins per pixel (latest-processing-time
order), then mosaic neighbouring tiles before the reproject. Valid-pixel union, not an
average.

``QA60`` is **out of scope here** (TASK-013c): N0511 SAFEs ship no ``QA60.jp2`` (replaced
by ``MSK_CLASSI``), and a naive repack does not reproduce GEE's backfilled ``QA60``. It
stays the ``-9999`` placeholder until TASK-013c reverse-engineers the mapping.
"""

from __future__ import annotations

import datetime
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog

from src.data.config import NO_DATA_VALUE
from src.data.local_sources._scene_ops import BandRead, coalesce_tile, mosaic_tiles
from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
    reproject_to_cell,
)

logger = structlog.get_logger(__name__)

#: Cube band → JP2 filename suffix (S2 JP2s use zero-padded band numbers).
_S2_BANDS: dict[str, str] = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B8": "B08",
    "B11": "B11",
    "B12": "B12",
}

#: The harmonization offset GEE subtracts from baseline ≥ N0400 L1C DN.
_HARMONIZE_OFFSET_DN: int = 1000

#: Baseline at/above which the −1000 DN offset applies (N0400 → 04.00).
_HARMONIZE_MIN_BASELINE: float = 4.00

#: The valid floor after harmonization; ``DN == 0`` (raw no-data) is excluded separately.
_VALID_MIN: float = -1.0

#: Granule SAFE-zip stem: ``S2[ABC]_MSIL1C_{acq}T..._N{baseline}_R{orbit}_{tile}_{proc}T...``.
#: Sentinel-2C (operational 2025) shares the L1C product structure; the unit letter is
#: parity-irrelevant — gating on ``S2[AB]`` only would silently drop every S2C granule.
_GRANULE_RE = re.compile(
    r"^S2[ABC]_MSIL1C_(?P<acq>\d{8})T\d{6}_N(?P<baseline>\d{4})_"
    r"R\d{3}_(?P<tile>T\d{2}\w{3})_(?P<proc>\d{8})T\d{6}"
)


@dataclass(frozen=True)
class _GranuleInfo:
    """Parsed identity of one clipped S2 SAFE granule."""

    path: Path
    tile: str
    acq: datetime.date
    proc: datetime.date
    baseline: float  # e.g. 5.11 for N0511


def _parse_granule(zip_path: Path) -> _GranuleInfo | None:
    """Parse a granule zip name; return ``None`` if it is not an L1C product."""
    # ``.SAFE.zip`` and ``.zip`` both appear in the archive; match on the basename.
    m = _GRANULE_RE.match(zip_path.name)
    if m is None:
        return None
    return _GranuleInfo(
        path=zip_path,
        tile=m.group("tile"),
        acq=datetime.datetime.strptime(m.group("acq"), "%Y%m%d").date(),
        proc=datetime.datetime.strptime(m.group("proc"), "%Y%m%d").date(),
        baseline=int(m.group("baseline")) / 100.0,
    )


def _read_baseline(zf: zipfile.ZipFile, name_baseline: float) -> float:
    """Read ``<PROCESSING_BASELINE>`` from ``MTD_MSIL1C.xml``; fall back to the name token.

    The authoritative source is the manifest (REVIEW_AUDIT #6); the name's ``N0511`` token
    is the fallback when the tag is absent.
    """
    mtd = next((n for n in zf.namelist() if n.endswith("MTD_MSIL1C.xml")), None)
    if mtd is None:
        return name_baseline
    try:
        root = ET.fromstring(zf.read(mtd))
        for el in root.iter():
            if el.tag.endswith("PROCESSING_BASELINE") and el.text:
                return float(el.text)
    except (ET.ParseError, ValueError):
        return name_baseline
    return name_baseline


class S2Adapter(LocalSourceAdapter):
    """Sentinel-2 ``[B2,B3,B4,B8,B11,B12]`` harmonized-reflectance adapter (``high`` tier).

    Args:
        archive_root: The clipped S2 archive root (holds ``S2?_MSIL1C_*.zip`` granules).
    """

    bands_out = list(_S2_BANDS)
    spatial_kind = "high"
    native_fill = None

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _granules_for_day(self, day: datetime.date) -> list[_GranuleInfo]:
        """All clipped granules acquired on ``day`` (any tile)."""
        granules = [_parse_granule(p) for p in sorted(self.archive_root.glob("*.zip"))]
        return [g for g in granules if g is not None and g.acq == day]

    def _read_band(self, granule: _GranuleInfo, band_suffix: str) -> BandRead | None:
        """Read one JP2 band, apply the −1000 DN harmonization, return a :class:`BandRead`.

        Returns ``None`` if the granule lacks the band JP2. ``DN == 0`` (L1C no-data) and
        sub-floor values become ``-9999``.
        """
        with zipfile.ZipFile(granule.path) as zf:
            jp2 = next(
                (n for n in zf.namelist() if n.endswith(f"_{band_suffix}.jp2") and "/IMG_DATA/" in n),
                None,
            )
            if jp2 is None:
                return None
            baseline = _read_baseline(zf, granule.baseline)

        with rasterio.open(f"/vsizip/{granule.path}/{jp2}") as ds:
            dn = ds.read(1).astype(np.float64)
            transform = ds.transform
            crs = str(ds.crs)

        raw_zero = dn == 0
        if baseline >= _HARMONIZE_MIN_BASELINE:
            dn = dn - _HARMONIZE_OFFSET_DN
        # Raw no-data (pre-offset 0) and any sub-floor value are fill.
        dn[raw_zero | (dn < _VALID_MIN)] = float(NO_DATA_VALUE)
        return BandRead(values=dn, transform=transform, crs=crs)

    def _band_on_cell(
        self, granules: list[_GranuleInfo], band_suffix: str, cell: GridCell
    ) -> npt.NDArray[np.float32] | None:
        """Coalesce per tile, mosaic across tiles, reproject one band to the cell grid.

        Returns ``None`` if no granule yields a readable band.
        """
        by_tile: dict[str, list[_GranuleInfo]] = {}
        for granule in granules:
            by_tile.setdefault(granule.tile, []).append(granule)

        tile_reads: list[BandRead] = []
        for tile_granules in by_tile.values():
            ordered = sorted(tile_granules, key=lambda g: g.proc, reverse=True)
            reads = [r for g in ordered if (r := self._read_band(g, band_suffix))]
            if reads:
                tile_reads.append(coalesce_tile(reads))

        if not tile_reads:
            return None

        merged, transform, crs = mosaic_tiles(tile_reads)
        # NEAREST: GEE upsamples to the 10 m cell as constant blocks (bit-exact); the 20 m
        # SWIR bands would smear under bilinear. Nearest also never blends valid + -9999.
        reprojected = reproject_to_cell(
            source=merged[np.newaxis, :, :],
            src_transform=transform,
            src_crs=crs,
            cell=cell,
            categorical=True,
            src_nodata=float(NO_DATA_VALUE),
        )
        return reprojected[0]

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the six harmonized reflectance bands on the cell grid (``-9999`` missing)."""
        if day is None:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
        granules = self._granules_for_day(day)
        if not granules:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.float32]] = []
        for band_name in self.bands_out:
            arr = self._band_on_cell(granules, _S2_BANDS[band_name], cell)
            if arr is None:
                return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
            bands.append(arr)

        logger.info(
            "s2_fetch",
            cell_id=cell.cell_id,
            day=day.isoformat(),
            granules=len(granules),
            tiles=sorted({g.tile for g in granules}),
        )
        return np.stack(bands, axis=0).astype(np.float32)
