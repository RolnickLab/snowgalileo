"""Sentinel-1 GRD adapter — ``[VV, VH, angle]`` σ⁰ dB + incidence angle (TASK-014).

Replaces the S1 placeholders (the head of the HIGH group, dynamic-block offsets
0–2) with the value domain GEE's ``COPERNICUS/S1_GRD`` produces: calibrated,
terrain-corrected backscatter in **dB** for VV/VH, plus the ellipsoid **incidence
angle** in degrees, on the cell grid. The project edge mask invalidates VV/VH
pixels ``< -30.0`` dB; the angle band (pure geometry) is never masked.

**Why this adapter is pure-raster despite S1 being the heaviest source.** S1 is not
clipped — it is *processed* from the raw granules. The expensive ESA SNAP chain —
Apply-Orbit → ThermalNoise → Border-Noise → Calibration(σ⁰) → Terrain-Correction(
EPSG:32611, + ellipsoid incidence) → post-TC AOI Subset — runs **once per raw granule,
offline**, into a cached 3-band dB+angle GeoTIFF
(:mod:`src.data.local_sources.s1_snap`). This
adapter reads that cache and runs the same coalesce → mosaic → reproject path as
the S2/Landsat scene adapters, with **no SNAP dependency** — so it is fast and
unit-testable. Build the cache before exporting cubes:

    uv run python -m src.data.local_sources.s1_snap

**`angle` = ellipsoid incidence (~43.6°), NOT local incidence.** GEE ``S1_GRD``'s
``angle`` band varies only with range (near-constant across a small patch); the
SNAP graph emits it via ``saveIncidenceAngleFromEllipsoid``. Local incidence
(which swings with terrain) would not match the reference patches.

**dB is applied here, not in SNAP.** The SNAP cache stores **linear** σ⁰ + the
angle band (the graph has no ``LinearToFromdB`` node — scoping it to the σ⁰ bands
would drop the angle band from the output). This adapter converts VV/VH to dB with
``10·log10`` (identical math to SNAP's ``LinearToFromdB``); the angle band stays in
degrees.

**Band order is pinned by index, not name.** SNAP's BigTIFF writer persists no band
descriptions, so the adapter maps by the graph's fixed output order: band 1 =
Sigma0_VH, band 2 = Sigma0_VV, band 3 = incidence angle (SNAP emits VH before VV).

**Bilinear/continuous reproject.** Unlike the coarse-source NEAREST rule for
S2-SWIR/MODIS, S1 is already terrain-corrected onto the 10 m EPSG:32611 grid in
the cache; the reproject here is a grid-snap/crop to the exact cell transform, and
σ⁰ dB + angle are continuous quantities.

**Same-date coalesce + cross-granule mosaic** mirror S2 (shared via
``_scene_ops``): one satellite pass delivers consecutive slices on a date (e.g.
``...T013724`` and ``...T013749``); gather all, first-valid-wins per pixel, then
mosaic before the reproject. Valid-pixel union, not an average.

**Sentinel-1C.** The Bow Valley archive is all ``S1C_*`` (launched Dec 2024). The
SNAP cache step reads it natively; ``xarray-sentinel``/``sarsen`` cannot (the
``s1[ab]`` regex bug). The granule regex here accepts ``S1[A-Z]`` so a future
S1A/B granule also parses.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog

from src.data.config import NO_DATA_VALUE
from src.data.local_sources._scene_ops import BandRead, cell_window, mosaic_tiles
from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
    reproject_to_cell,
)

logger = structlog.get_logger(__name__)

#: SAR backscatter below this (dB) is edge/no-data (GEE S1_GRD convention). Applied
#: to VV/VH only — the incidence-angle band is geometry and is never edge-masked.
S1_EDGE_MASK_DB: float = -30.0

#: Cube band → 1-based band index in the SNAP cache tif. **Pinned by the graph's
#: output order** (SNAP emits VH before VV; angle last) because the BigTIFF writer
#: persists no band descriptions. The parity test asserts the angle domain, which
#: would catch any future SNAP reordering.
_BAND_INDEX: dict[str, int] = {
    "VH": 1,
    "VV": 2,
    "angle": 3,
}

#: Bands stored as **linear** σ⁰ in the cache that the adapter converts to dB
#: (``10·log10``) — the backscatter bands. The angle band is left in degrees.
_DB_BANDS: frozenset[str] = frozenset({"VV", "VH"})

#: Cache-tif stem (per-granule, AOI-wide):
#: ``s1_grd_S1[A-Z]_IW_GRDH_..._{acq}T..._{end}T..._{orbit}_{batch}_{uid}``.
#: ``uid`` is the product's unique-id hex; sorting on it gives a deterministic, stable
#: coalesce order (proxy for processing recency — later products sort higher). One tif
#: per granule covers the whole AOI; the adapter windows it per cell (see s1_snap.py).
_GRANULE_RE = re.compile(
    r"^s1_grd_S1[A-Z]_IW_GRDH_\w+?_(?P<acq>\d{8})T\d{6}_"
    r"\d{8}T\d{6}_\d{6}_[0-9A-F]{6}_(?P<uid>[0-9A-F]{4})$"
)


@dataclass(frozen=True)
class _GranuleInfo:
    """Parsed identity of one cached per-granule AOI-wide S1 dB+angle GeoTIFF."""

    path: Path
    acq: datetime.date
    uid: str  # product unique-id hex; deterministic coalesce order key


def _parse_granule(tif_path: Path) -> _GranuleInfo | None:
    """Parse a cache-tif name; return ``None`` if it is not an S1 cache product."""
    m = _GRANULE_RE.match(tif_path.stem)
    if m is None:
        return None
    return _GranuleInfo(
        path=tif_path,
        acq=datetime.datetime.strptime(m.group("acq"), "%Y%m%d").date(),
        uid=m.group("uid"),
    )


class S1Adapter(LocalSourceAdapter):
    """Sentinel-1 ``[VV, VH, angle]`` σ⁰-dB + incidence adapter (``high`` tier).

    Reads the SNAP dB+angle cache (:mod:`src.data.local_sources.s1_snap`), coalesces
    same-date granules, mosaics, edge-masks VV/VH ``< -30`` dB, and reprojects onto
    the cell grid. A missing acquisition (the common case — S1 covers ~16 dates)
    returns the ``-9999`` placeholder.

    Args:
        cache_root: Directory holding the per-granule ``s1_grd_*.tif`` AOI-wide SNAP
            cache (NOT the raw SAFE archive — run ``build_s1_cache`` offline first).
    """

    bands_out = ["VV", "VH", "angle"]
    spatial_kind = "high"
    native_fill = None

    def __init__(self, *, cache_root: Path) -> None:
        self.cache_root = cache_root

    def _cached_for(self, day: datetime.date) -> list[_GranuleInfo]:
        """Cached AOI-wide granules acquired on ``day``.

        The cache is keyed by granule (one AOI-wide tif each); every same-day granule
        is read, then coalesced/mosaicked and windowed to the cell by
        :meth:`_band_on_cell` (``reproject_to_cell`` crops the AOI tif to the cell).
        """
        granules = [_parse_granule(p) for p in sorted(self.cache_root.glob("s1_grd_*.tif"))]
        return [g for g in granules if g is not None and g.acq == day]

    def _read_band(self, granule: _GranuleInfo, band_name: str, cell: GridCell) -> BandRead | None:
        """Read one band from a cache tif: linear→dB for VV/VH, edge-mask, nodata-clean.

        Reads **only the cell's windowed footprint** (via :func:`cell_window`), never the
        full granule — a per-granule SNAP swath can be 14466×9637 (~1.1 GB/band as float64),
        and a full read ×3 bands ×8 parallel workers exhausts memory and OOM-kills a worker
        (surfacing as ``BrokenProcessPool``). The window is the same neighbourhood-padded
        read the S2/Landsat adapters use, so the subsequent reproject is bit-identical.

        VV/VH are stored as linear σ⁰; they are converted to dB (``10·log10``) and
        pixels ``< -30`` dB (or with non-positive σ⁰, where log is undefined) become
        ``-9999``. The ``angle`` band is left in degrees and never edge-masked (it is
        geometry), only nodata-cleaned. Returns ``None`` if the band index is absent **or**
        the cell does not intersect this granule (``cell_window`` → ``None``).
        """
        idx = _BAND_INDEX[band_name]
        with rasterio.open(granule.path) as ds:
            if idx > ds.count:
                return None
            window = cell_window(ds, cell)
            if window is None:
                return None
            values = ds.read(idx, window=window).astype(np.float64)
            transform = ds.window_transform(window)
            crs = str(ds.crs)

        if band_name in _DB_BANDS:
            # Linear σ⁰ → dB; σ⁰ ≤ 0 (incl. SNAP's 0 fill) is invalid (log undefined).
            positive = values > 0
            db = np.full(values.shape, float(NO_DATA_VALUE), dtype=np.float64)
            db[positive] = 10.0 * np.log10(values[positive])
            db[positive & (db < S1_EDGE_MASK_DB)] = float(NO_DATA_VALUE)
            values = db
        else:
            values[~np.isfinite(values)] = float(NO_DATA_VALUE)
        return BandRead(values=values, transform=transform, crs=crs)

    def _band_on_cell(
        self, granules: list[_GranuleInfo], band_name: str, cell: GridCell
    ) -> npt.NDArray[np.float32] | None:
        """Mosaic same-date granules, reproject one band onto the cell grid.

        Returns ``None`` if no granule yields the band.
        """
        # Same-date S1 granules are distinct per-granule SNAP outputs: same zone
        # (EPSG:32611) and 10 m res, but **different footprints/extents** (adjacent
        # sub-swath segments of one pass — verified: a 393×550 segment alongside a
        # 14466×9637 one on 2025-04-06). They are the *mosaic* case, not the coalesce
        # case — each granule is its own single-read "tile". ``mosaic_tiles`` merges
        # across the differing grids (``rasterio.merge`` method="first": first dataset's
        # valid pixels win, later granules fill only nodata gaps), so ordering
        # latest-uid first preserves the deterministic-winner rule. Routing these
        # through ``coalesce_tile`` (which assumes one shared grid) raised an IndexError
        # on the shape mismatch.
        ordered = sorted(granules, key=lambda g: g.uid, reverse=True)
        reads = [r for g in ordered if (r := self._read_band(g, band_name, cell))]
        if not reads:
            return None

        merged, transform, crs = mosaic_tiles(reads)
        # Bilinear/continuous: S1 is already on the 10 m 32611 grid post-TC; this is a
        # grid-snap/crop to the exact cell transform. dB σ⁰ + angle are continuous.
        reprojected = reproject_to_cell(
            source=merged[np.newaxis, :, :],
            src_transform=transform,
            src_crs=crs,
            cell=cell,
            categorical=False,
            src_nodata=float(NO_DATA_VALUE),
        )
        return reprojected[0]

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the [VV, VH, angle] bands on the cell grid (``-9999`` if missing)."""
        if day is None:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
        granules = self._cached_for(day)
        if not granules:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.float32]] = []
        for band_name in self.bands_out:
            arr = self._band_on_cell(granules, band_name, cell)
            if arr is None:
                return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
            bands.append(arr)

        logger.info(
            "s1_fetch",
            cell_id=cell.cell_id,
            day=day.isoformat(),
            granules=len(granules),
        )
        return np.stack(bands, axis=0).astype(np.float32)
