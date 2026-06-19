"""MODIS MOD09GA adapter — the seven ``sur_refl`` bands + ``state_1km`` (TASK-009).

Replaces the MODIS placeholders with real values on the cell grid, reading the
**500 m sinusoidal** science grid and **preserving the native ``-28672`` fill** — the
loader treats an encountered ``-28672`` as the "MODIS data was present" sentinel
(``landsat_eval.py:317,331``); stripping it crashes the loader. A companion
:class:`ModisCloudAdapter` emits the 1 km ``state_1km`` bit-flag for the cloud slot.

**Already-extracted GeoTIFFs.** The clip stage exploded each HDF4 granule into
per-band sinusoidal GeoTIFFs (``MODIS_Grid_500m_2D__sur_refl_bNN_1.tif``,
``MODIS_Grid_1km_2D__state_1km_1.tif``), so no HDF4 driver is needed — rasterio reads
them directly. Each grid carries its own resolution/transform; we never hardcode a
pixel count.

**Resample = NEAREST.** The 500 m grid is far coarser than the 10 m cell, so GEE
upsamples it as a constant block per MODIS pixel. Nearest reproduces GEE **bit-exactly**
(median 0 across the reference window; bilinear smears ~322 DN — PARITY_SPIKE_NOTES §8).
Nearest also cannot blend a valid reflectance with ``-28672``, so the edge-bleed risk
the original spec guarded against simply cannot occur. Routed through
:func:`~src.data.local_sources.base.reproject_to_cell` with ``categorical=True`` (its
nearest path), which propagates the source nodata into the output.

We do **not** apply the MODIS scale factor — the integer-like domain must match the
downstream normalization constants. Cross-tile mosaic-before-crop is implemented for
cells spanning a tile seam (this AOI is single-tile ``h10v03``, but the contract holds).
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from affine import Affine
from rasterio.merge import merge

from src.data.config import MODIS_FILL_VALUE
from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
    reproject_to_cell,
)

logger = structlog.get_logger(__name__)

_SUR_REFL_BANDS: list[str] = [f"sur_refl_b0{i}" for i in range(1, 8)]

#: Granule folder name → MOD09GA.A{YYYY}{DOY}.{tile}.{coll}.{prodtime}. We match by
#: acquisition date (``A{YYYY}{DOY}``); a day may have several tiles (mosaic them).
_GRANULE_RE = re.compile(r"MOD09GA\.A(?P<year>\d{4})(?P<doy>\d{3})\.")


def _granule_tag(day: datetime.date) -> str:
    """Return the ``A{YYYY}{DOY}`` acquisition tag for ``day``."""
    return f"A{day.year:04d}{day.timetuple().tm_yday:03d}"


class _ModisBase(LocalSourceAdapter):
    """Shared MODIS read/mosaic/reproject machinery (sinusoidal → cell grid, NN)."""

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _granule_dirs(self, day: datetime.date) -> list[Path]:
        """All granule folders acquired on ``day`` (one per MODIS tile)."""
        tag = _granule_tag(day)
        return sorted(p for p in self.archive_root.glob(f"MOD09GA.{tag}.*") if p.is_dir())

    def _band_tifs(self, day: datetime.date, filename: str) -> list[Path]:
        """The per-tile GeoTIFFs for ``filename`` acquired on ``day`` (may be empty)."""
        return [p / filename for p in self._granule_dirs(day) if (p / filename).exists()]

    def _mosaic(
        self, tifs: list[Path]
    ) -> tuple[npt.NDArray[np.float64], Affine, str, float]:
        """Mosaic per-tile GeoTIFFs in their native sinusoidal CRS.

        Returns:
            ``(array, transform, crs_wkt, src_nodata)`` — the single-tile array if only
            one tile, else the merged mosaic. ``src_nodata`` is read from the source.
        """
        srcs = [rasterio.open(t) for t in tifs]
        try:
            src_nodata = srcs[0].nodata if srcs[0].nodata is not None else MODIS_FILL_VALUE
            crs_wkt = srcs[0].crs.to_wkt()
            if len(srcs) == 1:
                return srcs[0].read(1).astype(np.float64), srcs[0].transform, crs_wkt, float(src_nodata)
            mosaic, transform = merge(srcs, nodata=src_nodata)
            return mosaic[0].astype(np.float64), transform, crs_wkt, float(src_nodata)
        finally:
            for s in srcs:
                s.close()


class ModisAdapter(_ModisBase):
    """MODIS ``sur_refl_b01..b07`` adapter (``low`` tier, ``-28672`` fill preserved).

    Args:
        archive_root: The clipped MODIS archive root (holds ``MOD09GA.A*`` granule dirs).
    """

    bands_out = _SUR_REFL_BANDS
    spatial_kind = "low"
    native_fill = MODIS_FILL_VALUE

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the seven sur_refl bands on the cell grid (``-28672`` preserved).

        A missing day (no granule folder) or any missing band returns the all-``-9999``
        placeholder for the whole stack.
        """
        if day is None or not self._granule_dirs(day):
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.floating]] = []
        for band in self.bands_out:
            tifs = self._band_tifs(day, f"MODIS_Grid_500m_2D__{band}_1.tif")
            if not tifs:
                return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
            arr, transform, crs_wkt, src_nodata = self._mosaic(tifs)
            # Nearest (categorical path): bit-exact GEE parity; -28672 propagated, never
            # blended. restore_fill keeps the native sentinel rather than -9999.
            reprojected = reproject_to_cell(
                source=arr[np.newaxis, :, :],
                src_transform=transform,
                src_crs=crs_wkt,
                cell=cell,
                categorical=True,
                src_nodata=src_nodata,
                restore_fill=MODIS_FILL_VALUE,
            )
            bands.append(reprojected[0])

        logger.info("modis_fetch", cell_id=cell.cell_id, day=day.isoformat(), bands=len(bands))
        return np.stack(bands, axis=0).astype(np.float32)


class ModisCloudAdapter(_ModisBase):
    """MODIS ``state_1km`` cloud-flag adapter (1 km grid, categorical/NN, cloud slot).

    Args:
        archive_root: The clipped MODIS archive root.
    """

    bands_out = ["state_1km"]
    spatial_kind = "time"
    native_fill = None

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the ``state_1km`` bit-flag on the cell grid (NN; ``day`` resolved)."""
        if day is None or not self._granule_dirs(day):
            return create_placeholder(n_bands=1, shape=cell.shape)

        tifs = self._band_tifs(day, "MODIS_Grid_1km_2D__state_1km_1.tif")
        if not tifs:
            return create_placeholder(n_bands=1, shape=cell.shape)
        arr, transform, crs_wkt, src_nodata = self._mosaic(tifs)
        reprojected = reproject_to_cell(
            source=arr[np.newaxis, :, :],
            src_transform=transform,
            src_crs=crs_wkt,
            cell=cell,
            categorical=True,  # bit-flag — never interpolate
            src_nodata=src_nodata,
        )
        logger.info("modis_cloud_fetch", cell_id=cell.cell_id, day=day.isoformat())
        return reprojected.astype(np.float32)
