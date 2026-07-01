"""VIIRS VNP09GA adapter — fine ``[I1, I3]`` + coarse ``[M5, M7, M10, M11]`` (TASK-010).

VIIRS contributes two band groups, each its own adapter sharing the sinusoidal
read/mosaic/reproject machinery:

- **fine** ``[I1, I3]`` from the 500 m grid → ``space_time_low_res_x`` (``low`` tier).
- **coarse** ``[M5, M7, M10, M11]`` from the 1 km grid → ``time_x`` (``time`` tier),
  emitted as a **per-pixel raster** ``(4, H, W)``. The downstream loader does the
  spatial mean into ``time_x``; the adapter must **not** pre-average (that would break
  ``time_x`` and bias the mean over the diagonal-band clip's nodata).

**Scale = x0.0001 (reflectance).** Unlike MODIS (kept raw DN), GEE exports VNP09GA as
reflectance — the normalizer ``(x+0.795)/0.805`` confirms the domain. Validated
bit-exact vs the Phase-0 reference patch (DN/ref ratio exactly 10000). Valid pixels are
scaled; the native ``-28672`` fill is restored **after** scaling (never scaled itself),
preserving the loader sentinel exactly as for MODIS.

**Resample = NEAREST.** The 500 m / 1 km grids are coarser than the 10 m cell; nearest
reproduces GEE bit-exactly and cannot blend a valid reflectance with ``-28672``
(PARITY_SPIKE_NOTES §9). The clip stage already extracted per-band sinusoidal GeoTIFFs,
so rasterio reads them directly (no HDF5 driver needed).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from affine import Affine
from rasterio.merge import merge

from snow_galileo.data.config import MODIS_FILL_VALUE
from snow_galileo.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
    reproject_to_cell,
)

logger = structlog.get_logger(__name__)

#: VNP09GA stored DN → reflectance scale (GEE export domain).
_REFLECTANCE_SCALE: float = 0.0001


class _ViirsBase(LocalSourceAdapter):
    """Shared VIIRS read/mosaic/reproject machinery (sinusoidal → cell grid, NN, scaled).

    Subclasses declare ``bands_out``, ``spatial_kind``, ``native_fill``, the source grid
    label (``_grid``: ``500m``/``1km``), and the per-band source-filename template.
    """

    _grid: str

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _granule_dirs(self, day: datetime.date) -> list[Path]:
        tag = f"A{day.year:04d}{day.timetuple().tm_yday:03d}"
        return sorted(p for p in self.archive_root.glob(f"VNP09GA.{tag}.*") if p.is_dir())

    def _band_filename(self, band: str) -> str:
        return f"VIIRS_Grid_{self._grid}_2D__SurfReflect_{band}_1.tif"

    def _band_tifs(self, day: datetime.date, band: str) -> list[Path]:
        fn = self._band_filename(band)
        return [p / fn for p in self._granule_dirs(day) if (p / fn).exists()]

    def _mosaic(self, tifs: list[Path]) -> tuple[npt.NDArray[np.float64], Affine, str, float]:
        srcs = [rasterio.open(t) for t in tifs]
        try:
            src_nodata = srcs[0].nodata if srcs[0].nodata is not None else MODIS_FILL_VALUE
            crs_wkt = srcs[0].crs.to_wkt()
            if len(srcs) == 1:
                return (
                    srcs[0].read(1).astype(np.float64),
                    srcs[0].transform,
                    crs_wkt,
                    float(src_nodata),
                )
            mosaic, transform = merge(srcs, nodata=src_nodata)
            return mosaic[0].astype(np.float64), transform, crs_wkt, float(src_nodata)
        finally:
            for s in srcs:
                s.close()

    def _scaled_band_on_cell(
        self, band: str, day: datetime.date, cell: GridCell
    ) -> npt.NDArray[np.floating] | None:
        """Return one scaled reflectance band on the cell grid, or ``None`` if absent.

        Scales valid pixels by ``0.0001`` and restores ``-28672`` **after** scaling (the
        fill is never scaled), then nearest-reprojects onto the cell grid.
        """
        tifs = self._band_tifs(day, band)
        if not tifs:
            return None
        arr, transform, crs_wkt, src_nodata = self._mosaic(tifs)
        fill_mask = arr == src_nodata
        scaled = arr * _REFLECTANCE_SCALE
        scaled[fill_mask] = MODIS_FILL_VALUE  # restore the unscaled native sentinel
        reprojected = reproject_to_cell(
            source=scaled[np.newaxis, :, :],
            src_transform=transform,
            src_crs=crs_wkt,
            cell=cell,
            categorical=True,  # nearest — bit-exact, no fill blending
            src_nodata=MODIS_FILL_VALUE,
            restore_fill=MODIS_FILL_VALUE,
        )
        return reprojected[0]

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return this VIIRS group's scaled bands on the cell grid (``-28672`` preserved)."""
        if day is None or not self._granule_dirs(day):
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.floating]] = []
        for band in self.bands_out:
            on_cell = self._scaled_band_on_cell(band, day, cell)
            if on_cell is None:
                return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
            bands.append(on_cell)

        logger.info(
            "viirs_fetch",
            cell_id=cell.cell_id,
            day=day.isoformat(),
            grid=self._grid,
            bands=len(bands),
        )
        return np.stack(bands, axis=0).astype(np.float32)


class ViirsFineAdapter(_ViirsBase):
    """VIIRS fine ``[I1, I3]`` adapter (500 m grid, ``low`` tier, ``-28672`` preserved).

    Args:
        archive_root: The clipped VIIRS archive root (holds ``VNP09GA.A*`` granule dirs).
    """

    bands_out = ["I1", "I3"]
    spatial_kind = "low"
    native_fill = MODIS_FILL_VALUE
    _grid = "500m"


class ViirsCoarseAdapter(_ViirsBase):
    """VIIRS coarse ``[M5, M7, M10, M11]`` adapter (1 km grid, ``time`` tier, per-pixel).

    Emitted as a per-pixel ``(4, H, W)`` raster — the loader does the spatial mean into
    ``time_x``; this adapter never pre-averages.

    Args:
        archive_root: The clipped VIIRS archive root.
    """

    bands_out = ["M5", "M7", "M10", "M11"]
    spatial_kind = "time"
    native_fill = MODIS_FILL_VALUE
    _grid = "1km"
