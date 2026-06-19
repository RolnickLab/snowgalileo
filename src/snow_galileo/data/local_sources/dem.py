"""Copernicus DEM adapter — the static ``[DEM, slope, aspect]`` terrain bands (TASK-007).

Replaces the DEM/slope/aspect placeholders with real Copernicus GLO-30 derivatives
on the cell grid, reproducing GEE ``ee.Terrain`` + ``create_ee_image`` export.

**Two-step order (non-negotiable, matching GEE).**

1. **Compute terrain in the DEM's native EPSG:4326 frame.** The clipped GLO-30
   tiles are geographic with *anisotropic* degree spacing (longitude steps are
   thinned poleward). ``ee.Terrain.slope``/``aspect`` run a 3×3 Horn kernel using
   each pixel's **true ground metres** — so we convert the degree spacing to metres
   with the latitude-correct factors (``dy = yres·M_PER_DEG``,
   ``dx = xres·M_PER_DEG·cos(lat)``) *before* the kernel. Running Horn on a raw
   degree grid (unit ``1°≈1`` spacing) inflates gradients ×111 000 and pins every
   slope at ≈90° — the bug the degenerate guard in the test catches.

2. **Resample DEM + slope + aspect to the cell's EPSG:32611 grid — nearest.** GEE
   produced these at the native ~30 m scale then upsampled to the 10 m export grid;
   **nearest** replicates that pixel reuse (validated: DEM median 0.0 m, slope median
   ≤1.2°, aspect ≤10° across the six Phase-0 reference patches — bilinear roughly
   doubles the slope error). This is a *deliberate* deviation from
   :func:`~snow_galileo.data.local_sources.base.reproject_to_cell`'s bilinear-for-continuous
   rule, justified by GEE parity (decision recorded 2026-06-04, PARITY_SPIKE_NOTES.md).

Terrain is **never** computed in a projected grid: that would diverge from
``ee.Terrain``'s native-frame result regardless of the final cell CRS.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from affine import Affine
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box

from snow_galileo.data.config import NO_DATA_VALUE
from snow_galileo.data.local_sources.base import GridCell, LocalSourceAdapter

logger = structlog.get_logger(__name__)

#: DEM source CRS (the clipped GLO-30 tiles are geographic).
_SOURCE_CRS: str = "EPSG:4326"

#: Metres per degree of latitude on the WGS84 authalic sphere GEE uses for
#: ``ee.Terrain`` metric spacing (``2πR/360`` with ``R = 6378137`` m).
_M_PER_DEG: float = 2.0 * math.pi * 6_378_137.0 / 360.0

#: Margin (degrees) added around the cell when selecting/cropping source tiles so
#: the native-frame Horn kernel has real neighbours right at the cell edge.
_TILE_MARGIN_DEG: float = 0.05

#: Below this elevation a DEM pixel is treated as invalid (Copernicus sets voids to
#: ~0; see ``src/data/config.py`` valid-threshold ``0.0000001``).
_DEM_VALID_MIN: float = 0.0000001


class DemAdapter(LocalSourceAdapter):
    """Static Copernicus DEM adapter emitting ``[DEM, slope, aspect]`` (degrees).

    Terrain derivatives are computed in the native EPSG:4326 frame with
    latitude-correct metric spacing, then nearest-resampled to the cell grid
    (see the module docstring for the parity rationale). Static: ``day`` ignored.

    Args:
        archive_root: The clipped DEM archive root (holds ``*_DEM.tif`` tiles).
    """

    bands_out = ["DEM", "slope", "aspect"]
    spatial_kind = "space"
    native_fill = None

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _tiles(self) -> list[Path]:
        """Return all clipped Copernicus ``DEM`` GeoTIFF tiles in the archive."""
        tiles = sorted(self.archive_root.rglob("*_DEM.tif"))
        if not tiles:
            tiles = sorted(self.archive_root.rglob("*.tif"))
        if not tiles:
            raise FileNotFoundError(f"No DEM tiles under {self.archive_root}")
        return tiles

    def _mosaic_for_cell(self, cell: GridCell) -> tuple[npt.NDArray[np.float64], Affine]:
        """Mosaic the source tiles overlapping the cell (in the 4326 source CRS).

        Args:
            cell: Target grid cell (its UTM polygon is reprojected to 4326 to pick
                tiles, with a margin so the Horn kernel has edge neighbours).

        Returns:
            ``(mosaic, transform)`` — the cropped native-frame DEM covering the cell.
        """
        lon0, lat0, lon1, lat1 = transform_bounds(cell.crs, _SOURCE_CRS, *cell.polygon.bounds)
        m = _TILE_MARGIN_DEG
        want = box(lon0 - m, lat0 - m, lon1 + m, lat1 + m)

        srcs = []
        for tile in self._tiles():
            src = rasterio.open(tile)
            if box(*src.bounds).intersects(want):
                srcs.append(src)
            else:
                src.close()
        if not srcs:
            raise FileNotFoundError(f"No DEM tile intersects cell {cell.cell_id} ({want.bounds}).")
        try:
            mosaic, transform = merge(srcs, bounds=want.bounds)
        finally:
            for src in srcs:
                src.close()
        return mosaic[0].astype(np.float64), transform

    def _slope_aspect(
        self, dem: npt.NDArray[np.float64], transform: Affine
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Horn slope/aspect (degrees) on the native frame with metric spacing.

        Args:
            dem: Native-frame elevation ``(H, W)`` in metres.
            transform: The DEM mosaic's affine (degree spacing).

        Returns:
            ``(slope, aspect)`` in degrees. ``slope ∈ [0, 90]``; ``aspect ∈ [0, 360)``
            with 0=N, 90=E, 180=S, 270=W (the ``ee.Terrain.aspect`` convention).
        """
        n_rows, n_cols = dem.shape
        x_res_deg, y_res_deg = transform.a, -transform.e
        # Latitude at each row centre (transform.e is negative — north-up).
        row_lats = transform.f + (np.arange(n_rows) + 0.5) * transform.e
        d_y = y_res_deg * _M_PER_DEG
        d_x = x_res_deg * _M_PER_DEG * np.cos(np.deg2rad(row_lats))  # per-row (H,)

        def shift(arr: npt.NDArray[np.float64], di: int, dj: int) -> npt.NDArray[np.float64]:
            padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
            return padded[1 + di : 1 + di + n_rows, 1 + dj : 1 + dj + n_cols]

        a, b, c = shift(dem, -1, -1), shift(dem, -1, 0), shift(dem, -1, 1)
        d, f = shift(dem, 0, -1), shift(dem, 0, 1)
        g, h, i = shift(dem, 1, -1), shift(dem, 1, 0), shift(dem, 1, 1)

        dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * d_x[:, np.newaxis])
        dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * d_y)

        slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
        # ee.Terrain.aspect: clockwise from north. arctan2(dz_dy, -dz_dx) is the
        # mathematical (CCW-from-east) direction; (450 - that) mod 360 rotates it.
        aspect = np.mod(450.0 - np.degrees(np.arctan2(dz_dy, -dz_dx)), 360.0)
        return slope, aspect

    def _resample(
        self, band: npt.NDArray[np.float64], src_transform: Affine, cell: GridCell
    ) -> npt.NDArray[np.float64]:
        """Nearest-resample one native-frame band onto the cell's UTM grid.

        Nearest (not bilinear) is the deliberate GEE-parity choice — see the module
        docstring. nodata is not propagated here: the DEM crop is gap-free over the
        cell footprint, and invalid elevations are floored below.
        """
        dst = np.empty(cell.shape, dtype=np.float64)
        reproject(
            source=band,
            destination=dst,
            src_transform=src_transform,
            src_crs=_SOURCE_CRS,
            dst_transform=cell.transform,
            dst_crs=cell.crs,
            resampling=Resampling.nearest,
        )
        return dst

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return ``[DEM, slope, aspect]`` on the cell grid (``day`` ignored).

        Args:
            cell: Target :class:`GridCell` (supplies the UTM ``crs``/``transform``/``shape``).
            day: Ignored — DEM-derived terrain is a static layer.

        Returns:
            ``(3, H, W)`` ``float32`` array — elevation (m), slope (°), aspect (°)
            on the cell's UTM grid, with sub-threshold/invalid pixels set to ``-9999``.
        """
        dem, transform = self._mosaic_for_cell(cell)
        slope, aspect = self._slope_aspect(dem, transform)

        dem_c = self._resample(dem, transform, cell)
        slope_c = self._resample(slope, transform, cell)
        aspect_c = self._resample(aspect, transform, cell)

        # Mask invalid elevations (Copernicus voids) consistently across all three.
        invalid = ~np.isfinite(dem_c) | (dem_c < _DEM_VALID_MIN)
        for band in (dem_c, slope_c, aspect_c):
            band[invalid | ~np.isfinite(band)] = float(NO_DATA_VALUE)

        out = np.stack([dem_c, slope_c, aspect_c], axis=0).astype(np.float32)
        logger.info(
            "dem_fetch",
            cell_id=cell.cell_id,
            dem_range=(float(np.nanmin(dem_c)), float(np.nanmax(dem_c))),
            slope_median=float(np.median(slope_c[~invalid])) if (~invalid).any() else None,
        )
        return out
