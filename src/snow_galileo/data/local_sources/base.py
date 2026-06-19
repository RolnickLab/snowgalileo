"""Adapter contract for the local-source pipeline — the interface every later
task depends on.

This module defines, **once**, the shared types and behaviours every modality
adapter (TASK-006…TASK-014) must conform to, following Ports & Adapters: the
business logic (exporter, driver) depends on the :class:`LocalSourceAdapter`
*port*, never on a concrete adapter's I/O.

Contents:
- :class:`GridCell` — the per-cell target-grid triple (CRS, affine transform,
  shape) + the source UTM polygon. **CRS is law**: every adapter reprojects onto
  this exact grid.
- :class:`CellWindow` — a ``(GridCell, [day…])`` pairing for one 8-day inference
  window (the exporter iterates these).
- :class:`LocalSourceAdapter` — the abstract base every adapter implements.
- :func:`reproject_to_cell` — the **shared, nodata-aware** resampler
  (bilinear for continuous, nearest for QA/categorical) all continuous adapters
  inherit, so the edge-bleed guard lives in one place (REVIEW_AUDIT #4).
- :func:`create_placeholder` — the ``-9999`` array a missing acquisition returns.

**Target grid (CRS is law, corrected 2026-06-04).** Cells target
``EPSG:32611`` (UTM 11N) at ``scale=10`` m, ``100×100`` px — matching the GEE
reference patches produced by ``export_from_csv_utm`` (see PLAN §3 Grid+CRS table
and ``docs/agents/KNOWLEDGE.md``). The downstream loader reads neither the tif CRS
nor transform, so this choice is driven purely by AC-27 parity.

**Out of scope here (declared, not implemented).** Same-tile/date coalesce and
cross-tile mosaic-before-crop are part of the *contract* (documented on
:meth:`LocalSourceAdapter.fetch`) but implemented per-adapter in TASK-012/013.
This module ships no real adapter logic.
"""

from __future__ import annotations

import abc
import datetime
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import Polygon, box

from snow_galileo.data.config import (
    DATASET_OUTPUT_HW_HIGH_RES,
    EXPORTED_HEIGHT_WIDTH_METRES,
    MODIS_FILL_VALUE,
    NO_DATA_VALUE,
)

# --- Target-grid constants (CRS is law) ------------------------------------ #

#: Per-cell target CRS — UTM 11N. Matches the GEE reference patches' grid.
CELL_TARGET_CRS: str = "EPSG:32611"

#: Per-cell raster side length in pixels (100 = 1000 m / 10 m).
CELL_TARGET_PX: int = DATASET_OUTPUT_HW_HIGH_RES

#: Per-pixel ground sample distance in metres (the EE ``scale=10``).
CELL_SCALE_M: float = EXPORTED_HEIGHT_WIDTH_METRES / DATASET_OUTPUT_HW_HIGH_RES

#: Sentinels treated as fill by the nodata-aware resampler. ``-9999`` is the
#: universal nodata; ``-28672`` is the native MODIS fill that must survive
#: (NDSI/NDVI assertions at ``landsat_eval.py:317,331`` depend on it).
FILL_SENTINELS: tuple[float, ...] = (float(NO_DATA_VALUE), float(MODIS_FILL_VALUE))

SpatialKind = Literal["high", "med", "low", "time", "space", "static"]


@dataclass(frozen=True)
class GridCell:
    """One 1 km grid cell and the target raster grid every adapter writes onto.

    CRS is law: the ``(crs, transform, shape)`` triple is the *only* grid an
    adapter may produce. :meth:`reproject_to_cell` consumes exactly these.

    Attributes:
        cell_id: Stable identifier (unique within a grid; used for the cube-cache
            shard path ``cube_cache/{cell_id}/…``).
        crs: Target CRS string (always :data:`CELL_TARGET_CRS`).
        transform: North-up affine mapping pixel → :data:`CELL_TARGET_CRS` metres,
            origin at the cell's top-left ``(min_x, max_y)``, ``10`` m pixels.
        shape: ``(height, width)`` in pixels — always
            ``(CELL_TARGET_PX, CELL_TARGET_PX)``.
        polygon: The cell footprint as a shapely polygon in :data:`CELL_TARGET_CRS`
            metres (the UTM bbox the cell was built from).
    """

    cell_id: int
    crs: str
    transform: Affine
    shape: tuple[int, int]
    polygon: Polygon

    @classmethod
    def from_utm_bounds(
        cls,
        *,
        cell_id: int,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        crs: str = CELL_TARGET_CRS,
        px: int = CELL_TARGET_PX,
    ) -> GridCell:
        """Build a :class:`GridCell` from its UTM bounding box.

        The affine transform is north-up with origin at ``(min_x, max_y)`` and
        ``px × px`` pixels sized so the raster spans exactly the bbox — i.e.
        ``(max_x - min_x) / px`` metres per pixel (10 m for a 1 km cell at
        ``px=100``).

        Args:
            cell_id: Stable cell identifier.
            min_x: Western bound (CRS metres).
            min_y: Southern bound (CRS metres).
            max_x: Eastern bound (CRS metres).
            max_y: Northern bound (CRS metres).
            crs: Target CRS (defaults to :data:`CELL_TARGET_CRS`).
            px: Raster side length in pixels (defaults to :data:`CELL_TARGET_PX`).

        Returns:
            The fully-specified :class:`GridCell`.

        Raises:
            ValueError: If the bbox is non-positive in either dimension.
        """
        width_m = max_x - min_x
        height_m = max_y - min_y
        if width_m <= 0 or height_m <= 0:
            raise ValueError(
                f"Degenerate cell bbox for cell {cell_id}: ({min_x}, {min_y}, {max_x}, {max_y})."
            )
        x_res = width_m / px
        y_res = height_m / px
        transform = from_origin(min_x, max_y, x_res, y_res)
        return cls(
            cell_id=cell_id,
            crs=crs,
            transform=transform,
            shape=(px, px),
            polygon=box(min_x, min_y, max_x, max_y),
        )


@dataclass(frozen=True)
class CellWindow:
    """A grid cell paired with the days of one inference window.

    Attributes:
        cell: The target :class:`GridCell`.
        days: The window days in ascending order; the last is the window-end day
            (``len(days) == NUM_TIMESTEPS`` for a full 8-day window).
    """

    cell: GridCell
    days: tuple[datetime.date, ...]

    @property
    def window_end(self) -> datetime.date:
        """The window-end day (the prediction day)."""
        return self.days[-1]


def create_placeholder(
    *,
    n_bands: int,
    shape: tuple[int, int] = (CELL_TARGET_PX, CELL_TARGET_PX),
    fill: float = float(NO_DATA_VALUE),
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Return an all-``fill`` ``(n_bands, H, W)`` array for a missing acquisition.

    A missing ``(source, day)`` is the *normal* case for many modalities (S1 is
    present on only ~16 archive dates), not an error — the model masks it. The
    exporter writes this in the source's band slots.

    Args:
        n_bands: Number of bands the adapter declares (``len(bands_out)``).
        shape: Target ``(height, width)`` (defaults to the 100×100 cell grid).
        fill: Fill value (defaults to ``-9999``).
        dtype: Output dtype (defaults to ``float32``).

    Returns:
        An array of shape ``(n_bands, *shape)`` filled with ``fill``.
    """
    return np.full((n_bands, *shape), fill, dtype=dtype)


def _mask_fill_to_nan(
    source: npt.NDArray[np.floating],
    src_nodata: float | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Replace fill sentinels with NaN so bilinear never blends valid + fill.

    Args:
        source: Source band stack ``(C, H, W)``.
        src_nodata: The source's own nodata value (in addition to the universal
            :data:`FILL_SENTINELS`), or ``None``.

    Returns:
        ``(masked, fill_mask)`` where ``masked`` is ``source`` with all fill
        pixels set to NaN, and ``fill_mask`` is True wherever a pixel was fill.
    """
    sentinels = set(FILL_SENTINELS)
    if src_nodata is not None:
        sentinels.add(float(src_nodata))

    fill_mask = np.zeros(source.shape, dtype=bool)
    for sentinel in sentinels:
        fill_mask |= source == sentinel

    masked = source.astype(np.float64, copy=True)
    masked[fill_mask] = np.nan
    return masked, fill_mask


def reproject_to_cell(
    *,
    source: npt.NDArray[np.floating],
    src_transform: Affine,
    src_crs: str,
    cell: GridCell,
    categorical: bool = False,
    src_nodata: float | None = None,
    restore_fill: float = float(NO_DATA_VALUE),
) -> npt.NDArray[np.floating]:
    """Reproject a source band stack onto a cell's target grid (the shared resampler).

    Dispatch:
    - **continuous** (``categorical=False``) → **nodata-aware bilinear**. Fill
      sentinels (:data:`FILL_SENTINELS` plus ``src_nodata``) are masked to NaN
      *before* the warp so the interpolator never blends a valid value with a
      sentinel (edge-bleed guard, REVIEW_AUDIT #4); any output pixel whose
      contributing source pixels were all fill is restored to ``restore_fill``.
    - **categorical / QA** (``categorical=True``) → **nearest**, which cannot
      invent intermediate class codes; no NaN masking needed.

    Args:
        source: Source bands ``(C, H, W)``.
        src_transform: Source affine transform.
        src_crs: Source CRS string.
        cell: Target :class:`GridCell` (supplies ``crs``, ``transform``, ``shape``).
        categorical: Use nearest (QA/categorical) instead of bilinear.
        src_nodata: Source-specific nodata to mask in addition to the universal
            sentinels (e.g. a sensor's own fill); ``None`` if only the universal
            sentinels apply.
        restore_fill: Value written where the bilinear result is undefined
            (all-fill neighbourhood). Defaults to ``-9999``. For MODIS, pass
            ``-28672`` so the native fill survives.

    Returns:
        The reprojected stack ``(C, *cell.shape)`` as ``float32``.
    """
    n_bands = source.shape[0]
    # Guard a degenerate source (a 0-px axis from an AOI-edge sliver window): rasterio's
    # ``reproject`` builds a source MemoryDataset and fails deep inside GDAL with
    # "Invalid dataset dimensions : 0 x N" — a cryptic, pool-killing error. A source with
    # no pixels contributes nothing, so return the cell-shaped fill instead (callers treat
    # ``restore_fill``/nodata as "band absent here"). Upstream ``cell_window`` already
    # rejects such windows; this is the shared-chokepoint backstop for any other path.
    if source.shape[1] == 0 or source.shape[2] == 0:
        return np.full((n_bands, *cell.shape), restore_fill, dtype=np.float32)
    dst = np.empty((n_bands, *cell.shape), dtype=np.float64)

    if categorical:
        # Nearest preserves discrete class codes; fill sentinels propagate as-is.
        reproject(
            source=source.astype(np.float64),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=cell.transform,
            dst_crs=cell.crs,
            resampling=Resampling.nearest,
            src_nodata=src_nodata if src_nodata is not None else NO_DATA_VALUE,
            dst_nodata=NO_DATA_VALUE,
        )
        return dst.astype(np.float32)

    # Continuous: mask fill → NaN, bilinear warp, restore fill where undefined.
    masked, _ = _mask_fill_to_nan(source, src_nodata)
    dst[:] = np.nan
    reproject(
        source=masked,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=cell.transform,
        dst_crs=cell.crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    dst[np.isnan(dst)] = restore_fill
    return dst.astype(np.float32)


class LocalSourceAdapter(abc.ABC):
    """Abstract base every modality adapter implements (the Port).

    Subclasses declare their output band order, spatial kind, and native fill,
    and implement :meth:`fetch` to read the **clipped** archive and return the
    source's bands on a cell's target grid.

    Class attributes (set by each subclass):
        bands_out: Exact band names this adapter emits, in order. Must be a
            contiguous slice of :data:`snow_galileo.data.local_sources.layout.DYNAMIC_BANDS`
            (or :data:`~snow_galileo.data.local_sources.layout.STATIC_BANDS` for static
            adapters) — the exporter relies on this to place them.
        spatial_kind: One of ``"high"|"med"|"low"|"time"|"space"|"static"`` (the
            band-group resolution tier).
        native_fill: The source's native fill value to preserve in addition to
            ``-9999`` (e.g. ``-28672`` for MODIS), or ``None``.
    """

    bands_out: list[str]
    spatial_kind: SpatialKind
    native_fill: float | None = None

    @abc.abstractmethod
    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return this source's bands for one ``(cell, day)`` on the cell grid.

        Contract every implementation must satisfy:
        - Output shape is ``(len(bands_out), *cell.shape)``, ``-9999`` nodata
          (plus :attr:`native_fill` where applicable), dtype ``float32``.
        - Output is reprojected onto ``cell``'s target grid via
          :func:`reproject_to_cell` (bilinear continuous, nearest QA/categorical).
        - A missing acquisition returns :func:`create_placeholder` — **not** an
          error.
        - **Same-tile/date coalesce runs before cross-tile mosaic-before-crop**
          for scene/granule sources (declared here; implemented per-adapter in
          TASK-012/013). Coalesce is a per-pixel valid-pixel union (first valid
          wins, deterministic latest-processing-time order), **not** an average.

        Args:
            cell: The target :class:`GridCell`.
            day: The acquisition day, or ``None`` for static layers (DEM,
                WorldCover) which ignore it.

        Returns:
            The band stack ``(C, H, W)`` on the cell grid.
        """
        raise NotImplementedError
