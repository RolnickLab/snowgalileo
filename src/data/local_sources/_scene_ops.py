"""Shared scene-source ops: per-pixel coalesce + cross-tile mosaic (TASK-012/013).

Scene/granule sources (Landsat tars, Sentinel-2 SAFE zips) share the same three-step
band assembly once a single band is read into a native-grid array:

1. **Coalesce** same-(tile, date) products per pixel — first valid wins, deterministic
   latest-processing-time order, a valid-pixel *union* (never an average).
2. **Mosaic** across tiles in their native CRS.
3. (the caller then reprojects the mosaic onto the cell grid.)

This module factors those two source-agnostic steps out of the per-source adapters so
``landsat.py`` and ``s2.py`` cannot drift. The per-source read (tar+MTL→TOA vs JP2−1000)
stays in each adapter; this module never touches I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.errors import WindowError
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell, reproject_to_cell

#: Pixels of margin added around the cell footprint when windowing a source band, so the
#: nearest/bilinear reproject onto the cell grid has the neighbouring source pixels it needs
#: at the edges (a too-tight window would nodata the cell border → parity regression). Four
#: 10–30 m source pixels comfortably brackets a 10 m cell pixel under either resampler.
_WINDOW_MARGIN_PX: int = 4


def cell_window(ds: rasterio.io.DatasetReader, cell: GridCell) -> Window | None:
    """Pixel window over ``ds`` covering ``cell``'s footprint plus a safety margin.

    Lets an adapter read **only** the cell's neighbourhood out of a full UTM tile
    (a 10980×10980 S2/Landsat band is ~900 MB as float64 if read whole — the windowed
    read is a few KB). The window is the cell's bounds reprojected into the band's CRS,
    converted to pixels, padded by :data:`_WINDOW_MARGIN_PX`, and clamped to the dataset.

    The window is intentionally a few pixels larger than the cell so the subsequent
    nearest/bilinear :func:`~src.data.local_sources.base.reproject_to_cell` has full
    edge coverage — the reprojected output is bit-identical to one taken from the full
    band (the reproject only ever samples source pixels inside the cell footprint).

    Args:
        ds: An open rasterio dataset for one source band (native CRS/transform).
        cell: The target grid cell (its ``polygon`` is in :data:`CELL_TARGET_CRS`).

    Returns:
        A clamped :class:`rasterio.windows.Window`, or ``None`` if the cell footprint
        does not intersect the band at all (caller treats as "band absent here").
    """
    min_x, min_y, max_x, max_y = cell.polygon.bounds
    left, bottom, right, top = transform_bounds(cell.crs, ds.crs, min_x, min_y, max_x, max_y)
    win = from_bounds(left, bottom, right, top, transform=ds.transform)
    # Pad by the margin, then clamp to the dataset extent. ``Window.intersection`` raises
    # ``WindowError`` when the two windows are disjoint (cell outside the tile) — treat that
    # as "band absent here" (None), the same as a tile the cell does not cover.
    padded = Window(
        col_off=win.col_off - _WINDOW_MARGIN_PX,
        row_off=win.row_off - _WINDOW_MARGIN_PX,
        width=win.width + 2 * _WINDOW_MARGIN_PX,
        height=win.height + 2 * _WINDOW_MARGIN_PX,
    )
    try:
        clamped = padded.intersection(Window(0, 0, ds.width, ds.height))
    except WindowError:
        return None
    if clamped.width <= 0 or clamped.height <= 0:
        return None
    # Round to whole pixels (windows must be integer for a clean read+transform).
    rounded = clamped.round_offsets(op="floor").round_lengths(op="ceil")
    # Re-clamp after rounding and reject a degenerate (zero-px) result. ``floor``/``ceil``
    # on a sub-pixel sliver at the tile edge can push the window past the dataset bound so
    # ``ds.read`` returns a 0-width/height array, which later crashes ``reproject`` with
    # "Invalid dataset dimensions : 0 x N". This surfaces only on AOI-edge cells (Mode B),
    # never on interior sample cells (Mode A). Treat it as "band absent here" (None).
    try:
        rounded = rounded.intersection(Window(0, 0, ds.width, ds.height))
    except WindowError:
        return None
    if rounded.width <= 0 or rounded.height <= 0:
        return None
    return rounded


@dataclass(frozen=True)
class BandRead:
    """One band read from a scene/granule: values + its native georeferencing.

    Attributes:
        values: ``(H, W)`` band array; ``-9999`` (:data:`NO_DATA_VALUE`) where the source
            pixel was fill/invalid.
        transform: Native affine transform.
        crs: Native CRS string.
    """

    values: npt.NDArray[np.float64]
    transform: Affine
    crs: str


def coalesce_tile(reads: list[BandRead]) -> BandRead:
    """Per-pixel coalesce of same-(tile, date) products: first valid wins.

    ``reads`` must share a native grid (same tile → identical transform/CRS) and be
    ordered **latest-processing-time first**. Keeps the first non-``-9999`` value at each
    pixel; ``-9999`` only where every product is fill. A valid-pixel union, never an
    average — preserving the GEE value domain.

    Args:
        reads: Same-(tile, date) products, latest-processing-time first (non-empty).

    Returns:
        The coalesced :class:`BandRead` on the shared native grid.
    """
    out = reads[0].values.copy()
    for nxt in reads[1:]:
        fill = out == float(NO_DATA_VALUE)
        out[fill] = nxt.values[fill]
    return BandRead(values=out, transform=reads[0].transform, crs=reads[0].crs)


def mosaic_tiles(tile_reads: list[BandRead]) -> tuple[npt.NDArray[np.float64], Affine, str]:
    """Merge per-tile arrays in their native zone → ``(array, transform, crs)``.

    A single tile is returned unchanged (no merge). ``merge`` requires a common CRS, which
    holds within an archive scene-group (Landsat per scene-group, S2 per MGRS zone).

    Args:
        tile_reads: One coalesced :class:`BandRead` per tile (non-empty).

    Returns:
        ``(mosaic_array, transform, crs)`` on the merged native grid.
    """
    if len(tile_reads) == 1:
        return tile_reads[0].values, tile_reads[0].transform, tile_reads[0].crs
    datasets = [
        MemoryFile().open(
            driver="GTiff",
            height=r.values.shape[0],
            width=r.values.shape[1],
            count=1,
            dtype="float64",
            crs=r.crs,
            transform=r.transform,
            nodata=float(NO_DATA_VALUE),
        )
        for r in tile_reads
    ]
    try:
        for ds, r in zip(datasets, tile_reads):
            ds.write(r.values, 1)
        mosaic, transform = merge(datasets, nodata=float(NO_DATA_VALUE))
        return mosaic[0], transform, tile_reads[0].crs
    finally:
        for ds in datasets:
            ds.close()


def mosaic_to_cell(
    tile_reads: list[BandRead], cell: GridCell, *, categorical: bool
) -> npt.NDArray[np.float32]:
    """Mosaic per-tile reads onto the cell grid, **safe across mixed CRSs**.

    :func:`mosaic_tiles` → ``rasterio.merge`` requires every input share one CRS and
    raises ``RasterioError: CRS mismatch`` otherwise. The Landsat archive is **mixed-UTM-
    zone per scene** (paths 043/044 → EPSG:32611, 042024 → EPSG:32612), so a cell whose day
    draws scenes from both zones yields ``tile_reads`` spanning two CRSs. This helper groups
    the reads by native CRS, merges only **within** a zone (where ``merge`` is valid),
    reprojects each zone's mosaic to the cell grid in its own native CRS, then
    first-valid-combines on the common cell grid — a valid-pixel union (first seen wins),
    never an average (same semantics as :func:`coalesce_tile`, just across zones on the
    shared grid). The single-CRS case is unchanged: one merge, one reproject (bit-exact
    with the prior path).

    Args:
        tile_reads: One :class:`BandRead` per tile (non-empty), already coalesced.
        cell: The target grid cell (supplies the destination ``crs``/``transform``/``shape``).
        categorical: ``True`` → nearest resample (QA/categorical); ``False`` → nodata-aware
            bilinear (continuous bands).

    Returns:
        The combined band on the cell grid, ``(H, W)`` ``float32``; ``-9999`` where no zone
        had a valid pixel.
    """
    by_crs: dict[str, list[BandRead]] = {}
    for read in tile_reads:
        by_crs.setdefault(read.crs, []).append(read)

    cell_arrays: list[npt.NDArray[np.floating]] = []
    for crs_reads in by_crs.values():
        merged, transform, crs = mosaic_tiles(crs_reads)
        reprojected = reproject_to_cell(
            source=merged[np.newaxis, :, :],
            src_transform=transform,
            src_crs=crs,
            cell=cell,
            categorical=categorical,
            src_nodata=float(NO_DATA_VALUE),
        )
        cell_arrays.append(reprojected[0])

    out = cell_arrays[0].copy()
    for nxt in cell_arrays[1:]:
        fill = out == float(NO_DATA_VALUE)
        out[fill] = nxt[fill]
    return out.astype(np.float32)
