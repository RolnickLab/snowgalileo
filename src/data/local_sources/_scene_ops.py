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
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.merge import merge

from src.data.config import NO_DATA_VALUE


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
