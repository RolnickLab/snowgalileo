"""Tests for the windowed-read helper ``cell_window`` (S2/Landsat OOM fix).

``cell_window`` lets the scene adapters read only a cell's neighbourhood out of a full
UTM tile instead of decoding the whole 10980×10980 band (~900 MB float64) — the change
that turned an unbounded multi-GB sweep into a flat ~600 MB one. The parity-critical
invariant proven here: a windowed read, reprojected onto the cell, is **bit-identical**
to reprojecting the full-band read — the window only ever drops source pixels outside the
cell footprint, which the reproject would never have sampled.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.data.config import NO_DATA_VALUE
from src.data.local_sources._scene_ops import cell_window
from src.data.local_sources.base import GridCell, reproject_to_cell

_CRS = "EPSG:32611"


def _cell(min_x: float, min_y: float, *, px: int = 100, m: float = 1000.0) -> GridCell:
    return GridCell.from_utm_bounds(
        cell_id=0, min_x=min_x, min_y=min_y, max_x=min_x + m, max_y=min_y + m, crs=_CRS, px=px
    )


def _band_memfile(width: int, height: int, transform: Affine, fill: float = 1.0):
    """An in-memory single-band dataset (constant value) on the given grid."""
    data = np.full((height, width), fill, dtype=np.float64)
    mem = MemoryFile()
    ds = mem.open(
        driver="GTiff", height=height, width=width, count=1, dtype="float64",
        crs=_CRS, transform=transform, nodata=float(NO_DATA_VALUE),
    )
    ds.write(data, 1)
    return ds


def test_window_covers_cell_plus_margin() -> None:
    """The window brackets the cell footprint and is far smaller than the full tile."""
    # 10 m tile, 4000×4000 px (40 km), origin at (450000, 5660000) top-left.
    tile_tf = from_origin(450_000.0, 5_660_000.0, 10.0, 10.0)
    ds = _band_memfile(4000, 4000, tile_tf)
    cell = _cell(460_000.0, 5_640_000.0)  # a 1 km cell well inside the tile

    win = cell_window(ds, cell)
    assert win is not None
    # 1 km / 10 m = 100 px + 2×4 margin → 108 px per side, not the 4000 px full tile.
    assert 100 <= win.width <= 120
    assert 100 <= win.height <= 120
    ds.close()


def test_disjoint_cell_returns_none() -> None:
    """A cell entirely outside the tile yields None (band absent here), not a crash."""
    tile_tf = from_origin(450_000.0, 5_660_000.0, 10.0, 10.0)
    ds = _band_memfile(1000, 1000, tile_tf)  # 10 km tile
    far_cell = _cell(600_000.0, 5_500_000.0)  # 150 km away

    assert cell_window(ds, far_cell) is None
    ds.close()


def test_windowed_read_reproject_is_bit_identical_to_full_read() -> None:
    """Parity invariant: windowed read → reproject == full read → reproject (bit-exact)."""
    tile_tf = from_origin(450_000.0, 5_660_000.0, 10.0, 10.0)
    # A gradient band so any pixel mis-placement would change the output.
    width = height = 2000
    grad = np.add.outer(
        np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64)
    )
    mem = MemoryFile()
    ds = mem.open(
        driver="GTiff", height=height, width=width, count=1, dtype="float64",
        crs=_CRS, transform=tile_tf, nodata=float(NO_DATA_VALUE),
    )
    ds.write(grad, 1)
    cell = _cell(455_000.0, 5_650_000.0)

    # Full-band path.
    full = ds.read(1)
    full_out = reproject_to_cell(
        source=full[np.newaxis], src_transform=ds.transform, src_crs=str(ds.crs),
        cell=cell, categorical=True, src_nodata=float(NO_DATA_VALUE),
    )[0]

    # Windowed path.
    win = cell_window(ds, cell)
    assert win is not None
    windowed = ds.read(1, window=win)
    win_out = reproject_to_cell(
        source=windowed[np.newaxis], src_transform=ds.window_transform(win),
        src_crs=str(ds.crs), cell=cell, categorical=True, src_nodata=float(NO_DATA_VALUE),
    )[0]

    np.testing.assert_array_equal(win_out, full_out)
    ds.close()


def test_edge_sliver_cell_returns_none_not_degenerate_window() -> None:
    """An AOI-edge cell that clips the tile by a sub-pixel sliver yields None, not a
    zero-width window that later crashes ``reproject`` (Mode-B day-1 pool-killer).

    Regression: the full Mode-B sweep died ~5 h in on an edge tile whose S2 QA60 read
    window rounded to ``0 x N``; ``ds.read`` returned an empty array, and ``reproject``
    raised ``CPLE_AppDefinedError: Invalid dataset dimensions : 0 x N``, killing the
    whole worker pool. ``cell_window`` must re-clamp after pixel rounding and reject a
    degenerate result.
    """
    # Tile right edge at x = 450000 + 1000*10 = 460000. Place a cell whose footprint sits
    # just past the right edge so only a thin sub-pixel sliver overlaps.
    tile_tf = from_origin(450_000.0, 5_660_000.0, 10.0, 10.0)
    ds = _band_memfile(1000, 1000, tile_tf)  # 10 km tile
    edge_cell = _cell(459_999.0, 5_655_000.0)  # 1 km cell starting 1 m before the edge

    win = cell_window(ds, edge_cell)
    # Either a valid in-bounds window or None — never a window that reads to 0 px.
    if win is not None:
        assert win.width > 0 and win.height > 0
        read = ds.read(1, window=win)
        assert read.shape[0] > 0 and read.shape[1] > 0
    ds.close()


def test_reproject_to_cell_zero_dim_source_returns_fill() -> None:
    """A degenerate (0-px axis) source returns cell-shaped fill, not a GDAL crash.

    Shared-chokepoint backstop for the ``cell_window`` fix: any adapter path that hands a
    source with a 0-width/height axis must get the fill value back, not the cryptic
    ``Invalid dataset dimensions : 0 x N`` that propagates through the process pool.
    """
    cell = _cell(455_000.0, 5_650_000.0)
    for shape in ((1, 0, 25), (1, 25, 0)):
        source = np.empty(shape, dtype=np.float64)
        out = reproject_to_cell(
            source=source, src_transform=from_origin(455_000.0, 5_651_000.0, 10.0, 10.0),
            src_crs=_CRS, cell=cell, categorical=True, src_nodata=float(NO_DATA_VALUE),
        )
        assert out.shape == (1, *cell.shape)
        assert np.all(out == float(NO_DATA_VALUE))


def test_window_handles_cross_zone_bounds() -> None:
    """A UTM-12N cell over a UTM-11N tile transforms bounds before windowing (no crash)."""
    tile_tf = from_origin(450_000.0, 5_660_000.0, 10.0, 10.0)
    ds = _band_memfile(4000, 4000, tile_tf)  # EPSG:32611 tile
    # Cell expressed in 32612; its 32611 footprint still lands inside the tile region.
    cell = GridCell.from_utm_bounds(
        cell_id=0, min_x=200_000.0, min_y=5_645_000.0, max_x=201_000.0, max_y=5_646_000.0,
        crs="EPSG:32612", px=100,
    )
    # Should not raise; returns either a window or None depending on overlap.
    result = cell_window(ds, cell)
    assert result is None or (result.width > 0 and result.height > 0)
    ds.close()
