"""ESA WorldCover adapter tests (TASK-006, AC-12 / AC-22).

The WorldCover adapter replaces the static-``Map`` placeholder with the real
v200 (2021) land-cover band on the cell grid. It is **static** (ignores ``day``)
and **categorical** (NN resampling, class codes preserved — never one-hot; the
loader one-hot-encodes ``Map`` itself).

Contract under test (DATA_ANALYSIS §ESA WorldCover, TASK-006 §2/§5):
- ``bands_out == ["Map"]``; output shape ``(1, *cell.shape)``.
- Reprojected onto the cell's **EPSG:32611 (UTM 11N)** grid — the source is
  EPSG:4326, the target is the UTM cell grid (the same UTM cascade as the rest of
  the pipeline; AC-1 corrected 2026-06-04 from a stale "4326 target").
- Class codes ∈ ``{10,20,30,40,50,60,70,80,90,95,100}`` (plus ``0``/``-9999`` for
  nodata) — **not** one-hot.
- Output is identical for any ``day`` (including ``None``) — static layer.

Uses the real clipped archive tile; skips cleanly if it is absent.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import CELL_TARGET_CRS, GridCell

#: Allowed WorldCover v200 class codes (plus 0 / -9999 for nodata).
_ALLOWED_CODES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0, int(NO_DATA_VALUE)}

#: Clipped WorldCover archive root (the adapter's input).
_WC_ROOT = Path("data/clipped_bow_valley_selection_raw/worldcover")


@pytest.fixture()
def cell() -> GridCell:
    """A UTM 11N 1 km cell fully inside the WorldCover tile coverage (~51.03N)."""
    min_x, min_y = 562_600.0, 5_653_000.0
    return GridCell.from_utm_bounds(
        cell_id=0,
        min_x=min_x,
        min_y=min_y,
        max_x=min_x + 1_000.0,
        max_y=min_y + 1_000.0,
    )


@pytest.fixture()
def adapter():
    """The real WorldCover adapter; skip if the clipped archive is missing."""
    if not any(_WC_ROOT.rglob("*.tif")):
        pytest.skip(f"No clipped WorldCover tiles under {_WC_ROOT}")
    from src.data.local_sources.worldcover import WorldCoverAdapter

    return WorldCoverAdapter(archive_root=_WC_ROOT)


def test_bands_out_is_single_map_band(adapter) -> None:
    """``bands_out == ["Map"]`` and spatial kind is the static-space tier (AC-1)."""
    assert adapter.bands_out == ["Map"]
    assert adapter.spatial_kind == "space"
    assert adapter.native_fill is None


def test_fetch_shape_and_grid(adapter, cell: GridCell) -> None:
    """Output is ``(1, H, W)`` on the cell's UTM grid (AC-1 golden-grid triple)."""
    out = adapter.fetch(cell, day=None)
    assert out.shape == (1, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS  # the grid we reprojected onto is UTM 11N


def test_class_codes_are_categorical_not_one_hot(adapter, cell: GridCell) -> None:
    """Output holds raw class codes in the allowed set — never one-hot (AC-2)."""
    out = adapter.fetch(cell, day=None)[0]
    codes = set(np.unique(out).astype(int).tolist())
    assert codes <= _ALLOWED_CODES, f"unexpected WorldCover codes: {codes - _ALLOWED_CODES}"
    # A real land-cover crop has at least one vegetated/built class, not just nodata.
    assert codes - {0, int(NO_DATA_VALUE)}, "all-nodata crop — wrong cell or clip"


def test_static_independent_of_day(adapter, cell: GridCell) -> None:
    """The Map band is identical for different ``day`` values and ``None`` (AC-2, static)."""
    a = adapter.fetch(cell, day=None)
    b = adapter.fetch(cell, day=datetime.date(2025, 4, 6))
    c = adapter.fetch(cell, day=datetime.date(2025, 5, 28))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
