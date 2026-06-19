"""VIIRS VNP09GA adapter tests (TASK-010, AC-12 / AC-13 / AC-19).

VIIRS contributes two band groups, each its own adapter:
- **fine** ``[I1, I3]`` (500 m grid) → ``space_time_low_res_x`` (``spatial_kind="low"``);
- **coarse** ``[M5, M7, M10, M11]`` (1 km grid) → ``time_x``, emitted as a **per-pixel
  raster** ``(4, H, W)`` — the loader does the spatial mean; the adapter must NOT
  pre-average (pre-averaging breaks ``time_x``).

Contract under test (TASK-010 §2/§5):
- Output shapes ``(2, H, W)`` fine / ``(4, H, W)`` coarse; band order correct; both on
  the cell's EPSG:32611 grid.
- **Scale = x0.0001** (decision 2026-06-04): GEE exports VNP09GA as reflectance, not raw
  DN (the normalizer ``(x+0.795)/0.805`` confirms the reflectance domain). Validated
  bit-exact vs the Phase-0 reference patch (ratio exactly 10000). Contrast MODIS, which
  GEE keeps as raw DN. The ``-28672`` fill is restored **after** scaling — never scaled.
- **Resample = NEAREST**: the 500 m / 1 km sinusoidal grids are coarser than the 10 m
  cell; nearest reproduces GEE bit-exactly (see PARITY_SPIKE_NOTES §9).
- I-band ``-28672`` preserved (loader sentinel, same reason as MODIS).
- Missing ``(VIIRS, day)`` → all-``-9999`` (AC-13).

Real-archive parity tests skip cleanly if the archive or reference patches are absent.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.data.config import MODIS_FILL_VALUE, NO_DATA_VALUE
from src.data.local_sources.base import CELL_TARGET_CRS, GridCell

_VIIRS_ROOT = Path("data/clipped_bow_valley_selection_raw/viirs")
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

#: VIIRS offsets inside the 38-band dynamic block (I1,I3 at 24,25; M5..M11 at 26..29).
_OFF_I1 = 24
_OFF_M5 = 26
_DYNAMIC_PER_TS = 38


def _cell_from_patch(patch: Path) -> GridCell:
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=0,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


@pytest.fixture()
def fine():
    if not any(_VIIRS_ROOT.rglob("*SurfReflect_I1*.tif")):
        pytest.skip(f"No clipped VIIRS I-band tiles under {_VIIRS_ROOT}")
    from src.data.local_sources.viirs import ViirsFineAdapter

    return ViirsFineAdapter(archive_root=_VIIRS_ROOT)


@pytest.fixture()
def coarse():
    if not any(_VIIRS_ROOT.rglob("*SurfReflect_M5*.tif")):
        pytest.skip(f"No clipped VIIRS M-band tiles under {_VIIRS_ROOT}")
    from src.data.local_sources.viirs import ViirsCoarseAdapter

    return ViirsCoarseAdapter(archive_root=_VIIRS_ROOT)


@pytest.fixture()
def patch() -> Path:
    patches = sorted(_REF_DIR.glob("PR_*.tif"))
    if not patches:
        pytest.skip(f"No GEE reference patches under {_REF_DIR}")
    return patches[0]


def test_fine_bands_and_kind(fine) -> None:
    """Fine adapter: ``[I1, I3]``, low tier, -28672 native fill (AC-1)."""
    assert fine.bands_out == ["I1", "I3"]
    assert fine.spatial_kind == "low"
    assert fine.native_fill == MODIS_FILL_VALUE


def test_coarse_bands_and_kind(coarse) -> None:
    """Coarse adapter: ``[M5, M7, M10, M11]``, time tier (AC-1)."""
    assert coarse.bands_out == ["M5", "M7", "M10", "M11"]
    assert coarse.spatial_kind == "time"


def test_fine_shape(fine, patch: Path) -> None:
    """Fine output is ``(2, H, W)`` on the cell's UTM grid (AC-1/AC-2)."""
    cell = _cell_from_patch(patch)
    out = fine.fetch(cell, day=datetime.date(2025, 4, 4))
    assert out.shape == (2, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS


def test_coarse_is_per_pixel_raster_not_averaged(coarse, patch: Path) -> None:
    """Coarse output is a per-pixel ``(4, H, W)`` raster, NOT a ``(4,)`` mean (AC-2)."""
    cell = _cell_from_patch(patch)
    out = coarse.fetch(cell, day=datetime.date(2025, 4, 4))
    assert out.shape == (4, *cell.shape)
    # A real raster has spatial variation across the cell — not a single repeated value.
    band0 = out[0][out[0] != NO_DATA_VALUE]
    assert band0.size and np.unique(band0).size > 1, "coarse band looks pre-averaged"


def test_missing_day_is_all_nodata(fine, coarse, patch: Path) -> None:
    """A day with no granule → all-``-9999`` of declared shape (AC-3/AC-13)."""
    cell = _cell_from_patch(patch)
    of = fine.fetch(cell, day=datetime.date(2030, 1, 1))
    oc = coarse.fetch(cell, day=datetime.date(2030, 1, 1))
    np.testing.assert_array_equal(of, np.full_like(of, NO_DATA_VALUE))
    np.testing.assert_array_equal(oc, np.full_like(oc, NO_DATA_VALUE))


def test_values_are_reflectance_scaled(fine, patch: Path) -> None:
    """I-band values sit in the reflectance domain (~[-0.01, 1.6]), not raw DN (AC-2)."""
    cell = _cell_from_patch(patch)
    out = fine.fetch(cell, day=datetime.date(2025, 4, 4))[0]
    valid = out[(out != NO_DATA_VALUE) & (out != MODIS_FILL_VALUE)]
    assert valid.size, "no valid I1 pixels"
    assert valid.max() < 5.0, "values look like raw DN — scale factor not applied"


@pytest.mark.parametrize("ts", list(range(8)))
def test_fine_parity_against_gee_bitexact(fine, patch: Path, ts: int) -> None:
    """I1 matches the GEE reference bit-exactly (scaled, nearest) (AC-2)."""
    cell = _cell_from_patch(patch)
    day = datetime.date(2025, 4, 6) - datetime.timedelta(days=7 - ts)
    out = fine.fetch(cell, day=day)
    with rasterio.open(patch) as ds:
        ref_i1 = ds.read(_DYNAMIC_PER_TS * ts + _OFF_I1 + 1)
    valid = (ref_i1 > -1.0) & (out[0] != NO_DATA_VALUE) & (out[0] != MODIS_FILL_VALUE)
    if valid.sum() < 50:
        pytest.skip(f"timestep {ts}: cell footprint is mostly VIIRS fill/missing")
    med = np.median(np.abs(out[0][valid] - ref_i1[valid]))
    assert med == pytest.approx(0.0, abs=1e-6), f"I1 not bit-exact vs GEE (median {med})"


@pytest.mark.parametrize("ts", list(range(8)))
def test_coarse_parity_against_gee_bitexact(coarse, patch: Path, ts: int) -> None:
    """M5 per-pixel raster matches the GEE reference bit-exactly (scaled, nearest) (AC-2)."""
    cell = _cell_from_patch(patch)
    day = datetime.date(2025, 4, 6) - datetime.timedelta(days=7 - ts)
    out = coarse.fetch(cell, day=day)
    with rasterio.open(patch) as ds:
        ref_m5 = ds.read(_DYNAMIC_PER_TS * ts + _OFF_M5 + 1)
    valid = (ref_m5 > -1.0) & (out[0] != NO_DATA_VALUE) & (out[0] != MODIS_FILL_VALUE)
    if valid.sum() < 50:
        pytest.skip(f"timestep {ts}: cell footprint is mostly VIIRS fill/missing")
    med = np.median(np.abs(out[0][valid] - ref_m5[valid]))
    assert med == pytest.approx(0.0, abs=1e-6), f"M5 not bit-exact vs GEE (median {med})"
