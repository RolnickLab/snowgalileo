"""Sentinel-3 OLCI adapter tests (TASK-011, AC-12 / AC-13 / AC-17).

The S3 adapter replaces the ``[Oa17_radiance, Oa21_radiance]`` placeholders with
top-of-atmosphere radiance on the cell grid, georeferenced from the OLCI per-pixel
``geo_coordinates`` grid (curvilinear swath → cell via ``scipy.griddata``).

Contract under test (TASK-011 §2/§5):
- ``bands_out == ["Oa17_radiance", "Oa21_radiance"]``; output ``(2, H, W)``;
  ``spatial_kind == "med"``; identity scale (radiance preserved as GEE exports it).
- **Scale = file ``scale_factor``** (Oa17 0.00493004, Oa21 0.00324118), which the GEE
  catalog lists identically — values are W m⁻² sr⁻¹ µm⁻¹ radiance.
- **Geolocation** decodes the ``geo_coordinates`` lat/lon CF scaling (int32 ×1e-6 — the
  landmine that once clipped radiance to (0,0)); the covering overpass is selected.
- Missing ``(S3, day)`` → all-``-9999`` (AC-13).

**Parity is intentionally loose.** GEE terrain-orthorectifies OLCI in SNAP; this
swath-warp cannot replicate that, leaving OLCI's intrinsic ~300–450 m geolocation gap
(corr ~0.66–0.73 vs GEE; a rigid shift does not close it). The scale is exact and the
geolocation input is the best available (full per-pixel grid). The residual is owned by
the out-of-scope S3-ortho/normalization follow-up (PARITY_SPIKE_NOTES §10). The patch is
only ~3 OLCI pixels wide, so the value test uses a wide tolerance and a correlation
floor rather than bit-exactness.

Skips cleanly if the archive or reference patches are absent.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import CELL_TARGET_CRS, GridCell
from tests._archive_fixtures import resolve_archive_root

_REF_DIR = Path("tests/fixtures/gee_reference_patches")

#: S3 offsets inside the 38-band dynamic block (Oa17 at 15, Oa21 at 16).
_OFF_OA17 = 15
_DYNAMIC_PER_TS = 38

#: A day the reference patch confirms S3 covers the cell (t2 = 2025-04-01).
_COVERED_DAY = datetime.date(2025, 4, 1)
_COVERED_TS = 2

#: Loose parity bar — radiance units; the un-orthorectified geolocation gap dominates.
_RAD_MED_TOL = 60.0
_CORR_FLOOR = 0.4


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
def adapter():
    """S3 OLCI adapter — archive-gated (SEN3 products too large to commit)."""
    root = resolve_archive_root("sentinel3", pattern="*.zip")
    if root is None:
        pytest.skip("No S3 products under tests/fixtures/archive/sentinel3 (download to run)")
    from src.data.local_sources.s3 import S3Adapter

    return S3Adapter(archive_root=root)


@pytest.fixture()
def patch() -> Path:
    patches = sorted(_REF_DIR.glob("PR_*.tif"))
    if not patches:
        pytest.skip(f"No GEE reference patches under {_REF_DIR}")
    return patches[0]


def test_bands_out_and_kind(adapter) -> None:
    """``bands_out`` is the two radiance bands in order; med tier (AC-1)."""
    assert adapter.bands_out == ["Oa17_radiance", "Oa21_radiance"]
    assert adapter.spatial_kind == "med"
    assert adapter.native_fill is None


def test_fetch_shape_and_grid(adapter, patch: Path) -> None:
    """Output is ``(2, H, W)`` on the cell's UTM grid (AC-1)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=_COVERED_DAY)
    assert out.shape == (2, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS


def test_missing_day_is_all_nodata(adapter, patch: Path) -> None:
    """A day with no S3 overpass → all-``-9999`` of declared shape (AC-3/AC-13)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2030, 1, 1))
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


def test_values_are_radiance_domain(adapter, patch: Path) -> None:
    """Values are TOA radiance (scaled), not raw DN — ~[0, 400], not tens of thousands (AC-2)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=_COVERED_DAY)[0]
    valid = out[out != NO_DATA_VALUE]
    assert valid.size, "no valid Oa17 pixels on a covered day"
    assert valid.max() < 1000.0, "values look like raw DN — scale_factor not applied"
    assert valid.min() >= -1.0, "radiance below the valid threshold"


def test_georeferencing_aligns_to_gee(adapter, patch: Path) -> None:
    """Warped radiance aligns with the GEE reference (loose; ortho gap is out of scope) (AC-2)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=_COVERED_DAY)[0]
    with rasterio.open(patch) as ds:
        ref = ds.read(_DYNAMIC_PER_TS * _COVERED_TS + _OFF_OA17 + 1)

    valid = (ref > -1.0) & (out != NO_DATA_VALUE)
    assert valid.sum() > 100, "S3 did not cover the cell on the expected day"
    med = np.median(np.abs(out[valid] - ref[valid]))
    corr = np.corrcoef(out[valid], ref[valid])[0, 1]
    assert med <= _RAD_MED_TOL, f"radiance median |Δ| {med:.1f} > {_RAD_MED_TOL} (loose bar)"
    assert corr >= _CORR_FLOOR, f"georeferencing correlation {corr:.3f} < {_CORR_FLOOR}"
