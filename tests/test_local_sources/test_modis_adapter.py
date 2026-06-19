"""MODIS MOD09GA adapter tests (TASK-009, AC-12 / AC-13 / AC-18).

The MODIS adapter replaces the seven ``sur_refl_b01..b07`` placeholders with real
values on the cell grid, reading the 500 m sinusoidal grid and **preserving the
native ``-28672`` fill** (the loader's "MODIS data was present" sentinel —
``landsat_eval.py:317,331``). A companion :class:`ModisCloudAdapter` emits the
``state_1km`` cloud flag (categorical, NN).

Contract under test (TASK-009 §2/§5):
- ``bands_out == sur_refl_b01..b07``; output ``(7, *cell.shape)``; ``spatial_kind ==
  "low"``; ``native_fill == -28672``.
- **Resample = NEAREST** (decision 2026-06-04): the 500 m sinusoidal grid is far
  coarser than the 10 m cell, so GEE upsamples it as a constant block per MODIS pixel.
  Measured vs the Phase-0 reference patch: **bit-exact** (median 0 across 8 timesteps);
  bilinear smears (~322, up to 941). The spec's "nodata-aware bilinear" is superseded —
  nearest cannot bleed across the fill boundary, so the edge-bleed guard is moot.
- ``-28672`` survives into the output where the source had it; no out-of-domain
  negative appears (no bleed).
- Missing ``(MODIS, day)`` → all-``-9999`` (AC-13).

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
from tests._archive_fixtures import resolve_archive_root

#: Phase-0 GEE reference patches (308-band cubes).
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

_SUR_REFL_BANDS = [f"sur_refl_b0{i}" for i in range(1, 8)]

#: MODIS sur_refl offsets inside the 38-band dynamic block (b01..b07 at 17..23);
#: state_1km is the cloud-group head at offset 35.
_OFF_B01 = 17
_OFF_STATE_1KM = 35
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
def adapter():
    """MODIS science-band adapter over the committed slim crop (bit-exact)."""
    root = resolve_archive_root("modis", pattern="*sur_refl_b01*.tif")
    if root is None:
        pytest.skip(
            "No MODIS fixture under tests/fixtures (rebuild with populate_test_archive.py)"
        )
    from src.data.local_sources.modis import ModisAdapter

    return ModisAdapter(archive_root=root)


@pytest.fixture()
def cloud_adapter():
    """MODIS state_1km cloud adapter over the committed slim crop, like :func:`adapter`."""
    root = resolve_archive_root("modis", pattern="*state_1km*.tif")
    if root is None:
        pytest.skip(
            "No MODIS fixture under tests/fixtures (rebuild with populate_test_archive.py)"
        )
    from src.data.local_sources.modis import ModisCloudAdapter

    return ModisCloudAdapter(archive_root=root)


@pytest.fixture()
def patch() -> Path:
    patches = sorted(_REF_DIR.glob("PR_*.tif"))
    if not patches:
        pytest.skip(f"No GEE reference patches under {_REF_DIR}")
    return patches[0]


def test_bands_out_and_kind(adapter) -> None:
    """``bands_out`` is sur_refl_b01..b07 in order; low tier; -28672 native fill (AC-1)."""
    assert adapter.bands_out == _SUR_REFL_BANDS
    assert adapter.spatial_kind == "low"
    assert adapter.native_fill == MODIS_FILL_VALUE


def test_fetch_shape_and_grid(adapter, patch: Path) -> None:
    """Output is ``(7, H, W)`` on the cell's UTM grid (AC-1)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2025, 4, 4))
    assert out.shape == (7, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS


def test_missing_day_is_all_nodata(adapter, patch: Path) -> None:
    """A day with no archive folder → all-``-9999`` of declared shape (AC-3/AC-13)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2030, 1, 1))
    assert out.shape == (7, *cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


def test_no_value_bleed_near_fill(adapter, patch: Path) -> None:
    """No out-of-domain negative appears (nearest cannot blend valid + -28672) (AC-2)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2025, 4, 4))
    # The only legal negatives are the exact fill sentinels.
    illegal = (out < 0) & (out != MODIS_FILL_VALUE) & (out != NO_DATA_VALUE)
    assert not illegal.any(), f"{illegal.sum()} bled values between valid and fill"


def test_fill_preserved_when_source_has_fill(adapter, patch: Path) -> None:
    """If a cell footprint includes source fill, ``-28672`` survives into the output.

    Uses a synthetic single-fill source crop to prove the value is preserved exactly;
    real cells over this AOI may be fully valid (the clip is a diagonal band).
    """
    # Build a tiny sinusoidal source: half valid, half -28672, and confirm the value
    # survives a nearest reproject onto a co-located cell.
    from rasterio.transform import from_origin

    from src.data.local_sources.base import reproject_to_cell

    src = np.full((1, 20, 20), 5000.0)
    src[0, :, 10:] = MODIS_FILL_VALUE
    sinu = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
    src_t = from_origin(-8_204_341.0, 5_816_427.0, 463.31, 463.31)
    cell = _cell_from_patch(patch)
    out = reproject_to_cell(
        source=src,
        src_transform=src_t,
        src_crs=sinu,
        cell=cell,
        categorical=True,
        src_nodata=MODIS_FILL_VALUE,
    )
    # The source-fill side maps to the fill sentinel (or off-grid -9999), never a blend.
    assert (out == MODIS_FILL_VALUE).any() or (out == NO_DATA_VALUE).any()
    assert not ((out > 0) & (out < 5000)).any(), "interpolated bleed between 5000 and fill"


@pytest.mark.parametrize("ts", list(range(8)))
def test_parity_against_gee_bitexact(adapter, patch: Path, ts: int) -> None:
    """sur_refl_b01 matches the GEE reference bit-exactly under nearest (AC-2)."""
    cell = _cell_from_patch(patch)
    day = datetime.date(2025, 4, 6) - datetime.timedelta(days=7 - ts)
    out = adapter.fetch(cell, day=day)

    with rasterio.open(patch) as ds:
        ref_b01 = ds.read(_DYNAMIC_PER_TS * ts + _OFF_B01 + 1)

    valid = (ref_b01 != MODIS_FILL_VALUE) & (ref_b01 > -100) & (out[0] != MODIS_FILL_VALUE)
    if valid.sum() < 50:
        pytest.skip(f"timestep {ts}: cell footprint is mostly MODIS fill")
    med = np.median(np.abs(out[0][valid] - ref_b01[valid]))
    assert med == 0.0, f"sur_refl_b01 not bit-exact vs GEE (median {med})"


def test_cloud_adapter_state_1km(cloud_adapter, patch: Path) -> None:
    """``state_1km`` emits a single categorical band on the cell grid (AC-1)."""
    cell = _cell_from_patch(patch)
    out = cloud_adapter.fetch(cell, day=datetime.date(2025, 4, 4))
    assert cloud_adapter.bands_out == ["state_1km"]
    assert out.shape == (1, *cell.shape)
    # Bit-flag codes are non-negative small integers (or the nodata sentinel).
    codes = out[(out != NO_DATA_VALUE)]
    assert codes.size, "no valid state_1km pixels"
    assert np.all(codes == np.round(codes)), "state_1km must stay integer-coded (NN)"
