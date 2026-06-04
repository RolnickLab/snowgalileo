"""ERA5-Land adapter tests (TASK-008, AC-12 / AC-13 / AC-20 / AC-20b).

The ERA5 adapter replaces the five meteorological placeholders with raw values on
the cell grid, reading the **already-daily** archive (one slice/day, no hourly
re-aggregation). Contract under test (TASK-008 §2/§5):

- ``bands_out`` is the five bands in canonical order; output ``(5, *cell.shape)``;
  ``spatial_kind == "time"``; ``native_fill is None``; raw **Kelvin**/native units
  (the temperature-shift bug lives in ``Normalizer`` downstream — NOT the adapter).
- **Precip day-shift (AC-20b):** ``total_precipitation_sum`` is an accumulation
  stamped at ``00:00`` of the *next* day, so precip for inference day ``d`` is read
  from the ``d+1`` ``00:00`` slice. Instantaneous vars (temps, winds) carry **no**
  shift — read the slice labelled ``d``. The day-shift test uses synthetic NetCDF so
  the off-by-one is caught deterministically.
- **Resample = NEAREST** (decision 2026-06-04): the ~11 km ERA5 grid is far coarser
  than the 1 km cell, so GEE upsamples it as a constant block per ERA5 cell. Measured
  vs the Phase-0 reference patches: t2m median ~0.0001 K, precip ~0.0001 m (exact);
  bilinear smears (~0.26 K). The spec's "bilinear" text is superseded.
- Missing ``(ERA5, day)`` → all-``-9999`` of declared shape (AC-13).

Real-archive parity tests skip cleanly if the archive or reference patches are absent.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from shapely.geometry import box

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import CELL_TARGET_CRS, GridCell

#: Clipped ERA5 archive root (the adapter's input).
_ERA5_ROOT = Path("data/clipped_bow_valley_selection_raw/era5")

#: Phase-0 GEE reference patches (308-band cubes).
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

_ERA5_BANDS = [
    "skin_temperature",
    "temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

#: ERA5 band offsets inside the 38-band dynamic block (skin..v at 30..34).
_OFF_SKT, _OFF_T2M, _OFF_TP, _OFF_U, _OFF_V = 30, 31, 32, 33, 34
_DYNAMIC_PER_TS = 38

#: Measured parity tolerances vs GEE (nearest resample): essentially exact.
_T2M_MED_TOL_K = 0.01
_TP_MED_TOL_M = 0.001


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
    """The real ERA5 adapter; skip if the clipped archive is missing."""
    if not any(_ERA5_ROOT.rglob("*.nc")):
        pytest.skip(f"No ERA5 NetCDF under {_ERA5_ROOT}")
    from src.data.local_sources.era5 import Era5Adapter

    return Era5Adapter(archive_root=_ERA5_ROOT)


@pytest.fixture()
def patch() -> Path:
    patches = sorted(_REF_DIR.glob("PR_*.tif"))
    if not patches:
        pytest.skip(f"No GEE reference patches under {_REF_DIR}")
    return patches[0]


def test_bands_out_and_kind(adapter) -> None:
    """``bands_out`` is the five bands in order; time tier (AC-1)."""
    assert adapter.bands_out == _ERA5_BANDS
    assert adapter.spatial_kind == "time"
    assert adapter.native_fill is None


def test_fetch_shape_and_grid(adapter, patch: Path) -> None:
    """Output is ``(5, H, W)`` on the cell's UTM grid (AC-1)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2025, 4, 4))
    assert out.shape == (5, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS


def test_units_are_raw_kelvin(adapter, patch: Path) -> None:
    """Temps stay in Kelvin — the adapter applies NO Celsius shift (AC-2)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2025, 4, 4))
    skt, t2m = out[0], out[1]
    for band in (skt, t2m):
        valid = band[band != NO_DATA_VALUE]
        assert valid.size, "no valid temperature pixels"
        # Kelvin near-surface temps sit well above 200 K; Celsius would be ~ -10..10.
        assert valid.min() > 200.0, "temperature looks Celsius-shifted — adapter must stay raw"


def test_missing_day_is_all_nodata(adapter, patch: Path) -> None:
    """A day with no archive file → all-``-9999`` of declared shape (AC-3/AC-13)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=datetime.date(2030, 1, 1))  # far outside the archive
    assert out.shape == (5, *cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


# ---- Precip day-shift: deterministic synthetic NetCDF (AC-20b) -------------- #


def _write_synthetic_era5(root: Path, year: int, month: int, n_days: int) -> None:
    """Write a minimal monthly ERA5 archive folder + precip file.

    Each instantaneous var's slice for day ``d`` is the constant ``d`` (so a read of
    day ``d`` is trivially identifiable). The precip ``tp`` slice stamped day ``d`` is
    the constant ``d * 100`` — so if the adapter reads the ``d+1`` slice for inference
    day ``d`` it gets ``(d+1) * 100``, and a naive same-day read gets ``d * 100``.
    """
    lats = np.array([51.2, 51.1, 51.0, 50.9], dtype=np.float64)  # descending, 0.1°
    lons = np.array([-116.2, -116.1, -116.0, -115.9], dtype=np.float64)
    times = np.array(
        [np.datetime64(f"{year:04d}-{month:02d}-{d:02d}T00:00") for d in range(1, n_days + 1)]
    )
    folder = root / f"{year:04d}{month:02d}_ERA5LAND"
    folder.mkdir(parents=True, exist_ok=True)

    inst = {
        "skin_temperature_0_daily-mean.nc": "skt",
        "2m_temperature_0_daily-mean.nc": "t2m",
        "10m_u_component_of_wind_0_daily-mean.nc": "u10",
        "10m_v_component_of_wind_0_daily-mean.nc": "v10",
    }
    for fname, var in inst.items():
        data = np.empty((n_days, lats.size, lons.size), dtype=np.float64)
        for di in range(n_days):
            data[di] = float(di + 1) + (273.0 if var in ("skt", "t2m") else 0.0)
        xr.Dataset(
            {var: (("valid_time", "latitude", "longitude"), data)},
            coords={"valid_time": times, "latitude": lats, "longitude": lons},
        ).to_netcdf(folder.parent / folder.name / fname, engine="h5netcdf")

    tp = np.empty((n_days, lats.size, lons.size), dtype=np.float64)
    for di in range(n_days):
        tp[di] = float((di + 1) * 100)  # day-(d) slice value = d*100
    xr.Dataset(
        {"tp": (("valid_time", "latitude", "longitude"), tp)},
        coords={"valid_time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(root / f"{year:04d}{month:02d}_ERA5LAND_totalprecip.nc", engine="h5netcdf")


def test_precip_day_shift_and_temp_no_shift(tmp_path: Path) -> None:
    """Precip(d) = the d+1 00:00 slice; temp(d) = the day-d slice (AC-20b)."""
    from src.data.local_sources.era5 import Era5Adapter

    _write_synthetic_era5(tmp_path, 2025, 4, n_days=10)
    adapter = Era5Adapter(archive_root=tmp_path)

    # A cell inside the synthetic grid (UTM 11N near 51.05 N, -116.05 E).
    cell = GridCell.from_utm_bounds(
        cell_id=0, min_x=562_000.0, min_y=5_654_000.0, max_x=563_000.0, max_y=5_655_000.0
    )

    d = 4  # inference day 2025-04-04
    out = adapter.fetch(cell, day=datetime.date(2025, 4, d))
    t2m = out[1]
    tp = out[2]
    t2m_v = np.median(t2m[t2m != NO_DATA_VALUE])
    tp_v = np.median(tp[tp != NO_DATA_VALUE])

    # temp day-d slice == d + 273 (NO shift)
    assert t2m_v == pytest.approx(d + 273.0, abs=1e-6), f"temp shifted: {t2m_v}"
    # precip day-d == the (d+1) slice == (d+1)*100 (shift), NOT d*100
    assert tp_v == pytest.approx((d + 1) * 100, abs=1e-6), f"precip not shifted: {tp_v}"
    assert tp_v != pytest.approx(d * 100, abs=1e-6), "precip read same-day (off-by-one bug)"


def test_precip_shift_crosses_month_boundary(tmp_path: Path) -> None:
    """Precip for the last day of a month reads the next month's first slice (AC-20b)."""
    from src.data.local_sources.era5 import Era5Adapter

    _write_synthetic_era5(tmp_path, 2025, 4, n_days=30)
    _write_synthetic_era5(tmp_path, 2025, 5, n_days=31)
    adapter = Era5Adapter(archive_root=tmp_path)
    cell = GridCell.from_utm_bounds(
        cell_id=0, min_x=562_000.0, min_y=5_654_000.0, max_x=563_000.0, max_y=5_655_000.0
    )

    # Inference day 2025-04-30: precip must come from 2025-05-01 (slice value 1*100).
    out = adapter.fetch(cell, day=datetime.date(2025, 4, 30))
    tp = out[2]
    tp_v = np.median(tp[tp != NO_DATA_VALUE])
    assert tp_v == pytest.approx(100.0, abs=1e-6), f"cross-month precip wrong: {tp_v}"


# ---- Real-archive parity vs GEE (AC-2 / AC-20) ------------------------------ #


def test_parity_against_gee(adapter, patch: Path) -> None:
    """t2m and precip match the GEE reference within nearest-resample tolerances."""
    cell = _cell_from_patch(patch)
    # patch PR_20250406 → window end 2025-04-06, timestep t=5 → 2025-04-04.
    t = 5
    day = datetime.date(2025, 4, 6) - datetime.timedelta(days=7 - t)
    out = adapter.fetch(cell, day=day)

    with rasterio.open(patch) as ds:
        ref_t2m = ds.read(_DYNAMIC_PER_TS * t + _OFF_T2M + 1)
        ref_tp = ds.read(_DYNAMIC_PER_TS * t + _OFF_TP + 1)

    v_t = ref_t2m > 100.0  # Kelvin valid (0 is ERA5 nodata in the patch)
    assert v_t.any(), "no valid reference t2m"
    t2m_med = np.median(np.abs(out[1][v_t] - ref_t2m[v_t]))
    assert t2m_med <= _T2M_MED_TOL_K, f"t2m median {t2m_med:.5f} K > {_T2M_MED_TOL_K}"

    v_p = ref_tp != NO_DATA_VALUE
    tp_med = np.median(np.abs(out[2][v_p] - ref_tp[v_p]))
    assert tp_med <= _TP_MED_TOL_M, f"precip median {tp_med:.7f} m > {_TP_MED_TOL_M}"
