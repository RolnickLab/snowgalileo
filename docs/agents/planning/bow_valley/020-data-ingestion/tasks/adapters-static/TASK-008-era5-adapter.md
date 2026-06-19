# TASK-008: Implement the ERA5-Land adapter

## 1. Goal
Replace the ERA5 placeholder with a real adapter that emits the five meteorological
bands in raw Kelvin / native units on the cell grid, reading the already-daily archive
files (one slice/day, no hourly re-aggregation) and applying the ERA5-Land precip
day-shift so day `i`'s total comes from the `i+1` `00:00` accumulation slice.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #3 — low parity risk, well-defined NetCDF).
- **SPEC:** FR-14, AC-12, AC-13, AC-20, AC-20b; Verification Plan step 6.
- **PLAN:** §4 adapter rule (ERA5 emits raw Kelvin; the known temperature-shift sign
  bug lives in `Normalizer`, downstream — adapter is NOT responsible).
- **Upstream tasks:** TASK-002 (clipped ERA5), TASK-003, TASK-004.
- **Source semantics (DATA_ANALYSIS.md §ERA5-Land + §Raw Archive Formats):**
  - NetCDF (`.nc`), `EPSG:4326`, 0.1° grid → **full AOI every day** (only source with
    guaranteed complete spatial coverage).
  - Bands (in order): `skin_temperature, temperature_2m, total_precipitation_sum,
    u_component_of_wind_10m, v_component_of_wind_10m`.
  - Archive layout: monthly `YYYYMM_ERA5LAND/` folders with per-variable `*_daily-mean.nc`
    files (instantaneous vars); precip as `YYYYMM_ERA5LAND_totalprecip.nc` in the parent.
  - **The archive is already daily** — one slice per day. Read directly via
    `xarray`+`h5netcdf`; **do NOT re-aggregate hourly data** (there is none on disk).
  - **`total_precipitation_sum` is an accumulation** (verified file facts: var `tp`,
    dims `(valid_time=days, lat, lon)`, `GRIB_stepType=accum`, `units=m`, `valid_time`
    stamped `YYYY-MM-DDT00:00`). ERA5-Land stamps the total that **closes** day `i` at
    **`00:00` of day `i+1`**. So precip for inference day `i` = the `i+1` `00:00` slice
    (`tp[index] → day index−1`). This is a **silent off-by-one** if done naively — it
    passes shape/type checks while attributing rain to the wrong day.
  - **Instantaneous vars** (`temperature_2m`, `skin_temperature`,
    `u/v_component_of_wind_10m`) carry **no** day shift — read the slice labelled `i`.
  - Units: Kelvin for temps, native wind units, metres (depth) for precip. Missing day →
    `-9999`. `spatial_kind="time"` (loader spatially averages into `time_x`).
- **Relevant skills:** `geospatial`, `software-dev` (xarray/h5netcdf), `tdd`.

## 3. Subtasks
- [x] 1. Write `test_era5_adapter.py` (Red): golden-grid triple; five bands in order;
      Kelvin/native units preserved (no Celsius shift in the adapter); missing day →
      all-`-9999`.
- [x] 2. Write the **precip day-shift test** (Red, AC-20b): build a synthetic `tp` where
      the `00:00` slice stamped day `d+1` has a known total and day `d`'s slice differs;
      assert the adapter's precip output for inference day `d` equals the `d+1` slice
      value (accumulation closing day `d`), and that `temperature_2m` for day `d` is read
      from the day-`d` slice (no shift). This is the test that catches the off-by-one.
- [x] 3. Implement `era5.py`: locate the daily NetCDF files for `day`, read one slice/day
      (no hourly aggregation), apply the precip `i+1` day-shift, leave temps/winds
      unshifted, reproject to cell grid (**nearest** — corrected 2026-06-04, see
      §4 Technical), stack `(5, H, W)`.
- [x] 4. Wire into exporter. 5. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** `xarray` + `h5netcdf` (already a dep). **Reproject = NEAREST, not
  bilinear (corrected 2026-06-04).** The 0.1° (~11 km) grid is far coarser than the
  1 km cell, so GEE upsamples it as a constant block per ERA5 cell; nearest reproduces
  GEE to ~1e-4 (t2m 0.0001 K) while bilinear smears across cell boundaries (~0.26 K)
  for no benefit. Validated across the 8-day window vs `PR_20250406` (PARITY_SPIKE_NOTES
  §7). Read the `valid_time` axis explicitly to resolve the precip slice for a given
  day — do **not** assume positional `tp[day_of_month-1]` aligns to that day; the
  `i+1` slice may live in the **next month's** precip file.
- **Business:** Raw Kelvin — do NOT replicate the `Normalizer` temperature-shift bug
  here. Precip day-shift (`i+1` `00:00` slice) is **mandatory** and applies to
  `total_precipitation_sum` only; temps/winds are unshifted. The archive is already
  daily — no hourly re-aggregation.
- **Out of scope:** Normalization, the temperature-sign fix (SPEC §6 out of scope);
  regenerating daily files from hourly CDS data (reference only, not this archive).

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-12): golden-grid triple; `bands_out` = the five bands in order.
- [x] AC-2 (SPEC AC-20): five bands in Kelvin/native units read from the daily archive;
      missing day → `-9999`.
- [x] AC-2b (SPEC AC-20b): **precip day-shift** — precip for day `d` equals the `d+1`
      `00:00` accumulation slice; `temperature_2m` for day `d` is read from the day-`d`
      slice (no shift).
- [x] AC-3 (SPEC AC-13): missing `(ERA5, day)` → all-`-9999` of declared shape.
- [x] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_era5_adapter.py -v
uv run ruff check src/data/local_sources/era5.py
uv run mypy src/data/local_sources/era5.py
```
Expected: adapter test green; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/era5.py tests/test_local_sources/test_era5_adapter.py
   git commit -m "feat(bow-valley): ERA5-Land adapter (raw Kelvin, precip i+1 day-shift) — closes TASK-008"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-009.
