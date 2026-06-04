# TASK-009: Implement the MODIS MOD09GA adapter (preserve -28672 fill)

## 1. Goal
Replace the MODIS placeholder with a real adapter that emits `sur_refl_b01..b07` on
the cell grid from the 500 m sinusoidal grid, **preserving the native `-28672` fill
value** in addition to `-9999`.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #4 — "mind the `-28672` fill").
- **SPEC:** FR-10, AC-12, AC-13, AC-18; Verification Plan step 6.
- **PLAN:** §4 adapter rule (preserve `-28672`), §6 FMEA ("MODIS native fill stripped").
- **Upstream tasks:** TASK-002 (clipped MODIS, per-grid extents), TASK-003, TASK-004.
- **Clipped extent is a diagonal band, not a filled rectangle (TASK-002 fix,
  2026-06-03).** MODIS is clipped by AOI *geometry* in Sinusoidal CRS, so the per-grid
  GeoTIFFs are ~66 % nodata (the sheared corners outside the AOI) — only ~33.7 % is
  valid data over this AOI. This is correct, not a clip defect. The adapter's
  nodata-aware bilinear (below) already masks `-28672`/`-9999` to NaN before warping, so
  the larger nodata fraction is handled; just don't assume the clipped tile is fully
  populated when reasoning about valid-pixel counts.
- **Why `-28672` is load-bearing:** `src/fsc/landsat_eval.py:317,331` asserts the MODIS
  fill value is *encountered* in NDSI/NDVI (`assert (ndsi != MODIS_FILL_VALUE).any()`,
  sentinel for "MODIS data was actually present"). Stripping it **crashes the loader**.
  Verified from `landsat_eval.py` read.
- **Source semantics (DATA_ANALYSIS.md §MODIS + §Verified Catalog):**
  - HDF4, **two co-registered sinusoidal grids**: 1 km (`MODIS_Grid_1km_2D`, 1200²,
    holds `state_1km`) and 500 m (`MODIS_Grid_500m_2D`, 2400², holds `sur_refl_b01..b07`).
  - Science bands come from the **500 m** grid; index each grid at its own resolution.
  - `uint16`; native `_FillValue = -28672`.
  - Reproject sinusoidal → cell grid (**EPSG:32611** UTM 11N — CORRECTED 2026-06-04
    from "EPSG:4326"; see KNOWLEDGE.md) via `base.reproject_to_cell`; mosaic tiles
    when a cell crosses a tile boundary (cross-tile mosaic-before-crop).
  - Preserve value convention (integer-like MODIS range), preserve fill so the
    `MODIS_FILL_VALUE` / `>= -100` threshold checks work.
  - `spatial_kind="low"` (loader downsamples to 2×2). NDSI/NDVI are derived downstream.
  - **System `gdalinfo`/`gdal_translate` for HDF4** (rasterio's GDAL lacks HDF4 driver).
  - Cloud flag `state_1km` is emitted separately in the cloud-flag slot.
- **Relevant skills:** `geospatial` (sinusoidal reproject, mosaic, NN for QA), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_modis_adapter.py` (Red): golden-grid triple; `bands_out =
      sur_refl_b01..b07`; **`-28672` present** in output where the source had it; missing
      day → all-`-9999`; reads the 500 m grid (not the 1 km clamp); **nodata-aware
      bilinear: no interpolated value bleeds toward `-28672` at a fill edge (assert
      no out-of-domain negative appears adjacent to a fill pixel)**.
- [ ] 2. Implement `modis.py`: read 500 m subdatasets via `gdal_translate`, mosaic tiles,
      reproject sinusoidal→cell grid, stack `(7, H, W)`; `native_fill=-28672`.
- [ ] 3. Implement the `state_1km` cloud-flag path (NN), emitted in the cloud slot.
- [ ] 4. Wire into exporter. 5. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Per-grid indexing (no hardcoded `1200`); **nodata-aware**
  bilinear for science bands, NN for `state_1km`; system GDAL HDF4 driver.
- **Nodata-aware bilinear (edge-bleed guard).** Before the bilinear warp, mask
  `-28672` and `-9999` pixels to NaN so the interpolator never blends a valid
  reflectance with the fill sentinel; restore `-28672`/`-9999` in the output where
  the contributing source pixels were fill. A naive bilinear across a valid value
  and `-28672` yields a garbage negative (e.g. `-5000`) that slips past
  `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`. This logic lives in the shared `base.py`
  resampler (TASK-003) — MODIS is its highest-risk consumer, not its only one.
  (REVIEW_AUDIT.md verdict #4.)
- **Business:** `-28672` MUST survive into the output (loader sentinel). Do not apply
  the MODIS scale factor (changes the numeric domain vs normalization constants).
- **Out of scope:** NDSI/NDVI derivation (loader), VIIRS (TASK-010).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-12): golden-grid triple; `bands_out` = `sur_refl_b01..b07` in order.
- [ ] AC-2 (SPEC AC-18): `-28672` present in output where source had it; loader NDSI/NDVI
      assertions (`landsat_eval.py:317,331`) do not crash on this adapter's output.
- [ ] AC-3 (SPEC AC-13): missing `(MODIS, day)` → all-`-9999`.
- [ ] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_modis_adapter.py -v
uv run ruff check src/data/local_sources/modis.py
uv run mypy src/data/local_sources/modis.py
```
Expected: adapter test green (fill preserved); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/modis.py tests/test_local_sources/test_modis_adapter.py
   git commit -m "feat(bow-valley): MODIS MOD09GA adapter (preserve -28672 fill) — closes TASK-009"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-010.
