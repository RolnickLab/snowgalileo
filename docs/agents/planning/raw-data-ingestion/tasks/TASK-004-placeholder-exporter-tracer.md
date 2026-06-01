# TASK-004: Build the placeholder exporter and the tracer-bullet end-to-end test

## 1. Goal
Plumb the whole pipeline end to end with all-`-9999` placeholder adapters: a
`LocalSourceExporter` that assembles the canonical 308-band cube and a tracer test
that exports one cell × one window-end-day, reads it through `LandsatEvalDataset`,
runs `EncoderWithHead`, and asserts the nine PLAN §6 conditions. FSC is degenerate
but every shape, band name, and mask path is proven correct before any real adapter.

## 2. Context & References
- **FDD step:** §4.4 — "Build placeholder exporter + tracer test"; §3 names
  `test_tracer_end_to_end.py` as the Red entry point for Phase 3.
- **SPEC:** FR-17, FR-18, AC-13, AC-23, AC-24, AC-25, AC-26; Verification Plan step 4.
- **PLAN:** §4 LocalSourceExporter, §6 Tracer-bullet (nine assertions), §3 filename
  convention, §9 non-negotiables (band order).
- **Upstream tasks:** TASK-002 (clipped archive), TASK-003 (`base.py`, `layout.py`,
  `cube_cache.py`, `grid.py`, filename contract).
- **Exact dynamic band order (FR-17, from `layout.py`/`eo.py`):**
  `S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5 + MODIS cloud +
  S2 cloud + Landsat cloud`, then static `DEM, slope, aspect, WorldCover Map`.
  Result: 35 time-varying × 8 + 3 cloud × 8 + 4 static = **308 bands**.
- **Tracer assertions (PLAN §6 — assert all nine):**
  `space_time_high_res_x == (100,100,8,15)`, `space_time_med_res_x == (5,5,8,2)`,
  `space_time_low_res_x == (2,2,8,11)`, `time_x == (8,9)`, `space_x == (100,100,14)`,
  `static_x == (3,)`, FSC `(10,10) ∈ [0,1]`, masks set on `-9999`/threshold,
  filename parses to `window_end.month`.
- **Downstream (UNCHANGED):** `LandsatEvalDataset`, `EncoderWithHead`
  (`src/fsc/patch_predict.py`), `Normalizer`.
- **Relevant skills:** `software-dev` (Ports & Adapters), `geospatial`, `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_tracer_end_to_end.py` (Red): export one placeholder cube → read
      through `LandsatEvalDataset` → `EncoderWithHead`; assert the nine conditions.
- [ ] 2. Write/extend the band-name equality test (AC-26): exporter band list ==
      `create_ee_image` band list (via `layout.py` re-export from `eo.py`).
- [ ] 3. Implement placeholder adapters (one per modality, all returning `-9999`
      arrays of declared shape via the `create_placeholder` helper from `base.py`).
- [ ] 4. Implement `LocalSourceExporter.export(cell, window_end)`: iterate 8 days,
      call each time-varying adapter in fixed order, append static stack, assemble the
      308-band GeoTIFF in canonical order, write to
      `data/bow_valley_processing/cubes/` under the `PR_..._SC00.tif` filename.
- [ ] 5. Green: make the tracer + band-name tests pass with placeholders.
- [ ] 6. Refactor on green.

## 4. Requirements & Constraints
- **Technical:** Write multiband GeoTIFF with `rasterio`; EPSG:4326, scale=10,
  `-9999` nodata, dims ≈ 159×100 (latitude convergence at ~51°N, satisfies H,W ≥ 100
  for the loader crop). Use `layout.py` for band order — never retype it.
- **Business:** Band order, tensor shapes, mask semantics, and the `PR` filename are
  the fixed contract; downstream code is not modified. Cubes go only to
  `cubes/`; cache to `cube_cache/`.
- **Out of scope:** No real source reads, no parity, no value domains (placeholders
  only), no inference driver/mosaic (TASK-015). Real adapters replace placeholders in
  TASK-006…TASK-014.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-23): assembled tensors have the six exact shapes listed above.
- [ ] AC-2 (SPEC AC-24): `EncoderWithHead` returns FSC shape `(10,10)`, values ∈ `[0,1]`.
- [ ] AC-3 (SPEC AC-25): `valid_data_mask_*` set wherever inputs are `-9999`/below
      `CHANNEL_WISE_INVALID_DATA_THRESHOLDS` (here: everywhere, since all placeholder).
- [ ] AC-4 (SPEC AC-13): a missing `(source, day)` placeholder path returns an
      all-`-9999` array of declared shape for every time-varying adapter.
- [ ] AC-5 (SPEC AC-26): exporter GeoTIFF band-name list == `create_ee_image` band-name
      list (byte-for-byte band-name equality).
- [ ] AC-6: emitted filename matches the FR-18 regex and parses to `window_end.month`.
- [ ] AC-7: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_tracer_end_to_end.py -v
uv run pytest tests/test_local_sources/test_filename_contract.py -v

# Build one placeholder cube and inspect band count
uv run python -m src.data.local_sources.exporter --cell 0 --window-end 2025-04-06 \
    --placeholder
uv run python -c "import rasterio; \
src=rasterio.open(sorted(__import__('pathlib').Path('data/bow_valley_processing/cubes').glob('PR_*.tif'))[-1]); \
print('bands', src.count); assert src.count == 308"

uv run ruff check src/data/local_sources/exporter.py
uv run mypy src/data/local_sources/exporter.py
```
Expected: tracer test green (all nine assertions), band count == 308, filename parses,
ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify every AC in Section 5.
2. Run all Section 6 commands; confirm expected output.
3. Commit:
   ```bash
   git add src/data/local_sources/exporter.py src/data/local_sources/*.py \
           tests/test_local_sources/test_tracer_end_to_end.py
   git commit -m "feat(bow-valley): placeholder exporter + tracer-bullet end-to-end test — closes TASK-004"
   ```
4. Check off subtasks/ACs; note deviations.
5. Notify the user (pipeline plumbed end-to-end, degenerate FSC) and request approval
   before TASK-005.
