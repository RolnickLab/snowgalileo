# TASK-015: Wire the InferenceGridDriver and DailyMosaicWriter

## 1. Goal
Add the inference orchestration: for each day in the configured window, build the
8-day window per cell, GPU-batch `EncoderWithHead` over cells, and mosaic the per-cell
10×10 FSC predictions into one daily COG in EPSG:32611, recording per-day AOI coverage.

## 2. Context & References
- **FDD step:** §4.7 — "Wire `InferenceGridDriver` + `DailyMosaicWriter`".
- **SPEC:** FR-21, FR-22, AC-28, AC-29, AC-31; Verification Plan steps 7 & 9.
- **PLAN:** §5 Inference Grid Driver (pseudocode), §3 Directory layout (daily COGs →
  `daily_fsc/`), §3 Grid+CRS (mosaic in EPSG:32611, NN reproject of 10×10 FSC), §8 Q4
  RESOLVED (driver ignores CSV `date`).
- **Upstream tasks:** TASK-004 (exporter), TASK-006…TASK-014 (all 9 real adapters),
  TASK-003 (`grid.py`, `cube.yaml`).
- **Driver loop (PLAN §5 — implement this shape):**
  ```python
  grid = build_grid(aoi_bbox_utm, cell_size_m=1000, crs="EPSG:32611", mode=mode_A_or_B)
  for day in daterange(start, end):                 # configured window, NOT CSV date
      cube_paths = [(cell, exporter.export(cell, window_end=day)) for cell in grid]  # parallel
      preds = run_encoder_with_head(model, batch_cells(cube_paths, N))  # (B,10,10) FSC
      mosaic.write_day(day, preds, grid)            # COG, EPSG:32611
  ```
- **Q4 (RESOLVED, AC-31):** the driver iterates the configured window × all in-AOI cells
  and does **not** read the CSV `date` column (it is train/eval metadata). The generated
  `cube_cells.csv` dates ARE the configured window enumerated.
- **Mosaic rules (FR-22):** one COG/day in EPSG:32611 → `data/bow_valley_processing/daily_fsc/`;
  each 10×10 FSC patch reprojected from EPSG:4326 with **nearest-neighbour** (FSC is a
  prediction — bilinear would blend invalid neighbours); stitch only valid predictions;
  all-masked cells → `nodata`; record per-day AOI-coverage fraction in metadata.
- **Downstream (UNCHANGED):** `EncoderWithHead`, `LandsatEvalDataset`, `Normalizer`.
- **Relevant skills:** `geospatial` (COG, NN reproject, seams), `software-dev`
  (multiprocessing, GPU batching), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_inference_driver.py` (Red): driver builds `[d-7, d]` per cell over
      the configured window; **AC-31** — two cells with different legacy-CSV `date` values
      are both predicted on the same configured inference day (CSV `date` has no effect).
- [ ] 2. Write the 2×2 mosaic seam test (Red, AC-29): four adjacent cells → non-overlapping
      seams (no double-written pixels); FSC reprojected NN only.
- [ ] 3. Write the COG-validity test (Red, AC-28): output is a valid COG in EPSG:32611;
      all-input-masked cell → `nodata`; per-day coverage fraction in metadata.
- [ ] 4. Implement `src/inference/windows.py` (sliding 8-day window builder),
      `driver.py` (`InferenceGridDriver`, parallel export + GPU batching), `mosaic.py`
      (`DailyMosaicWriter`, per-day COG, NN reproject, coverage metric).
- [ ] 5. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** `multiprocessing.Pool` for per-cell export; GPU-batched inference; COG
  via `rasterio` with overviews; NN reproject for FSC; `structlog` per-day coverage log.
- **Business:** Mosaic stays in EPSG:32611 (per-cell tifs in 4326 feed the loader
  unchanged; reprojection happens once at mosaic-write on the 10×10 FSC, low IO).
  Driver ignores CSV `date`. Heterogeneous daily coverage is structural (expected nodata).
- **Out of scope:** Entry-point scripts + full-stack parity gate (TASK-016). Per-cell
  cross-cell context (a documented modelling limitation, not fixed here).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-28): valid COG in EPSG:32611; all-masked cell → `nodata`; per-day
      AOI-coverage fraction recorded in metadata.
- [ ] AC-2 (SPEC AC-29): 2×2 mosaic non-overlapping seams; FSC NN-reprojected only.
- [ ] AC-3 (SPEC AC-31): driver iterates configured window × all in-AOI cells; two cells
      with different CSV `date` are predicted on the same configured day (CSV `date`
      ignored).
- [ ] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_inference_driver.py -v
uv run ruff check src/inference/
uv run mypy src/inference/driver.py src/inference/mosaic.py src/inference/windows.py

# Validate a produced daily COG
uv run python -c "import rasterio; from rasterio.errors import RasterioIOError; \
src=rasterio.open(sorted(__import__('pathlib').Path('data/bow_valley_processing/daily_fsc').glob('*.tif'))[-1]); \
assert src.crs.to_epsg()==32611; print('coverage', src.tags().get('aoi_coverage_fraction'))"
```
Expected: driver + mosaic tests green; daily COG is EPSG:32611 with coverage tag;
ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/inference/ tests/test_local_sources/test_inference_driver.py
   git commit -m "feat(bow-valley): InferenceGridDriver + DailyMosaicWriter — closes TASK-015"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-016.
