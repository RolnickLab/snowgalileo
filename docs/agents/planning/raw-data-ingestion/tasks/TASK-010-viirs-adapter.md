# TASK-010: Implement the VIIRS VNP09GA adapter (fine + coarse, per-pixel raster)

## 1. Goal
Replace the VIIRS placeholder with a real adapter that emits fine bands `[I1, I3]`
(`space_time_low_res_x`) and coarse bands `[M5, M7, M10, M11]` as a **per-pixel raster**
`(4, H, W)` (`time_x`) — never pre-averaged in the adapter.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #5).
- **SPEC:** FR-13, AC-12, AC-13, AC-19; Verification Plan step 6.
- **PLAN:** §4 module note ("emits both fine (per-pixel) and coarse (per-pixel raster,
  loader averages)"), §9 ("VIIRS coarse exported as per-pixel rasters; loader does the
  spatial mean into `time_x`"), §6 FMEA.
- **Upstream tasks:** TASK-002 (clipped VIIRS, per-grid extents), TASK-003, TASK-004.
- **Source semantics (DATA_ANALYSIS.md §VIIRS + §Verified Catalog):**
  - HDF5, **two co-registered grids**: `VIIRS_Grid_500m_2D` (2400², fine `I1/I3`) and
    `VIIRS_Grid_1km_2D` (1200², coarse `M5/M7/M10/M11`).
  - Fine `I1/I3` → `space_time_low_res_x` (loader downsamples with the low-res group).
  - Coarse `M*` → `time_x`; emit as `(4, H, W)` per-pixel raster; **loader** does the
    spatial mean. Do NOT pre-average.
  - `int16`/`uint16`; I-bands carry `_FillValue = -28672` (preserve, same loader reason
    as MODIS).
  - Baseline normalization downstream: `(x + 0.795) / 0.805`; valid `>= -0.01`.
  - Reproject sinusoidal→cell grid; mosaic tiles. System GDAL for HDF5 if needed.
- **Relevant skills:** `geospatial` (per-grid reproject, mosaic), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_viirs_adapter.py` (Red): fine emits `(2, H, W)` `[I1, I3]`; coarse
      emits `(4, H, W)` `[M5,M7,M10,M11]` (per-pixel, NOT `(4,)`); loader spatial mean
      over the coarse raster reproduces GEE `time_x` values; missing day → `-9999`;
      I-band `-28672` preserved.
- [ ] 2. Implement `viirs.py`: read both grids, reproject each to the cell grid, return
      fine (`spatial_kind="low"`) and coarse (`spatial_kind="time"`, per-pixel) outputs.
- [ ] 3. Wire into exporter. 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Per-grid indexing; bilinear for reflectance; preserve `-28672`.
- **Business:** Coarse stays a per-pixel raster — pre-averaging breaks `time_x`.
- **Out of scope:** VIIRS QF1 cloud flag (not active in `MODALITIES`), S3 (TASK-011).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-12): golden-grid triple for both fine and coarse; band order correct.
- [ ] AC-2 (SPEC AC-19): fine `(2,H,W)`; coarse `(4,H,W)` per-pixel; loader spatial mean
      reproduces GEE `time_x`.
- [ ] AC-3 (SPEC AC-13): missing `(VIIRS, day)` → all-`-9999`.
- [ ] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_viirs_adapter.py -v
uv run ruff check src/data/local_sources/viirs.py
uv run mypy src/data/local_sources/viirs.py
```
Expected: adapter test green (coarse shape `(4,H,W)`); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/viirs.py tests/test_local_sources/test_viirs_adapter.py
   git commit -m "feat(bow-valley): VIIRS VNP09GA adapter (fine + coarse per-pixel) — closes TASK-010"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-011.
