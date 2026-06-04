# TASK-006: Implement the ESA WorldCover adapter

## 1. Goal
Replace the WorldCover placeholder with a real adapter that returns the v200 (2021)
`Map` band as a single categorical channel on the cell grid, independent of `day`.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #1 — static, easiest first).
- **SPEC:** FR-16, AC-12, AC-22; Verification Plan step 6.
- **PLAN:** §4 adapter contract ("WorldCover ignores `day`, hardcoded"), §9 ("single
  `Map` band; loader one-hot encodes").
- **Upstream tasks:** TASK-002 (clipped WorldCover), TASK-003 (`base.py`, `GridCell`),
  TASK-004 (placeholder exporter to replace).
- **Source semantics (DATA_ANALYSIS.md §ESA WorldCover):**
  - Clipped GeoTIFF, **source CRS** `EPSG:4326`, single band `Map`, `uint8`.
  - **Reproject target = the cell grid `EPSG:32611` (UTM 11N)** — CORRECTED
    2026-06-04 from "EPSG:4326 target"; the cell grid is UTM, not 4326 (the same
    cascade as TASK-003/004/007/009/012; see `docs/agents/KNOWLEDGE.md`). The
    source is 4326; the adapter NN-reprojects it onto the `GridCell`'s UTM grid
    like every other adapter, so the assembled cube is single-CRS.
  - Class codes preserved exactly: `{10,20,30,40,50,60,70,80,90,95,100}`
    (this AOI contains a subset; `0` appears as nodata).
  - **Do NOT one-hot encode** — the loader does it. Emit one `Map` band.
  - Nodata/unknown → `0` or `-9999` only (loader maps those to `-9999` across one-hot).
  - **NN resampling only** (categorical) via `base.reproject_to_cell(categorical=True)`.
- **Relevant skills:** `geospatial` (NN for categorical, mosaic), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_worldcover_adapter.py` (Red): golden-grid `(transform, shape, crs)`
      triple; output is a single `Map` band; class codes ∈ allowed set; identical for
      two different `day` values (and `day=None`).
- [ ] 2. Implement `worldcover.py`: mosaic clipped tiles, NN-reproject to the cell grid,
      return shape `(1, H, W)`; `spatial_kind="space"`, `native_fill=None`.
- [ ] 3. Wire it into the exporter, replacing the placeholder.
- [ ] 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** NN resampling; `rasterio` mosaic; reads `archive_root` (clipped).
- **Business:** Static — ignores `day`. No one-hot, no class remapping (would break the
  loader's one-hot channel order).
- **Out of scope:** One-hot encoding (loader's job), DEM (TASK-007).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-12): golden-grid `(transform, shape, crs)` asserted; `bands_out`
      == `["Map"]` in order; **EPSG:32611 (UTM 11N) cell-grid target** (CORRECTED
      2026-06-04 from "EPSG:4326 target"; source is 4326, reproject target is the
      UTM cell grid — see §2).
- [ ] AC-2 (SPEC AC-22): single `Map` band, class codes ∈ `{10,…,95,100}` (not one-hot),
      independent of `day`.
- [ ] AC-3: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_worldcover_adapter.py -v
uv run ruff check src/data/local_sources/worldcover.py
uv run mypy src/data/local_sources/worldcover.py
```
Expected: adapter test green; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/worldcover.py tests/test_local_sources/test_worldcover_adapter.py
   git commit -m "feat(bow-valley): ESA WorldCover adapter — closes TASK-006"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-007.
