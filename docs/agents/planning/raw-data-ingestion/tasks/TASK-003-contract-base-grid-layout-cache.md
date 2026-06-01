# TASK-003: Define the adapter contract — base.py, productionized grid.py, layout.py, cube_cache.py

## 1. Goal
Establish the shared interfaces every later task depends on: the `LocalSourceAdapter`
contract, the `GridCell`/`CellWindow` data model, the productionized grid generator
(modes A/B + cross-product CSV + kept/dropped manifest), the canonical band-order
constants re-exported from `eo.py`, and the per-(modality, cell, day) `.npz` cache.

## 2. Context & References
- **FDD step:** §4.3 — "Define the contract".
- **SPEC:** FR-6, FR-7, FR-8, FR-19, FR-19b, FR-20, FR-20b, AC-9, AC-10, AC-11, AC-11b;
  Verification Plan step 3.
- **PLAN:** §4 Module layout, §4 Adapter contract, §4 Cube cache layout, §3 Directory
  layout, §3 filename convention.
- **Upstream tasks:** TASK-001 (geometry half of `grid.py` + `cube_cells.csv` exist;
  this task productionizes them), TASK-002 (clipped archive is the `archive_root`).
- **Contract excerpt (PLAN §4 — implement verbatim):**
  ```python
  class LocalSourceAdapter(Protocol):
      bands_out: list[str]
      spatial_kind: Literal["high","med","low","time","space","static"]
      native_fill: float | None     # -28672 for MODIS; else None
      def fetch(self, cell: GridCell, day: datetime.date | None) -> np.ndarray:
          ...  # (C, H, W); -9999 nodata
  ```
- **`base.py` enforces:** reproject to cell target grid (EPSG:4326, scale=10) —
  **bilinear** continuous, **nearest** QA/categorical; missing acquisition → `-9999`
  array of declared shape; **same-tile/date coalesce runs before cross-tile
  mosaic-before-crop** (see TASK-013/TASK-012 for the scene sources that use it).
- **Filename contract (PLAN §3):** exporter emits
  `PR_{YYYYMMDD_window_end}_{LAT}_{LON}_SC00.tif`; regex
  `^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$`; must satisfy
  `prediction_month_from_file` (`landsat_eval.py:171-176`, `PR` branch → `parts[1][4:6]`).
- **Directory contract (PLAN §3):** all Stage 2 writes under
  `data/bow_valley_processing/`; cache → `cube_cache/`. `processing_root` +
  `archive_root` + cache cap are `cube.yaml` settings.
- **Relevant skills:** `software-dev` (Ports & Adapters, pydantic-settings, Protocol/ABC),
  `geospatial` (CRS triple, resampling), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_filename_contract.py` (Red): a set of synthetic
      `(window_end, lat, lon)` triples → exporter filename → regex match **and**
      `prediction_month_from_file` returns `window_end.month`.
- [ ] 2. Promote `test_grid.py` (Red, extend TASK-001): mode A/B switch, cross-product
      CSV emission, kept/dropped manifest, non-overlap — now as the production module.
- [ ] 3. Implement `base.py`: `LocalSourceAdapter` Protocol/ABC, `GridCell`
      (polygon, CRS, target transform, shape), `CellWindow`, the shared resampler
      (bilinear/nearest dispatch), and the `create_placeholder` `-9999` helper.
- [ ] 4. Productionize `grid.py`: mode A (in-AOI legacy-CSV cells, geometry only) and
      mode B (tile `data/aoi.geojson`); both bounded by the AOI; emit the cross-product
      CSV + kept/dropped manifest. Re-use TASK-001 geometry.
- [ ] 5. Implement `layout.py`: re-export the canonical dynamic + static band-order
      lists **from `src/data/earthengine/eo.py`** (single source of truth; do not
      retype band names).
- [ ] 6. Implement `cube_cache.py`: per-(modality, cell, day) `.npz` get/put under
      `data/bow_valley_processing/cube_cache/`, FIFO eviction with configurable cap.
- [ ] 7. Add `configs/bow_valley/cube.yaml` with `archive_root`
      (`…/clipped_bow_valley_selection_raw`), `processing_root`
      (`…/bow_valley_processing`), `mode`, `window`, `crs`, cache cap.
- [ ] 8. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** `numpy.typing` for arrays, `typing.Annotated`/`Protocol`, pydantic-settings
  for `cube.yaml`, `pyproj`/`rasterio` for the CRS triple, `polars` for the CSV.
  No band names hardcoded in `layout.py` — import from `eo.py`.
- **Business:** Adapters read `archive_root` (clipped). The raw path appears **only**
  in the clip-stage config. Cache + assembled cubes never written outside
  `processing_root` subdirs.
- **Out of scope:** No real adapter logic (TASK-004 placeholder, TASK-006+ real), no
  exporter assembly (TASK-004), no model/inference. Coalesce/mosaic algorithms are
  declared in `base.py` as the contract but implemented per-adapter in TASK-012/013.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-9): every exporter filename matches the regex and
      `prediction_month_from_file` returns `window_end.month`.
- [ ] AC-2 (SPEC AC-10): `grid.py` mode A → 344 centre-in / 338 fully-inside; manifest
      sums to 500.
- [ ] AC-3 (SPEC AC-11): grid cells non-overlapping (pairwise area == 0).
- [ ] AC-4 (SPEC AC-11b): cross-product CSV schema + row count + `crs == EPSG:32611`
      verified (re-confirmed against the productionized emitter).
- [ ] AC-5: `layout.py` band lists are **identical objects/values** to `eo.py`'s
      (asserted by an equality test importing both).
- [ ] AC-6: `cube_cache.py` round-trips an array (`put` then `get` returns equal data)
      and evicts FIFO when the cap is exceeded.
- [ ] AC-7: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_filename_contract.py -v
uv run pytest tests/test_local_sources/test_grid.py -v
uv run pytest tests/test_local_sources/test_cube_cache.py -v

uv run ruff check src/data/local_sources/
uv run mypy src/data/local_sources/base.py src/data/local_sources/grid.py \
            src/data/local_sources/layout.py src/data/local_sources/cube_cache.py
```
Expected: all three test files green; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify every AC in Section 5.
2. Run all Section 6 commands; confirm expected output.
3. Commit:
   ```bash
   git add src/data/local_sources/base.py src/data/local_sources/grid.py \
           src/data/local_sources/layout.py src/data/local_sources/cube_cache.py \
           configs/bow_valley/cube.yaml tests/test_local_sources/test_filename_contract.py \
           tests/test_local_sources/test_grid.py tests/test_local_sources/test_cube_cache.py
   git commit -m "feat(bow-valley): adapter contract, grid generator, layout, cube cache — closes TASK-003"
   ```
4. Check off subtasks/ACs; note deviations.
5. Notify the user and request approval before TASK-004.
