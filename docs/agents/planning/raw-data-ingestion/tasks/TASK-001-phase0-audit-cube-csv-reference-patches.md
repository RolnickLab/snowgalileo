# TASK-001: Audit archive, emit the generated cube CSV, and build GEE reference patches (Phase 0)

## 1. Goal
Produce the Phase-0 prerequisites that every later task consumes: a written
archive audit, the **generated cross-product cube CSV** (`configs/bow_valley/cube_cells.csv`),
and a small set of **GEE reference patches** in `tests/fixtures/gee_reference_patches/`.
After this task, the inference sweep enumeration exists as a file and every parity
test has ground-truth fixtures to diff against.

## 2. Context & References
- **FDD step:** §4.1 — "Audit archive + generate cube CSV + GEE reference patches (Phase 0)".
- **SPEC:** FR-19, FR-19b, AC-10, AC-11, AC-11b, AC-30; Verification Plan step 1.
- **PLAN:** §3 "Generated cube CSV", §7 Phase 0, §3 Temporal window, §3 AOI (two
  definitions — `data/aoi.geojson` is authoritative).
- **Memory:** `bow-valley-inference-csv-decision` — the cell/date input is a
  generated CSV, NOT the legacy training CSV.
- **Key files:**
  - `sampled_cells_bow_river_with_dates.csv` (repo root) — **cell geometry only**;
    its `date` column is train/eval metadata and is NOT read here.
  - `data/aoi.geojson` — authoritative clip/inference boundary (EPSG:4326):
    `lon [-116.561936, -114.527659]`, `lat [50.729807, 52.306672]`.
  - `data/bow_valley_selection_raw/` — raw archive (read-only) to audit.
  - `scripts/export_for_inference.py` → `EarthEngineExporterEval.export_from_csv_utm`
    (`src/data/earthengine/eo_eval.py:576`) — the CSV-driven GEE exporter. **NOT
    `export_for_eval.py`.**
- **Contract excerpt — `export_from_csv_utm` reads exactly these columns**
  (`eo_eval.py:577-585`), so the generated CSV schema is fixed:
  ```
  date, crs, center_x, center_y, min_x, min_y, max_x, max_y
  ```
  It builds `filename = f"PR_{date}_{center_x:.16f}_{center_y:.16f}.tif"` and
  `WINDOW_END_DATE = strptime(date, "%Y%m%d")`, `WINDOW_START = END - (NUM_TIMESTEPS-1)`.
- **Relevant skills:** `geospatial` (CRS is law, GeoParquet/COG), `software-dev`
  (pydantic-settings, pathlib, polars, structlog), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_grid.py` geometry tests (Red): centre-in rule → **344** kept
      cells, `--require-fully-inside` → **338**, kept+dropped manifest sums to 500,
      pairwise cell intersection area == 0, every kept centre within `data/aoi.geojson`.
- [ ] 2. Write `test_cube_csv.py` (Red): generated CSV has the 8-column canonical
      schema, row count == kept-cells × window-days (full cross-product), every
      `crs == "EPSG:32611"`, and `export_from_csv_utm` parses it without error
      (call with a tmp dry-run / column-presence assertion against `eo_eval.py:577-585`).
- [ ] 3. Implement the **geometry half** of `grid.py` (pure CRS/polygon math, no
      adapters): load legacy CSV for `center_x/y` + bounds, reproject cells to
      EPSG:4326, apply the centre-in AOI filter, emit kept/dropped manifest.
- [ ] 4. Emit `configs/bow_valley/cube_cells.csv` — full cross-product of in-AOI
      cells × every day in the default window `2025-04-06 → 2025-05-28`, canonical
      schema, EPSG:32611. This file IS the sweep enumeration.
- [ ] 5. Write the archive audit: catalog per-modality paths/formats/CRS/native
      scale; verify ingest coverage of `start-7 → end` (`2025-03-30 → 2025-05-28`);
      assert DEM (9 `*_DEM.tif` tiles) and WorldCover (4 `*_Map.tif` tiles) mosaics reach `lat_max=52.31`;
      profile per-in-AOI-cell per-source 8-day-window completeness; **report the
      S1-fully-masked-window rate** across the inference range (S1 on ~16 dates only).
- [ ] 6. Verify the `PR` filename prefix meaning in `landsat_eval.py:172` and record
      the finding in the audit (RESOLVED: `PR` is parser-supported, unused on disk).
- [ ] 7. Sample **5–10 rows** of `cube_cells.csv` and run
      `scripts/export_for_inference.py` to write `tests/fixtures/gee_reference_patches/`.
- [ ] 8. Write `docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md` with the catalog,
      coverage matrix, S1-gap profile, and the `lat 52.31` assertion result.

## 4. Requirements & Constraints
- **Technical:** EPSG:32611 grid math; reproject to EPSG:4326 for the AOI test.
  Use `polars` for the cross-product, `pyproj.Transformer(always_xy=True)`,
  `rasterio`/`shapely` for footprint geometry, `structlog` JSON logs. System
  `gdalinfo` for HDF4 catalog (rasterio's GDAL build lacks the HDF4 driver).
- **Business:** The generated CSV's `date` column is the configured inference
  window enumerated — NOT per-cell label days. The legacy CSV `date` is never read.
  Sampling reference rows FROM the generated CSV guarantees in-AOI/in-archive
  parity cells by construction.
- **Out of scope:** No adapters, no clip, no exporter, no model. Productionizing
  `grid.py` (mode A/B switch, `cube_cache.py` wiring) is TASK-003 — this task ships
  only the geometry half + CSV emission needed for Phase 0.
- **Flag (filename discrepancy, do not silently reconcile):** `export_from_csv_utm`
  emits `PR_{date}_{cx}_{cy}.tif` (3 fields, UTM coords), but the **exporter contract**
  in FR-18/AC-9 is `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (5 fields, degrees). The GEE
  reference fixtures use the former; the new `LocalSourceExporter` (TASK-004) must
  emit the latter to satisfy `prediction_month_from_file`. Both parse via the `PR`
  branch (`parts[1][4:6]`). Record this in the audit; resolve filename ownership in
  TASK-004, not here.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-10): `grid.py` mode-A geometry yields **344** centre-in cells,
      **338** fully-inside cells; kept+dropped manifest sums to 500.
- [ ] AC-2 (SPEC AC-11): pairwise grid-cell intersection area == 0.
- [ ] AC-3 (SPEC AC-11b): `configs/bow_valley/cube_cells.csv` has the canonical
      8-column schema, row count == kept-cells × window-days, every `crs == EPSG:32611`,
      and `export_from_csv_utm` consumes it without raising.
- [ ] AC-4 (SPEC AC-30): per-cell per-source 8-day-window coverage fractions are
      written, and the S1-fully-masked-window rate over the inference range is reported.
- [ ] AC-5: `ARCHIVE_AUDIT.md` asserts DEM/WorldCover mosaics reach `lat 52.31` and
      that `2025-03-30 → 2025-05-28` is covered for every modality.
- [ ] AC-6: `tests/fixtures/gee_reference_patches/` contains 5–10 exported patches
      drawn from rows of `cube_cells.csv`.
- [ ] AC-7: All new code passes `ruff check` and `mypy` with zero errors.
- [ ] AC-8: Targeted new tests green; full suite introduces NO new failures vs
      `TEST_BASELINE.md` (delta check, NOT `pytest -x` — the suite is already red).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
# Red → Green: grid geometry + CSV schema
uv run pytest tests/test_local_sources/test_grid.py -v
uv run pytest tests/test_local_sources/test_cube_csv.py -v

# Emit the generated cube CSV (writes configs/bow_valley/cube_cells.csv)
uv run python -m src.data.local_sources.grid --emit-csv --mode A \
    --window-start 2025-04-06 --window-end 2025-05-28

# Confirm schema + row count
uv run python -c "import polars as pl; df=pl.read_csv('configs/bow_valley/cube_cells.csv'); \
print(df.columns); print(df.height); assert (df['crs']=='EPSG:32611').all()"

# Confirm the GEE exporter accepts the schema (dry validation of column presence)
uv run python -c "import pandas as pd; c=pd.read_csv('configs/bow_valley/cube_cells.csv').columns; \
assert set(['date','crs','center_x','center_y','min_x','min_y','max_x','max_y']).issubset(c)"

# Generate reference patches (5–10 sampled rows)
uv run python scripts/export_for_inference.py --path_to_csv <sampled_rows.csv>

uv run ruff check src/data/local_sources/grid.py
uv run mypy src/data/local_sources/grid.py
```
Expected: both test files green; `cube_cells.csv` exists with 8 columns and
`344 × 53 ≈ 18 200` rows (mode A default window); `ARCHIVE_AUDIT.md` present;
fixtures directory non-empty; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify every AC in Section 5 is checked.
2. Run all Section 6 commands; confirm expected output.
3. Commit:
   ```bash
   git add src/data/local_sources/grid.py tests/test_local_sources/test_grid.py \
           tests/test_local_sources/test_cube_csv.py configs/bow_valley/cube_cells.csv \
           docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md tests/fixtures/gee_reference_patches
   git commit -m "feat(bow-valley): Phase 0 audit + generated cube CSV + GEE reference patches — closes TASK-001"
   ```
4. Check off subtasks/ACs; note the filename-discrepancy finding inline.
5. Notify the user with a summary (kept-cell count, CSV row count, S1-gap rate) and
   request approval before TASK-002.
