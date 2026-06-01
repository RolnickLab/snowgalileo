# TASK-002: Implement the AOI clip stage (Phase 0.5)

## 1. Goal
Crop every raw dataset in `data/bow_valley_selection_raw` to `data/aoi.geojson`,
non-destructively, into `data/clipped_bow_valley_selection_raw` — the single archive
root every adapter reads. The stage gates each product through a two-stage intersect
check, emits a per-source manifest, and passes a post-run zero-all-nodata audit.

## 2. Context & References
- **FDD step:** §4.2 — "Implement the AOI clip stage (Phase 0.5)".
- **SPEC:** FR-1…FR-5, AC-1…AC-8; Verification Plan step 2.
- **PLAN:** §3 Pipeline Stages (clip is on-path, hard prerequisite), §3 Directory
  layout (clip stage writes **only** `clipped_bow_valley_selection_raw`), §7 Phase 0.5.
- **Detailed reference:** `CLIPPING_PLAN.md` (this directory) — §2.0 intersect gate,
  §2.7 per-grid MODIS/VIIRS clipping, §3 CLI.
- **Upstream task:** TASK-001 (archive audit catalogs the formats this stage reads).
- **Key files:**
  - `data/aoi.geojson` — binding extent.
  - `scripts/developer_scripts/clip_dataset.py` — **new** Typer CLI.
  - Raw archive layout (per `DATA_ANALYSIS.md` §"Raw Archive Directory Formats"):
    DEM (nested SAFE GeoTIFF), ERA5 (NetCDF), Landsat8/9 (`.tar` GeoTIFF, EPSG:32612),
    S1 (`.zip` SAFE TIFF), S2 (`.zip` SAFE JP2, EPSG:32611), S3 (`.zip` SEN3 NetCDF),
    MODIS (HDF4 sinusoidal, two grids), VIIRS (HDF5, two grids), WorldCover (GeoTIFF).
- **Relevant skills:** `geospatial` (windowed reads, nodata, NN for categorical),
  `software-dev` (Typer, pathlib, structlog), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_clip_dataset.py` (Red): synthetic footprint fully outside AOI →
      `SKIP_NO_OVERLAP` + no output file; intersection below `MIN_AOI_OVERLAP_AREA_KM2`
      (default 1 km²) → `SKIP_DEGENERATE_OVERLAP` + no file; real ~8% Landsat scene →
      `CLIP` with >0 valid pixels; per-grid MODIS extents (500 m index ≈ 2× the 1 km
      index, no half-band truncation); clipped Landsat `crs == EPSG:32612`, clipped S2
      `crs == EPSG:32611`; non-destructive pixel equality inside AOI; manifest one row
      per product; post-run zero-all-nodata audit.
- [ ] 2. Implement the two-stage **intersect gate**: (1) metadata-only footprint-vs-AOI
      polygon intersection; (2) `MIN_AOI_OVERLAP_AREA_KM2` + post-clip valid-pixel check.
      Failing products produce **no output file**.
- [ ] 3. Implement per-modality clip routines preserving native CRS, pixel values, and
      file format. MODIS/VIIRS index **each native grid** (1200²/2400²) from its own
      resolution/origin — never a hardcoded `1200` clamp. NN resampling for categorical/QA.
- [ ] 4. Emit the per-source clip manifest: one row per input product with
      `{product_id, footprint_bbox, intersects, aoi_overlap_km2, valid_pixel_count,
      action}`, `action ∈ {CLIP, SKIP_NO_OVERLAP, SKIP_DEGENERATE_OVERLAP}`.
- [ ] 5. Implement the post-run audit script: assert **zero** all-nodata / zero-valid
      outputs; re-check DEM/WorldCover mosaic reaches `lat 52.31`; manifest accounts
      for every input product.
- [ ] 6. Green: implement until all clip tests pass; Refactor on green.

## 4. Requirements & Constraints
- **Technical:** `rasterio` windowed reads (never full-scene loads for clipping);
  system `gdalinfo`/`gdal_translate` for HDF4 (rasterio's GDAL lacks the HDF4 driver).
  `MIN_AOI_OVERLAP_AREA_KM2` is a pydantic-settings constant (default 1 km²), not a
  magic number.
- **Business:** Non-destructive — clipped pixels inside the AOI equal raw pixels (no
  resampling/rescaling). Landsat stays in native EPSG:32612 (cross-zone reprojection
  to the 4326 cell grid happens later, in the Landsat adapter — TASK-012). Fail-safe
  on corrupt SAFE/HDF: skip + manifest row, never a partial output.
- **Out of scope:** No reprojection to the cell grid, no band assembly, no `-9999`
  placeholder fabrication (that is the adapter layer). This stage only crops extents.
- **Write boundary:** writes **only** `data/clipped_bow_valley_selection_raw`. Never
  writes into `data/bow_valley_selection_raw` or `data/bow_valley_processing`.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-1): outside-AOI footprint → `SKIP_NO_OVERLAP`, no output file.
- [ ] AC-2 (SPEC AC-2): sub-threshold overlap → `SKIP_DEGENERATE_OVERLAP`, no file.
- [ ] AC-3 (SPEC AC-3): ~8% Landsat scene → `CLIP`, output clipped to intersection,
      >0 valid pixels.
- [ ] AC-4 (SPEC AC-4): post-run audit finds **zero** all-nodata outputs.
- [ ] AC-5 (SPEC AC-5): manifest has exactly one row per input product with correct
      `action`, `aoi_overlap_km2`, `valid_pixel_count`.
- [ ] AC-6 (SPEC AC-6): for one MOD09GA file both grid outputs cover the same AOI
      corner; 500 m index ≈ 2× the 1 km index.
- [ ] AC-7 (SPEC AC-7): clipped Landsat `crs == EPSG:32612`; clipped S2 `crs == EPSG:32611`.
- [ ] AC-8 (SPEC AC-8): non-destructive — sampled clipped pixels equal raw pixels.
- [ ] AC-9: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_clip_dataset.py -v

# Dry-run the gate on a single modality (no writes)
uv run python scripts/developer_scripts/clip_dataset.py --modality landsat9 \
    --aoi data/aoi.geojson --dry-run

# Full clip of one small modality, then audit
uv run python scripts/developer_scripts/clip_dataset.py --modality worldcover \
    --aoi data/aoi.geojson --out data/clipped_bow_valley_selection_raw
uv run python scripts/developer_scripts/clip_audit.py \
    --root data/clipped_bow_valley_selection_raw

uv run ruff check scripts/developer_scripts/clip_dataset.py
uv run mypy scripts/developer_scripts/clip_dataset.py
```
Expected: clip tests green; manifest written with one row per product; audit reports
zero all-nodata outputs and exits 0; ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify every AC in Section 5.
2. Run all Section 6 commands; confirm expected output.
3. Commit:
   ```bash
   git add scripts/developer_scripts/clip_dataset.py scripts/developer_scripts/clip_audit.py \
           tests/test_clip_dataset.py
   git commit -m "feat(bow-valley): AOI clip stage with intersect gate + manifest + audit — closes TASK-002"
   ```
4. Check off subtasks/ACs; note deviations.
5. **Approval gate (PLAN §7 Phase 0.5):** notify the user with the manifest summary
   (CLIP/SKIP counts, audit result) and request approval before TASK-003.
