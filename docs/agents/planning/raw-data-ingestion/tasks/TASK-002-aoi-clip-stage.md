# TASK-002: Implement the AOI clip stage (Phase 0.5)

## 1. Goal
Crop every raw dataset in `data/bow_valley_selection_raw` to `data/bow_valley_inference_aoi.geojson`,
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
  - `data/bow_valley_inference_aoi.geojson` — binding extent.
  - `scripts/developer_scripts/clip_dataset.py` — **new** Typer CLI.
  - Raw archive layout (per `DATA_ANALYSIS.md` §"Raw Archive Directory Formats"):
    DEM (nested SAFE GeoTIFF), ERA5 (NetCDF), Landsat8/9 (`.tar` GeoTIFF, EPSG:32612),
    S1 (`.zip` SAFE TIFF), S2 (`.zip` SAFE JP2, EPSG:32611), S3 (`.zip` SEN3 NetCDF),
    MODIS (HDF4 sinusoidal, two grids), VIIRS (HDF5, two grids), WorldCover (GeoTIFF).
- **Relevant skills:** `geospatial` (windowed reads, nodata, NN for categorical),
  `software-dev` (Typer, pathlib, structlog), `tdd`.

## 3. Subtasks
- [x] 1. Write `test_clip_dataset.py` (Red): synthetic footprint fully outside AOI →
      `SKIP_NO_OVERLAP` + no output file; intersection below `MIN_AOI_OVERLAP_AREA_KM2`
      (default 1 km²) → `SKIP_DEGENERATE_OVERLAP` + no file; real ~8% Landsat scene →
      `CLIP` with >0 valid pixels; per-grid MODIS extents (the 500 m grid output ≈ 2×
      the 1 km grid on both axes, no half-band truncation); clipped Landsat
      `crs == EPSG:32612`, clipped S2 `crs == EPSG:32611`; non-destructive pixel
      equality inside AOI; manifest one row per product; post-run zero-all-nodata audit.
      NOTE: the per-grid ratio test asserts only the 2× grid ratio, **not** absolute
      dims — it passes for both the (correct) geometry-mask crop and the (buggy)
      corner-index-window crop, so it is not by itself proof of sinusoidal correctness
      (see subtask 3 / CLIPPING_PLAN §2.7 shear trap).
- [x] 2. Implement the two-stage **intersect gate** (`clip/gate.py`): (1) metadata-only
      footprint-vs-AOI polygon intersection; (2) `MIN_AOI_OVERLAP_AREA_KM2` + post-clip
      valid-pixel check. Failing products produce **no output file**.
- [x] 3. Implement per-modality clip routines (`clip/clippers.py`) preserving native CRS,
      pixel values, and file format. MODIS/VIIRS extract **each native grid** (1200²/2400²)
      to a per-grid GeoTIFF (sinusoidal CRS+transform preserved), then crop **by AOI
      geometry** with `rasterio.mask.mask(crop=True)` against the AOI reprojected into the
      subdataset's Sinusoidal CRS — NOT by a reprojected-corner index window.
      **Correction (2026-06-03):** the original implementation built an axis-aligned pixel
      window from the AOI's reprojected corners; in Sinusoidal (`x = R·λ·cos φ`) a lon/lat
      rectangle shears, so that bbox was ~5× too wide in X and kept a 10°-wide block of
      data instead of the AOI's ~2°-wide diagonal band. Fixed to geometry masking; MODIS +
      VIIRS re-clipped. See CLIPPING_PLAN §2.7 + `docs/agents/KNOWLEDGE.md`.
      **Second correction (2026-06-03):** the S3 OLCI clip compared `geo_coordinates`
      lat/lon (CF-scaled int32, `scale_factor ≈ 1e-6`) as **raw integers** against degree
      AOI bounds → empty mask → all 125 S3 products clipped with `(0,0)` radiance while the
      manifest still showed ~33 M valid pixels (counted from full-copied non-science
      datasets). Fixed with `_cf_scaled()` (decode scaling before the mask); S3 re-clipped.
      See CLIPPING_PLAN §2.6 + `docs/agents/KNOWLEDGE.md`. Both bugs were surfaced by the
      clip-viewer, not the test suite — a valid-pixel gate can't catch a wrong window when
      unrelated datasets pad the count.
- [x] 4. Emit the per-source clip manifest (`clip/manifest.py`): one row per input product
      with `{product_id, footprint_bbox, intersects, aoi_overlap_km2, valid_pixel_count,
      action}`, `action ∈ {CLIP, SKIP_NO_OVERLAP, SKIP_DEGENERATE_OVERLAP}`.
- [x] 5. Implement the post-run audit script (`clip_audit.py`): assert **zero** all-nodata
      / zero-valid outputs; re-check DEM/WorldCover mosaic reaches `lat 52.31`; manifest
      accounts for every input product.
- [x] 6. Green: all 10 clip tests pass; ruff + mypy clean on the `clip/` package.

## 4. Requirements & Constraints
- **Technical:** `rasterio` windowed reads (never full-scene loads for clipping);
  system `gdalinfo`/`gdal_translate` for HDF4 (rasterio's GDAL lacks the HDF4 driver).
  `MIN_AOI_OVERLAP_AREA_KM2` is a pydantic-settings constant (default 1 km²), not a
  magic number.
- **Business:** Non-destructive — clipped pixels inside the AOI equal raw pixels (no
  resampling/rescaling). Landsat stays in native EPSG:32612 (cross-zone reprojection
  to the **EPSG:32611** (UTM 11N) cell grid happens later, in the Landsat adapter —
  TASK-012; CORRECTED 2026-06-04 from "4326 cell grid", see KNOWLEDGE.md). Fail-safe
  on corrupt SAFE/HDF: skip + manifest row, never a partial output.
- **Landsat CRS — read the band header, don't assume.** Verified: every archive
  Landsat scene is natively `EPSG:32612` (`gdalinfo LC09_..._B4.TIF` →
  `ID["EPSG",32612]`; USGS assigns each WRS-2 path/row its own UTM zone regardless
  of AOI longitude). So `crs == 32612` here is correct, **not** the bug a prior
  review claimed (it reasoned from AOI longitude → UTM 11). Still, **query each
  band's CRS dynamically** and reproject the AOI to *that* CRS rather than
  hardcoding 32612 — defensive against a future mixed-zone pull. (REVIEW_AUDIT.md
  verdict #2.)
- **Out of scope:** No reprojection to the cell grid, no band assembly, no `-9999`
  placeholder fabrication (that is the adapter layer). This stage only crops extents.
- **Write boundary:** writes **only** `data/clipped_bow_valley_selection_raw`. Never
  writes into `data/bow_valley_selection_raw` or `data/bow_valley_processing`.

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-1): outside-AOI footprint → `SKIP_NO_OVERLAP`, no output file.
      (`test_gate_skips_disjoint_footprint`; verified live on 2 W120 WorldCover tiles.)
- [x] AC-2 (SPEC AC-2): sub-threshold overlap → `SKIP_DEGENERATE_OVERLAP`, no file.
      (`test_gate_skips_degenerate_overlap`.)
- [x] AC-3 (SPEC AC-3): Landsat scene → `CLIP`, clipped to intersection, >0 valid
      pixels. (`test_landsat_clip_keeps_zone_and_pixels`.)
- [x] AC-4 (SPEC AC-4): post-run audit finds **zero** all-nodata outputs.
      (`test_audit_passes_on_clipped_worldcover` + `clip_audit.py`.)
- [x] AC-5 (SPEC AC-5): manifest has exactly one row per input product with correct
      `action`, `aoi_overlap_km2`, `valid_pixel_count`. (`test_manifest_one_row_per_product`.)
- [x] AC-6 (SPEC AC-6): for one MOD09GA file the 500 m grid output is ~2× the 1 km grid
      on both axes (`test_modis_per_grid_index_ratio`; ±2 px). Absolute dims depend on the
      crop method and are no longer pinned — the geometry-mask crop yields the AOI's
      diagonal-band extent (~33.7 % valid fill), not the old corner-window block. The 2×
      *ratio* holds for both methods, so this AC alone does not prove sinusoidal-crop
      correctness; CLIPPING_PLAN §2.7 adds the per-row-span check.
- [x] AC-7 (SPEC AC-7): clipped Landsat `crs == EPSG:32612`; clipped S2 `crs == EPSG:32611`.
      (`test_landsat_clip_keeps_zone_and_pixels`, `test_sentinel2_clip_stays_utm11`.)
- [x] AC-8 (SPEC AC-8): non-destructive — sampled clipped pixel equals raw pixel.
      (`test_dem_clip_is_non_destructive`.)
- [x] AC-9: ruff + mypy clean on the `clip/` package; 10/10 new tests green; full suite
      `6 failed, 55 passed` — exactly the `TEST_BASELINE.md` 6, zero new failures.

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_clip_dataset.py -v

# Dry-run the gate on a single modality (no writes)
uv run python scripts/developer_scripts/clip_dataset.py --modality landsat9 \
    --aoi data/bow_valley_inference_aoi.geojson --dry-run

# Full clip of one small modality, then audit
uv run python scripts/developer_scripts/clip_dataset.py --modality worldcover \
    --aoi data/bow_valley_inference_aoi.geojson --out data/clipped_bow_valley_selection_raw
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
