# SPEC: Bow Valley Direct-Source Data Cube & Daily Snow Cover Inference

> **Status note (planning-skill ordering).** The planning skill sequences
> FDD → SPEC → Tasks. The design of record is `FDD_BOW_VALLEY_DATA.md`; this
> SPEC's acceptance criteria derive from that FDD's §3 (Verification & FMEA) and
> §4 (Implementation Steps). `PLAN_BOW_VALLEY_DATA.md` remains the **detailed
> reference** behind both — it carries the full grid/CRS tables, the verified
> per-source partial-coverage matrix, the temporal-window derivation, the
> per-modality archive formats, the cube-cache sizing, and the open-questions
> register that the FDD and this SPEC summarize but do not reproduce. This SPEC
> is sourced from the FDD, the PLAN, `DATA_ANALYSIS.md`, `CLIPPING_PLAN.md`, and
> `docs/agents/KNOWLEDGE.md`; nothing here should contradict the FDD or the PLAN.

This is the **central specification** for the direct-source pipeline. It is the
single source of acceptance criteria. Every AC is written as a falsifiable test
sentence and maps to at least one step in the Verification Plan (§7).

---

## 1. Overview

- **Goal:** Replace Google Earth Engine ingestion for the Bow Valley (Alberta)
  with a direct-source pipeline that (a) clips the raw multi-modal archive to
  `data/aoi.geojson`, (b) assembles per-cell multiband GeoTIFFs byte-compatible
  with `create_ee_image`, and (c) runs the pretrained `EncoderWithHead` as a
  per-day 1 km grid sweep, mosaicking 10×10 fractional-snow-cover (FSC)
  predictions into a daily COG over the AOI.
- **Problem Statement:** The repository's downstream code (`Dataset`,
  `LandsatEvalDataset`, `EncoderWithHead`, `LandsatEval`, metrics) consumes a
  GEE-exported logical raster stack with a fixed band order, tensor shapes,
  masks, normalization domain, and `-9999` nodata convention. We must reproduce
  that exact stack from local files **without modifying any downstream code**,
  for an archive that spans 2025-03 → 2025-06 and only partially covers the AOI
  on any given day.

---

## 2. Requirements

### Functional Requirements

**Stage 0→1 — AOI Clip (per `CLIPPING_PLAN.md`)**
- [ ] FR-1: A `clip_dataset.py` Typer CLI clips every raw dataset in
  `data/bow_valley_selection_raw` to `data/aoi.geojson` and writes
  `data/clipped_bow_valley_selection_raw`, preserving native pixel values, CRS,
  and file format (non-destructive).
- [ ] FR-2: A mandatory two-stage **intersect gate** runs before any clip on
  every product: (1) metadata-only footprint-vs-AOI **polygon** intersection;
  (2) minimum-useful-overlap (`MIN_AOI_OVERLAP_AREA_KM2`, default 1 km²) plus a
  post-clip valid-pixel check. Failing products produce **no output file**.
- [ ] FR-3: The clip stage emits a per-source clip manifest with one row per
  input product: `{product_id, footprint_bbox, intersects, aoi_overlap_km2,
  valid_pixel_count, action}` where `action ∈ {CLIP, SKIP_NO_OVERLAP,
  SKIP_DEGENERATE_OVERLAP}`.
- [ ] FR-4: MODIS/VIIRS clipping computes pixel indices **per native grid** (1 km
  = 1200², 500 m = 2400²) from each grid's own resolution/origin — never a single
  hardcoded `1200` clamp.
- [ ] FR-5: Landsat clips in native EPSG:32612; the clipped output preserves
  32612 (cross-zone reprojection to the 4326 cell grid happens later, in the
  Landsat adapter).

**Stage 1→2 — Adapters, Exporter, Grid (per `PLAN §4`)**
- [ ] FR-6: All `LocalSource*` adapters read **`data/clipped_bow_valley_selection_raw`**
  (the clipped archive), configured via `cube.yaml` `archive_root`. The raw path
  appears only in the clip stage's config.
- [ ] FR-7: Each adapter implements the `LocalSourceAdapter` contract (`fetch(cell,
  day) -> np.ndarray` of shape `(C, H, W)`, `-9999` nodata) and reprojects to the
  cell target grid (`EPSG:4326`, scale=10) using **bilinear** for continuous
  bands and **nearest** for QA/categorical. Bilinear resampling is
  **nodata-aware**: before warping, fill/nodata pixels (`-9999`, and the native
  `-28672` for MODIS) are masked to NaN so the interpolator never blends a valid
  value with a fill sentinel; the output restores `-9999` (or `-28672`) wherever
  the contributing source pixels were fill. This guards against edge-bleed (a
  valid reflectance interpolated against `-28672` would otherwise produce garbage
  negatives that bypass the `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`). This rule
  lives in the shared `base.py` resampler so every continuous adapter inherits it
  (MODIS, S2 swath edges, Landsat scene borders, etc.), not just MODIS.
- [ ] FR-8: A missing acquisition for `(source, day)` returns a `-9999` array of
  the declared shape (`create_placeholder` equivalent) — not an error.
- [ ] FR-9: For each source and each day, the adapter **mosaics all** AOI-
  intersecting granules/scenes/tiles **before** cropping to any cell
  (mosaic-before-crop).
- [ ] FR-9b: For scene/granule sources (S2, Landsat, swath sources), when more
  than one product shares the same (reference tile, date) — different
  orbit/satellite/reprocessing — the adapter **coalesces all of them per pixel**
  (first valid non-nodata/in-threshold value wins, fall through to the next where
  nodata; `-9999` only where all are nodata), in deterministic product order
  (latest processing time first). Coalesce is a valid-pixel union, **not** an
  average, and runs per-tile **before** FR-9's cross-tile mosaic. It replaces
  GEE's `.first()` selection.
- [ ] FR-10: The MODIS adapter preserves the native `-28672` fill value in
  addition to `-9999`.
- [ ] FR-11: The S2 adapter checks each granule's processing baseline and
  subtracts 1000 DN when baseline ≥ `04.00` (N0511) to match `S2_HARMONIZED`. The
  baseline is read from `<PROCESSING_BASELINE>` in the granule's `MTD_MSIL1C.xml`
  (verified present as `05.11`), falling back to the `N0511`-from-path token if
  the tag is absent. (Q8 already verified all 116 archive granules are N0511, so
  the subtraction applies to every granule here.)
- [ ] FR-12: The Landsat adapter encapsulates the L9→L8 fallback internally and
  emits renamed bands `B2_landsat … B7_landsat`.
- [ ] FR-13: The VIIRS coarse adapter emits a per-pixel raster on the cell grid
  (shape `(4, H, W)`), not a pre-averaged vector.
- [ ] FR-14: The ERA5 adapter emits raw Kelvin / native units, reading the
  already-daily archive files (one slice per day; no hourly re-aggregation). For
  `total_precipitation_sum` (a `stepType=accum` field) it applies the **ERA5-Land
  day-shift**: the daily total for day `i` is read from the **`i+1` `00:00` slice**
  (`tp[index] → day index−1`). Instantaneous temp/wind vars carry **no** shift. Missing
  day → `-9999`.
- [ ] FR-15: The DEM adapter computes slope and aspect with **latitude-correct
  metric pixel spacing**, matching GEE's `ee.Terrain.slope`/`aspect`
  (`src/data/earthengine/copernicus_dem.py:14-16`), then resamples elevation +
  slope + aspect to the 10 m cell grid (`EPSG:4326`, scale=10); emits
  `DEM, slope, aspect`. **Parity note (CRS is law):** GEE computes terrain on the
  DEM's native grid using true ground pixel dimensions (it does *not* take naive
  `dz/dx` in raw degree units), so the adapter must do the same — supply the
  correct metres-per-pixel in x/y at the cell's latitude to the slope/aspect
  kernel. Do **not** detour through EPSG:32611 to compute terrain: the GEE
  reference patches (AC-21) were never computed in a UTM frame, so a UTM-computed
  slope/aspect would fail parity. The risk being closed is running Horn's kernel
  on a degree grid with unit (`1°≈1 m`) pixel spacing, which scales gradients by
  ~111,000× — fixed by correct metric spacing, not by a projection change.
- [ ] FR-16: The WorldCover adapter ignores `day`, returns the v200 2021 `Map`
  band (single categorical band, not one-hot).
- [ ] FR-17: `LocalSourceExporter` writes a per-(cell, window-end-day) multiband
  GeoTIFF in the **exact** dynamic order `S1 + S2 + Landsat + S3 + MODIS + VIIRS
  fine + VIIRS coarse + ERA5 + MODIS cloud + S2 cloud + Landsat cloud`, then the
  static stack `DEM, slope, aspect, WorldCover Map`.
- [ ] FR-18: The exporter emits filenames matching
  `^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$` (`PR_{window_end_YYYYMMDD}_{LAT}_{LON}_SC00.tif`).
- [ ] FR-19: The grid generator (mode A) loads the **legacy CSV for cell geometry
  only**, reprojects each cell to EPSG:4326, keeps a cell iff its centre lies
  within `data/aoi.geojson` (→ 344 cells), supports `--require-fully-inside`
  (→ 338 cells), and emits a kept/dropped manifest. Mode B tiles
  `data/aoi.geojson` (legacy CSV not consumed). Both modes are bounded by the
  AOI, never the wider cell-sampling bbox.
- [ ] FR-19b: The grid generator **emits a generated cube CSV** with the
  canonical schema `date, crs, center_x, center_y, min_x, min_y, max_x, max_y`
  (EPSG:32611), consumable unchanged by
  `EarthEngineExporterEval.export_from_csv_utm`. Content is the **full
  cross-product** of in-AOI cells × every day in the inference window (one row
  per `(cell, day)`); this CSV is the inference sweep enumeration. The legacy
  CSV's `date` column is **not** read.
- [ ] FR-20: The cube cache stores per-modality per-(cell, day) `.npz` arrays
  (not per-window) in `data/bow_valley_processing/cube_cache/`, **sharded one
  subdirectory per cell** (`cube_cache/{cell_id}/{day}_{modality}.npz`),
  FIFO-evicted with a configurable size cap. A flat layout would put ~300k files
  (mode A: ~344 cells × ~96 archive days × ~9 modalities) in a single directory,
  degrading ext4/xfs directory indexing to O(N); the per-cell shard keeps each
  directory under ~1k entries. (Per-cell sharding suffices — no hash-prefix tier
  needed.)
- [ ] FR-20b: All Stage 2 (cube/inference) artifacts are written under
  `data/bow_valley_processing/`, each process to its **own subdirectory**:
  `cube_cache/` (.npz, intermediate/cleanable), `cubes/` (assembled 8-day cube
  tifs — final destination), `daily_fsc/` (daily FSC COGs — deliverable),
  `manifests/`, `scratch/` (transient/cleanable). No Stage 2 process writes into
  `data/clipped_bow_valley_selection_raw` or `data/bow_valley_selection_raw`. The
  clip stage writes **only** `data/clipped_bow_valley_selection_raw`.
  `processing_root` is a `cube.yaml` setting; subdirs derive from it.

**Stage 2 — Inference & Mosaic (per `PLAN §5`)**
- [ ] FR-21: `InferenceGridDriver` builds, for each inference day `d ∈ [start,
  end]`, the 8-day window `[d-7, d]` per cell, exports the cube, batches cells,
  runs `EncoderWithHead`, and produces per-cell 10×10 FSC.
- [ ] FR-22: `DailyMosaicWriter` writes one COG per day in EPSG:32611 to
  `data/bow_valley_processing/daily_fsc/` (overridable per Q5),
  reprojecting each 10×10 FSC patch from EPSG:4326 with **nearest-neighbour**,
  stitching only valid predictions and recording per-day AOI-coverage fraction.

### Non-Functional Requirements

- **Correctness / Interchangeability:** Output is byte-layout- and
  value-domain-compatible with `create_ee_image`. Downstream code is **not**
  modified (single permitted exception: an additive `PR`-prefix allowlist patch
  at `landsat_eval.py:172` only if Q9 reopens — currently RESOLVED, no patch
  needed).
- **Performance:** Per-cell export is parallel (`multiprocessing.Pool`); S1 SAFE
  archives use windowed reads of the cell footprint, never full-scene loads.
  Mode A (~344 cells × ~53 days) is hours on one GPU; mode B needs multi-GPU
  budget (Q7).
- **Storage:** all under `data/bow_valley_processing/` (intermediate
  `cube_cache/`+`scratch/` cleanable mid-run; `cubes/`+`daily_fsc/` kept). Mode A
  cache ≈ 72 GB + ≈ 75 GB assembled cubes in `cubes/`; cache size cap
  configurable (default 200 GB) in `cube.yaml`. Clipped archive
  (`data/clipped_bow_valley_selection_raw`) is written only by the clip stage.
- **Observability:** `structlog` JSON logging throughout; the clip manifest and
  the per-day coverage metric are the audit artifacts.
- **Security:** No hardcoded secrets; archive paths via config, not literals.

---

## 3. Technical Constraints & Assumptions

**Existing systems/libraries to use**
- Downstream (unchanged): `Dataset`, `LandsatEvalDataset`,
  `prediction_month_from_file` (`src/fsc/landsat_eval.py`), `EncoderWithHead`
  (`src/fsc/patch_predict.py`), `Normalizer`, band-group dicts in
  `src/data/earthengine/eo.py`, constants in `src/data/config.py`.
- Tooling: `uv`, `ruff`, `mypy`, `pytest`; `structlog`, `pydantic-settings`,
  `pathlib`, `polars`, `typer`; `rasterio`, `pyproj`, `xarray`/`h5netcdf`,
  `h5py`, system `gdalinfo`/`gdal_translate` (rasterio's GDAL build lacks the
  HDF4 driver — verified).

**Directory roots (per `PLAN §3` Directory layout)**
- `data/bow_valley_selection_raw/` — raw archive, read-only.
- `data/clipped_bow_valley_selection_raw/` — clipped archive; **written only by
  the clip stage**, read-only to everything else.
- `data/bow_valley_processing/` — all Stage 2 writes, one subdir per process
  (`cube_cache/`, `cubes/`, `daily_fsc/`, `manifests/`, `scratch/`); intermediate
  subdirs cleanable mid-run, `cubes/` + `daily_fsc/` are the kept deliverables.

**Key fixed constants (from `config.py` / `DATA_ANALYSIS.md`)**
- `NUM_TIMESTEPS=8`, `DAYS_PER_TIMESTEP=1`, `EXPORTED_HEIGHT_WIDTH_METRES=1000`,
  `DATASET_OUTPUT_HW_HIGH_RES=100`, `NO_DATA_VALUE=-9999`,
  `MODIS_FILL_VALUE=-28672`, med-res → 5×5, low-res → 2×2.

**Assumptions**
- Default inference window **2025-04-06 → 2025-05-28** (S1-start-limited start;
  S2/L8-end-limited end); archive ingest period `start-7 → end`. The 7-day
  prefill is non-negotiable.
- Sweep mode **A** (sample-only, ~344 in-AOI cells) is the default; mode B is a
  config switch (Q3 — confirm for production).
- `data/aoi.geojson` is the single authoritative clip/inference boundary; 31% of
  the original 500 cells fall outside it and are intentionally dropped.
- `sampled_cells_bow_river_with_dates.csv` is consumed for **cell geometry only**
  (`center_x/y`, bounds). Its `date` column is train/eval label-sampling
  metadata and is **not read** by this inference pipeline.
- DEM (9 elevation tiles) and WorldCover (4 `Map` tiles) mosaics must cover the
  AOI to `lat_max = 52.31` — Phase 0 asserts this (verified: DEM mosaic
  `lat[50,53]`, WorldCover `lat[48,54]`, both ⊇ AOI); per-tile bounds in
  `DATA_ANALYSIS.md` are examples, not the mosaic extent.
- The known ERA5 temperature-shift sign and S3 identity-normalization TODO are
  **preserved as-is** (model-numeric-domain concerns, out of scope).

**Open questions that gate specific ACs (do not invent answers):**
- **Q3 (sweep mode A vs B):** affects compute sizing only; default A holds.
- **Q4 (CSV `date` column semantics): RESOLVED.** The CSV `date` is train/eval
  label-sampling metadata, not a per-cell prediction day. This is an inference
  run: the driver iterates the configured §3 window (all in-AOI cells × every
  day) and ignores the CSV `date`; the CSV supplies cell geometry only. No AC is
  blocked by Q4 (see AC-31).
- **Q5 (output destination), Q6 (checkpoint path), Q7 (GPU budget):**
  configuration, not behaviour; do not block ACs.

---

## 4. Acceptance Criteria

Each AC is a test sentence (Red test waiting to be written). Pass/fail is
explicit.

**Clip stage (§2.0–§2.7 of `CLIPPING_PLAN.md`)**
- [ ] AC-1: Given a synthetic product footprint **fully outside** the AOI, the
  clip gate returns `SKIP_NO_OVERLAP` and **no output file** exists at the
  destination path.
- [ ] AC-2: Given a footprint whose AOI intersection is **below**
  `MIN_AOI_OVERLAP_AREA_KM2`, the gate returns `SKIP_DEGENERATE_OVERLAP` and no
  output file exists.
- [ ] AC-3: Given a **partially-overlapping** real Landsat scene (~8% AOI
  overlap), the gate returns `CLIP`, the output is clipped to the intersection,
  and the output contains **> 0** valid (non-nodata) pixels.
- [ ] AC-4: After any bulk clip, **zero** outputs in
  `data/clipped_bow_valley_selection_raw` are all-nodata / zero-valid-pixel
  (post-run audit asserts this).
- [ ] AC-5: The clip manifest contains exactly one row per input product, each
  with the correct `action` and measured `aoi_overlap_km2` / `valid_pixel_count`.
- [ ] AC-6: For a single MOD09GA file, the clipped 500 m-grid output extent and
  the 1 km-grid output extent both cover the same AOI corner, and the 500 m pixel
  index is ~2× the 1 km index for that corner (no half-band truncation).
- [ ] AC-7: A clipped Landsat band asserts `crs == EPSG:32612` (native zone
  preserved); a clipped S2 band asserts `crs == EPSG:32611`.
- [ ] AC-8: The clipped output is non-destructive: for a CLIP product, sampled
  pixel values inside the AOI equal the corresponding raw pixel values (no
  resampling/rescaling introduced by the clip).

**Filename & grid contract (tested before any adapter — `PLAN §6`)**
- [ ] AC-9: Every exporter-emitted filename matches
  `^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$` **and**
  `prediction_month_from_file` returns the month equal to `window_end.month`.
- [ ] AC-10: `grid.py` mode A produces **344** cells for the centre-in rule and
  **338** for `--require-fully-inside`, emits a kept/dropped manifest summing to
  500, and every kept cell centre lies within `data/aoi.geojson`.
- [ ] AC-11: Grid cells are non-overlapping (pairwise intersection area == 0).
- [ ] AC-11b: The generated cube CSV has the canonical 8-column schema, every row
  is one `(in-AOI cell, window-day)` pair covering the full cross-product
  (row count == kept-cell count × window-day count), every `crs == EPSG:32611`,
  and `pandas.read_csv` → `export_from_csv_utm` parses it without error (schema
  contract with `eo_eval.py:577-585`).

**Adapter value-domain & CRS (one test per adapter — `PLAN §6`, `DATA_ANALYSIS §source-by-source`)**
- [ ] AC-12: Every adapter's `fetch` output has the declared shape, `EPSG:4326`
  target transform/CRS, and exactly the declared `bands_out` in order; a golden-
  grid test asserts the exact `(transform, shape, crs)` triple.
- [ ] AC-13: A missing `(source, day)` yields an all-`-9999` array of declared
  shape for every time-varying adapter.
- [ ] AC-14: **S1** output bands are `[VV, VH, angle]`, pixels `< -30.0` are
  masked, value domain roughly `[-50, 1]` for VV/VH; parity vs GEE reference
  patch within threshold.
- [ ] AC-15: **S2** output bands `[B2,B3,B4,B8,B11,B12]`; for an N0511 granule the
  adapter subtracts 1000 DN; reflectance domain matches `S2_HARMONIZED` (÷10000
  downstream); parity within threshold.
- [ ] AC-15b: **Same-tile/date coalesce.** Given two products for the same tile
  and date with **complementary** nodata masks (each valid where the other is
  nodata), the coalesced adapter output has **zero** `-9999` at any pixel valid in
  at least one product, and at pixels valid in both the value equals the
  deterministic-order winner (latest processing time). With a real S2 case
  (`20250420 T11UNT`: R113 vs R070), coalesced valid-pixel count ≥ the
  max of either product alone. (Guards against `.first()` false nodata.)
- [ ] AC-16: **Landsat** exercises three cases — L9 present, L9 missing+L8
  present, both missing → all-`-9999`; output bands are renamed `B*_landsat`;
  cross-zone 32612→4326 reprojection asserted against the cell grid.
- [ ] AC-17: **S3** emits `[Oa17_radiance, Oa21_radiance]`, georeferenced via
  tie-point grids; identity normalization preserved.
- [ ] AC-18: **MODIS** emits `sur_refl_b01..b07`; the native `-28672` fill is
  **present** in the output where the source had it (NDSI/NDVI assertions at
  `landsat_eval.py:317,331` depend on it).
- [ ] AC-19: **VIIRS** fine emits `[I1, I3]` (shape `(2,H,W)`); coarse emits
  `[M5,M7,M10,M11]` as a per-pixel raster (shape `(4,H,W)`), and the loader's
  spatial mean over that raster reproduces the GEE `time_x` values.
- [ ] AC-20: **ERA5** emits the five bands in Kelvin/native units from the daily archive
  files; missing day → `-9999`.
- [ ] AC-20b: **ERA5 precip day-shift.** Given a synthetic `tp` array where the `00:00`
  slice stamped day `d+1` holds a known nonzero total and day `d`'s slice holds a
  different value, the adapter's precip output for inference day `d` equals the `d+1`
  slice value (the accumulation closing day `d`), **not** the day-`d` slice. The same
  test asserts an instantaneous variable (`temperature_2m`) for day `d` is read from the
  day-`d` slice (no shift). (Guards the silent off-by-one.)
- [ ] AC-21: **DEM** computes slope/aspect with latitude-correct metric pixel
  spacing matching `ee.Terrain` (asserted by comparing against GEE-derived
  slope/aspect within threshold — NOT against a EPSG:32611-computed reference),
  resamples to the 10 m cell grid, emits `[DEM, slope, aspect]`. A degenerate
  guard asserts slopes are **not** all ≈90° (the failure mode of running the
  kernel on a degree grid with unit pixel spacing).
- [ ] AC-22: **WorldCover** emits a single `Map` band with class codes in
  `{10,20,…,95,100}` (not one-hot), independent of `day`.

**Full-stack parity & tracer (the PLAN's nine primary ACs — `PLAN §6`)**
- [ ] AC-23: For one cell × one window-end-day exported from the **clipped**
  archive and read through `LandsatEvalDataset`, the assembled tensors have:
  `space_time_high_res_x == (100,100,8,15)`, `space_time_med_res_x == (5,5,8,2)`,
  `space_time_low_res_x == (2,2,8,11)`, `time_x == (8,9)`, `space_x ==
  (100,100,14)`, `static_x == (3,)`.
- [ ] AC-24: For that sample, `EncoderWithHead` produces an FSC prediction of
  shape `(10,10)` with all values in `[0.0, 1.0]`.
- [ ] AC-25: For that sample, `valid_data_mask_*` is set wherever inputs are
  `-9999` or below their `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`.
- [ ] AC-26: An integration test asserts byte-for-byte **band-name equality**
  between the exporter's GeoTIFF band list and `create_ee_image`'s output band
  list (band-order regression guard via `layout.py` re-export from `eo.py`).
- [ ] AC-27: Full-stack numeric parity: per-source diff between the direct-source
  cube and the Phase 0 GEE reference patches is within each source's documented
  tolerance. **Coordinate reconciliation:** parity pairs the direct-source cube to
  its GEE reference by **shared `cube_cells.csv` row** (FR-19b) — both pipelines
  are driven by the same generated CSV — not by raw filename-string equality, so
  the UTM-CSV (`export_from_csv_utm`) vs degree-filename (`PR_…_{LAT}_{LON}_…`)
  representation difference does not affect matching.

**Inference & mosaic**
- [ ] AC-28: `DailyMosaicWriter` output is a valid COG in EPSG:32611; cells with
  all input groups masked are `nodata`; the per-day AOI-coverage fraction is
  recorded in output metadata.
- [ ] AC-29: A 2×2 adjacent-cell mosaic test shows non-overlapping seams (no
  double-written pixels) and FSC reprojected with nearest-neighbour only.

**Coverage profiling (first-class per `PLAN §3`)**
- [ ] AC-30: Phase 0 produces, per in-AOI cell, the fraction of each 8-day window
  populated per source; the S1-fully-masked-window rate across the inference
  range is reported (S1 present on only ~16 archive dates).
- [ ] AC-31: The driver iterates **all in-AOI cells × every day in the configured
  window** (`2025-04-06 → 2025-05-28` by default); it does **not** read the CSV
  `date` column. A test asserts that two cells with different CSV `date` values
  are both predicted on the same configured inference day (i.e. the CSV `date`
  has no effect on the inference loop). The CSV is consumed for cell geometry
  (`center_x/y`, bounds) only.

**Directory contract**
- [ ] AC-32: After a cube+inference run, all new files live under
  `data/bow_valley_processing/` in the correct subdirs (assembled cubes in
  `cubes/`, daily COGs in `daily_fsc/`, `.npz` in `cube_cache/`); a test asserts
  **no** file was created or modified under `data/clipped_bow_valley_selection_raw`
  or `data/bow_valley_selection_raw` by the cube/inference stage, and that
  deleting `cube_cache/` + `scratch/` does not remove any file in `cubes/` or
  `daily_fsc/` (intermediate/deliverable separation).

---

## 5. Dependencies

- **Inputs:** `data/bow_valley_selection_raw` (raw, read-only),
  `data/aoi.geojson`, `sampled_cells_bow_river_with_dates.csv` (cell geometry
  only; `date` column unused — see §3 Q4).
- **Internal modules (unchanged):** `LandsatEvalDataset`, `EncoderWithHead`,
  `Normalizer`, `eo.py` band-group dicts, `config.py` constants.
- **Phase 0 artifacts (prerequisite):** `ARCHIVE_AUDIT.md`; the **generated cube
  CSV** `configs/bow_valley/cube_cells.csv` (FR-19b); GEE reference patches in
  `tests/fixtures/gee_reference_patches/` (produced by
  `scripts/export_for_inference.py` → `EarthEngineExporterEval.export_from_csv_utm`,
  **not** `export_for_eval.py`, over a 5–10 row sample **of the generated cube
  CSV** — guaranteeing in-AOI/in-archive parity cells) — every parity AC
  (AC-14,15,21,27) consumes these.
- **External data semantics:** Copernicus (S1/S2/S3/DEM), USGS (Landsat),
  NASA LP DAAC (MODIS/VIIRS), ECMWF CDS (ERA5), ESA (WorldCover).
- **Outputs:** clipped archive → `data/clipped_bow_valley_selection_raw` (clip
  stage only); all Stage 2 artifacts → `data/bow_valley_processing/` (assembled
  cubes in `cubes/`, daily FSC COGs in `daily_fsc/`; see §3 Directory roots).
- **Stage ordering:** Clip stage (FR-1…FR-5) is a hard prerequisite for all
  adapter/exporter/inference ACs — the cube is built from the clipped archive.

---

## 6. Out of Scope

- Retraining or fine-tuning the model.
- Changing band order, normalization constants, mask semantics, or tensor shapes.
- Fixing the ERA5 temperature-shift sign or the S3 identity-normalization TODO.
- Adding new modalities (cloud-flag bit decoding, VIIRS QF1).
- Real-time / streaming ingestion (batch over a static archive only).
- Widening the AOI to recover the 31% out-of-AOI cells.
- Padding clipped outputs to the full AOI (partial coverage is kept as-is).

---

## 7. Verification Plan

Ordered to match `PLAN §7` phasing; each AC maps to ≥1 step.

1. **Phase 0 — Archive audit + cube CSV + GEE reference patches.** Stand up the
   geometry half of `grid.py` and emit `configs/bow_valley/cube_cells.csv`
   (full cross-product); sample 5–10 rows and run
   `scripts/export_for_inference.py` (→ `export_from_csv_utm`) to build
   `tests/fixtures/gee_reference_patches/`; profile per-cell per-source window
   coverage; assert DEM/WorldCover mosaics reach lat 52.31. → **AC-11b, AC-30**;
   prerequisite for AC-14/15/21/27.
2. **Phase 0.5 — Clip stage.** Unit tests on synthetic footprints and real
   samples; post-run audit. → **AC-1…AC-8** (`test_clip_dataset.py`:
   outside→skip, degenerate→skip, partial→keep>0px, manifest correctness,
   per-grid MODIS extents, native-CRS preservation, non-destructive pixel
   equality, zero-all-nodata audit).
3. **Phase 3 Step 1 — Contract (grid + filename + layout).** → **AC-9, AC-10,
   AC-11** (`test_filename_contract.py`, `test_grid.py`) — run **before** any
   adapter.
4. **Phase 3 Step 2 — Placeholder exporter + tracer.** All-`-9999` cube through
   the loader + head. → **AC-23, AC-24, AC-25** plumbed (degenerate FSC),
   **AC-13** (placeholder path), **AC-26** (band-name equality).
5. **Phase 3 Step 0 — S1/S2 parity spikes (run early, de-risk).** →
   **AC-14, AC-15** drift quantified against reference patches; decision point.
6. **Phase 3 Step 3 — Adapters in difficulty/parity order** (worldcover, dem,
   era5, modis, viirs, s3, landsat, s2, s1), one `test_<modality>_adapter.py`
   each. → **AC-12, AC-15b, AC-16…AC-22** (+ AC-14/15 promoted to production).
   AC-15b (same-tile/date coalesce) is exercised in the S2 and Landsat adapter
   tests.
7. **Phase 3 Step 4 — Driver + mosaic.** → **AC-21(driver), AC-28, AC-29, AC-32**
   (`test_inference_driver.py`, 2×2 mosaic test, directory-contract test).
8. **Full-stack parity gate.** `test_exporter_parity.py` over the reference
   patches. → **AC-27**.
9. **Driver-loop semantics test** (part of Phase 3 Step 4): assert the driver
   iterates the configured window × all in-AOI cells and ignores the CSV `date`.
   → **AC-31**.

Every Phase 3 step must pass `ruff check`, `mypy`, and its test set before its
approval gate (per CLAUDE.md). Parity ACs (AC-14/15/21/27) are **blocked** until
Phase 0 reference patches exist.

---

## Pre-Approval Checklist

- [x] SPEC stored in a discoverable location
  (`docs/agents/planning/raw-data-ingestion/SPEC_BOW_VALLEY_DATA.md`).
- [x] All functional/non-functional requirements are measurable and falsifiable.
- [x] Every AC has a clear pass/fail state.
- [x] Every AC maps to at least one step in the Verification Plan.
- [ ] **User has explicitly approved this SPEC before implementation begins.**

### Outstanding blockers to flag before sign-off
- **Q3 (sweep mode A vs B)** drives compute budget (Q7); default A assumed.
- **Q6 (checkpoint path)** needed before any inference AC can actually run.
- **FDD exists** (`FDD_BOW_VALLEY_DATA.md`) and is the design of record; its own
  approval gate (FDD §5) must be cleared before this SPEC is signed off, per the
  planning skill's FDD → SPEC → Tasks ordering.
