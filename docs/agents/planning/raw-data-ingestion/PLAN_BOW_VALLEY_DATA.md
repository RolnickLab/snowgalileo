# Plan — Bow Valley Direct-Source Data Cube & Daily Snow Cover Inference

## 1. Goal

Adding a direct-source pipeline for the Bow Valley (Alberta, west of Calgary) to 
supplement Google Earth Engine ingestion to :

1. Build a spatiotemporal data cube matching the exact tensor contract defined
   in `DATA_ANALYSIS.md` (band order, shapes, masks, normalization, `-9999`
   nodata).
2. Run the pretrained SnowGalileo encoder + FSC head over the cube as a
   **1 km × 1 km grid sweep, per day**, producing a daily fractional snow cover
   raster mosaic for the entire AOI.

The repository's downstream code (`Dataset`, `LandsatEvalDataset`,
`EncoderWithHead`, `LandsatEval`, metrics) must remain untouched. We add two
new modules behind the existing contracts: a direct-source ingestion adapter
and an inference-grid driver.

---

## 2. Scope & Non-Goals

**In scope**
- A **non-destructive AOI clip stage** (`CLIPPING_PLAN.md`) that crops every raw
  dataset in `data/bow_valley_selection_raw` to `data/bow_valley_inference_aoi.geojson` and writes
  `data/clipped_bow_valley_selection_raw`. This is a **mandatory upstream stage**,
  not an optional utility — the adapters below read the **clipped** archive (see
  §3 "Pipeline Stages & Data Flow").
- Local-file adapters for S1, S2, Landsat 8/9, S3 OLCI, MODIS MOD09GA,
  VIIRS VNP09GA, ERA5-Land, Copernicus DEM GLO-30, ESA WorldCover v200.
- Producer of per-cell GeoTIFFs matching `create_ee_image` layout (dynamic
  stack + static stack, `-9999` nodata, `EPSG:4326`, scale=10 (approx. `0.0000898315` deg), dims ≈ 159×100 due to latitude convergence at ~51°N,
  `MODIS_FILL_VALUE=-28672` preserved in MODIS bands).
- A 1 km grid generator covering the Bow Valley AOI in EPSG:32611.
- A daily-stride inference driver that, for each day `d` in the configured
  window, builds an 8-day input window ending at `d`, runs `EncoderWithHead`
  over every grid cell, and mosaics the per-cell 10×10 FSC predictions into a
  daily COG.
- Validation harness comparing direct-source patches against GEE reference
  patches **generated as part of Phase 0** by running the existing GEE pipeline
  over a small held-out sample (5–10 cells × 3 days).

**Out of scope**
- Retraining or fine-tuning the model.
- Changing band order, normalization constants, mask semantics, or tensor
  shapes.
- Adding new modalities (cloud-flag decoding, VIIRS QF1, etc.).
- Real-time / streaming ingestion. Batch job over a static archive.
- Fixing the known ERA5 temperature shift sign or the S3 identity normalization
  TODO — these are model-numeric-domain concerns, addressed in a separate
  migration.

---

## 3. AOI, Grid, and Temporal Window

### Pipeline Stages & Data Flow (where clipping happens — RESOLVED)

The full pipeline is **three sequential stages**. The clip stage was previously
only implicit; it is formalized here as a mandatory stage with its own approval
gate (§7 Phase 0.5).

```
  Stage 0 — RAW ARCHIVE (read-only, never mutated)
    data/bow_valley_selection_raw/   (10 modalities, native CRS/format)
        │
        │  AOI clip  (CLIPPING_PLAN.md):
        │  • §2.0 intersect gate runs HERE, once per product
        │  • non-destructive: native pixels/CRS/format preserved
        │  • out-of-AOI products skipped (no output file)
        ▼
  Stage 1 — CLIPPED ARCHIVE   ◄── the adapters' archive root
    data/clipped_bow_valley_selection_raw/   (same layout, cropped to bow_valley_inference_aoi.geojson)
        │
        │  LocalSource* adapters (§4) + LocalSourceExporter
        │  • per-(cell, day) reprojection to EPSG:4326 scale=10
        │  • mosaic-before-crop, -9999 placeholders
        ▼
  Stage 2 — CUBE + INFERENCE   (all writes under data/bow_valley_processing/)
    .npz cache (cube_cache/) → assembled 8-day cubes (cubes/)
      → EncoderWithHead → DailyMosaicWriter (daily COG → daily_fsc/)
```

**Data-flow contract (RESOLVED).** The `LocalSource*` adapters in §4 read
**`data/clipped_bow_valley_selection_raw`** (Stage 1 output), **not** the raw
archive. Consequences, binding on every downstream component:

1. **The clip stage is on the inference path and is a hard prerequisite.** It
   must run (and pass its post-run audit) before any adapter, cube export, or
   inference job. It is **not** a storage-shrink convenience.
2. **`data/bow_valley_inference_aoi.geojson` is the single binding extent end-to-end.** Because the
   clipped archive contains no data outside the AOI, the §2.0 intersect gate is
   the *one* place footprint-vs-AOI filtering happens. Adapters do **not**
   re-implement it — they assume every product they see already intersects the
   AOI. This keeps the gate from being duplicated across 9 adapters (DRY;
   contract-first).
3. **Adapters still handle partial coverage.** Clipping crops to the AOI but does
   not fabricate coverage: a clipped product may still cover only part of the AOI
   (the 8–42% Landsat case), and many (source, day) pairs have no product at all.
   Adapters emit `-9999` placeholders for those — see "Per-Timestep, Per-Source
   Partial AOI Coverage" below. The clip stage and the placeholder path are
   complementary, not redundant.
4. **Config wiring.** `configs/bow_valley/cube.yaml` `archive_root` points at
   `data/clipped_bow_valley_selection_raw`. The raw path appears **only** in the
   clip stage's config, nowhere in the adapter/exporter config.

### Directory layout (inputs, processing, outputs — RESOLVED)

Three disk roots, with **strictly separated** read/write semantics. CRS is law;
so is provenance — nothing downstream of the clip stage ever writes back into the
archives.

| Root | R/W | Lifetime | Contents |
| --- | --- | --- | --- |
| `data/bow_valley_selection_raw/` | **read-only** | permanent (source) | Raw archive (Stage 0 input). Never mutated. |
| `data/clipped_bow_valley_selection_raw/` | clip stage writes; everything else **read-only** | permanent (until re-clip) | Clipped archive (Stage 1). Same layout as raw. The **only** thing the clip stage writes; **no other process writes here.** |
| `data/bow_valley_processing/` | all Stage 2 processes write | **ephemeral — safe to delete and regenerate**, except `cubes/` and `daily_fsc/` which are the deliverables | All intermediate and final cube/inference artifacts. **Each process gets its own subdirectory** (below). |

**`data/bow_valley_processing/` subdirectory contract — every process writes only
to its own subdir:**

```
data/bow_valley_processing/
  cube_cache/     # cube_cache.py — per-(modality, cell, day) .npz. Intermediate; FIFO-evicted, cleanable.
  cubes/          # LocalSourceExporter — assembled 8-day multiband cube tifs (per (cell, window-end-day)). FINAL DESTINATION of the cubes.
  daily_fsc/      # DailyMosaicWriter — daily FSC COGs over the AOI (the inference deliverable). (See §8 Q5 for object-storage option.)
  manifests/      # clip manifest copy + per-cell/per-source 8-day-window coverage profiles + kept/dropped cell manifest.
  scratch/        # transient per-worker temp (unpacked SAFE members, intermediate reprojections). Always cleanable mid-run.
```

- **Intermediate vs deliverable:** `cube_cache/` and `scratch/` are throwaway —
  a run may delete them to reclaim space and regenerate on demand. `cubes/`
  (assembled 8-day cubes) and `daily_fsc/` (daily FSC COGs) are the **kept
  outputs**.
- **No cross-writing:** the clip stage writes **only**
  `data/clipped_bow_valley_selection_raw`; the cube/inference stage writes
  **only** under `data/bow_valley_processing/`. Neither writes into the other's
  root, and neither writes into `data/bow_valley_selection_raw`.
- **Config wiring (cont.):** `cube.yaml` carries `archive_root`
  (`…/clipped_bow_valley_selection_raw`), `processing_root`
  (`…/bow_valley_processing`), and the cache size cap; the exporter derives
  `cube_cache/`, `cubes/`, `scratch/`, `manifests/` from `processing_root`.
  `inference.yaml` points the daily-COG output at
  `…/bow_valley_processing/daily_fsc/` (overridable per Q5).

### AOI — two distinct definitions (DO NOT CONFLATE)

There are **two** AOI definitions in this project and they are **not** the same
extent. CRS is law; both are stated explicitly below.

1. **Cell-sampling extent** — convex-bbox of the 500 cells in
   `sampled_cells_bow_river_with_dates.csv` (EPSG:32611):

   | Bound | Value (EPSG:32611, m) |
   | --- | --- |
   | `min_x` | `518363.85` |
   | `max_x` | `705363.85` |
   | `min_y` | `5599583.79` |
   | `max_y` | `5761583.79` |
   | width  | `187_000` (187 km) |
   | height | `162_000` (162 km) |
   | max-tile grid | `187 × 162 = 30_294` 1 km cells if fully tiled |

2. **Clip / inference AOI** — `data/bow_valley_inference_aoi.geojson` (EPSG:4326), the boundary that
   `CLIPPING_PLAN.md` clips every raw dataset to:

   | Bound | Value (EPSG:4326, deg) |
   | --- | --- |
   | `lon_min` | `-116.561936219710887` |
   | `lon_max` | `-114.527659450240762` |
   | `lat_min` | `50.729806886838752` |
   | `lat_max` | `52.306672311654424` |

**The clip AOI does NOT contain all 500 cells.** Reprojecting cell extents to
EPSG:4326 shows the cell envelope spans `lon=[-116.7408, -114.0104]`,
`lat=[50.5121, 52.0046]` — wider east/west and further south than the clip AOI.
**156 of 500 cells (31%) have their centre outside `data/bow_valley_inference_aoi.geojson`**; only
**338 cells fall fully inside** and **344 are inside under a centre-in rule**.
Max spillover: ~31.5 km east, ~20.5 km south, ~12.4 km west.

**Decision (resolved):** `data/bow_valley_inference_aoi.geojson` is the **authoritative clip and
inference boundary by design**. Cells whose centre falls outside it are
**intentionally dropped** — they are not served by the clipped archive and the
grid generator MUST filter them out. The cell-sampling extent above is retained
only as provenance for how the cells were originally drawn; it is **not** the
sweep extent.

**Grid-generator contract (mode A):** load the legacy CSV **for cell geometry
only**, reproject each cell to EPSG:4326, keep a cell iff its centre lies within
`data/bow_valley_inference_aoi.geojson` (centre-in rule → **344 cells**), drop the rest. A
`--require-fully-inside` flag restricts to the **338** fully-contained cells.
Emit a manifest of kept/dropped cell ids for auditability.

**Generated cube CSV (the pipeline's actual cell/date input — RESOLVED).** The
grid generator does **not** consume the legacy
`sampled_cells_bow_river_with_dates.csv` `date` column (that column is train/eval
label-sampling metadata — see §8 Q4). Instead it **emits its own CSV** with the
canonical schema `date, crs, center_x, center_y, min_x, min_y, max_x, max_y`
(EPSG:32611) so that `EarthEngineExporterEval.export_from_csv_utm`
(`src/data/earthengine/eo_eval.py:576`) consumes it **unchanged**. Content is the
**full cross-product**: every in-AOI cell × every day in the inference window
(default `2025-04-06 → 2025-05-28`), one row per `(cell, day)`. This generated
CSV **is** the inference sweep enumeration:
- **Mode A:** cells are the 344 in-AOI legacy-CSV cells (geometry only) × window
  days ≈ 344 × 53 ≈ **18 k rows**.
- **Mode B:** cells are tiled directly from `data/bow_valley_inference_aoi.geojson` (legacy CSV not
  needed at all) × window days.
- Per-row `date` is the window-end; the GEE/export side derives
  window-start = `date − (NUM_TIMESTEPS−1)`. The direct-source driver reads the
  same rows. This does **not** reintroduce Q4: the dates here are simply the
  configured inference window enumerated, not per-cell label days.
- The **Phase 0 GEE reference-patch** run (§7) samples a **small subset (5–10
  rows)** of this same generated CSV — guaranteeing the parity cells are in-AOI
  and in-archive by construction.

Coordinates are load-bearing: `center_x/y` build the per-cell filename (→
`static_x` location) and `min/max_*` build the export polygon
(`eo_eval.py:599, 606`).

**Decision required before FDD (see §8 Q3):** sweep mode.
- **(A) Sample-only:** infer over the in-AOI CSV cells only (~344 cells after
  the AOI filter above). Cheap.
- **(B) Full tile:** tile the **clip AOI** (`data/bow_valley_inference_aoi.geojson`), not the wider
  cell-sampling bbox. Storage and compute are ~60× larger; needs explicit GPU
  budget.

Plan assumes **(A)** as default. (B) is a configuration switch on the grid
generator. **Both modes are bounded by `data/bow_valley_inference_aoi.geojson`, never by the wider
cell-sampling bbox** — the clipped archive contains no data outside the AOI.

### Grid + CRS

| Parameter | Value | Source / Rationale |
| --- | --- | --- |
| Grid math CRS | `EPSG:32611` (UTM 11N) | Matches CSV cell extents; preserves 1 km cell metric. |
| Per-cell export CRS | `EPSG:4326`, `scale=10` | Matches `create_ee_image`; downstream loader assumes this. Scale 10 equates to `0.0000898315` degrees. |
| Daily mosaic CRS | `EPSG:32611` | Mosaic stays in metric CRS for analysis. **Per-cell rasters in 4326, mosaic in UTM is intentional** — per-cell tifs feed the loader unchanged; mosaic is a separate output product. Reprojection happens once, at mosaic-write time, on 10×10 FSC outputs (low IO). |
| Grid cell size | 1000 m × 1000 m (dims ≈ 159×100 px in EPSG:4326 due to latitude convergence at 51°N) | `EXPORTED_HEIGHT_WIDTH_METRES`. Converging longitudes stretch WGS84 cell width to ~159 px, satisfying `H >= 100` and `W >= 100` for dataset cropping. |
| Cell layout | Non-overlapping; centred on CSV `center_x, center_y` (mode A, after AOI filter) or tiled across `data/bow_valley_inference_aoi.geojson` (mode B) | Matches existing CSV semantics; both modes bounded by the clip AOI |

**CRS is law** — every cell carries an explicit `transform`, `crs`, and `shape`
triple that all adapters must conform to.

### Fixed Extent Mosaic & Scene Coverage Complexity

A fixed spatial extent is provided for the full Bow Valley AOI desired daily mosaic. This introduces significant operational complexity:
- **Multi-Scene Composition**: The clip AOI (`data/bow_valley_inference_aoi.geojson`) is approximately 140 km × 175 km. This exceeds a single Sentinel-2 tile and requires the 2×2 tile grid (`T11UNS/NT/PS/PT`, ~4 scenes of Landsat or Sentinel-2) to approach complete spatial coverage. (Note: the wider 187 km × 162 km figure refers to the cell-sampling bbox, not the clip AOI — see §3 AOI.)
- **Incomplete Daily Coverage**: Because of sensor orbit path timings, swath widths, and scene collection grids, we will **never** have 100% spatial coverage of the full AOI on a single acquisition day/timestamp. Some parts of the AOI will have scenes on day `d`, while other parts will have nodata. **This is quantified per-source from the archive in "Per-Timestep, Per-Source Partial AOI Coverage" below.**
- **Orbit/Swath Boundary Nodata**: Scenes near orbit boundaries or swath edges often contain significant regions of native nodata. Cells that overlap scene edges will have partial observations.
- **Mosaicing & Composite Strategy in Direct-Source Pipeline**:
  - For a given 1 km x 1 km grid cell, it may fall in the overlap region of multiple adjacent/swath-overlapping scenes on the same day, or it may fall on the edge of a scene where part of the cell is nodata.
  - If we replicate GEE's naive `.first()` selection per day per cell, we might ingest a scene that only partially covers the cell even if a fully-covering scene is available, or we might miss coverage in the overlap areas.
  - Therefore, the local adapters must handle mosaicing of all available scenes/granules for the target day *before* cropping to the 1 km grid cell. This ensures maximum coverage and reduces artificial `nodata` boundaries within grid cells.
- **Heterogeneous Daily Coverage in daily mosaic**:
  - The final daily mosaic will always be incomplete (heterogeneous coverage). Some 1 km grid cells will be completely invalid (all `-9999` inputs, leading to `nodata` in predictions), while others will have valid outputs.
  - The `DailyMosaicWriter` must be robust to missing grid cells or cells with degenerate outputs, stitching only valid predictions into the daily UTM 11N COG.

### Per-Timestep, Per-Source Partial AOI Coverage (first-class concern)

The complexity above is not just about *cell-edge* nodata — it is a structural
property of the whole AOI: **on any given timestep `d`, most observational
sources cover only part of `data/bow_valley_inference_aoi.geojson`, and some cover none of it.** The
AOI is ~140 km × 175 km, which exceeds a single Sentinel-2 tile (110 km), a
single MODIS/VIIRS sinusoidal-tile footprint, and a single S1/S3 swath. No
source's per-day footprint is a superset of the AOI.

**This is verified from the archive, not assumed:**

| Source | Per-timestep AOI coverage (observed in `data/bow_valley_selection_raw`) |
| --- | --- |
| **Sentinel-2** | 4-tile AOI grid (`T11UNS/NT/PS/PT`). Of all acquisition dates: **13** cover all 4 tiles, **10** cover 3, **12** cover 2, **2** cover only 1. Most days are **partial**. |
| **Sentinel-1** | Only **16 acquisition dates** over the full 2025-03→05 span (6–12 day revisit), 2 scenes each. **The majority of inference days have NO S1 at all** (→ full `-9999` for the S1 group); on covered days, only a swath-width strip of the AOI is observed. |
| **Landsat 8/9** | 16-day revisit each (≈8-day combined). On a given day, at most one path crosses the AOI → a single ~185 km swath, frequently only partial AOI overlap; many days have neither L8 nor L9. |
| **Sentinel-3 OLCI** | Daily but swath-geometry (~1270 km swath, so usually full AOI when present) — still subject to orbit gaps and edge nodata. |
| **MODIS / VIIRS** | Daily, single sinusoidal tile `h10v03`. Tile footprint does **not** span the full AOI; AOI cells outside the tile are nodata even on a "covered" day. Cross-tile cells need a tile mosaic. |
| **ERA5-Land** | Continuous 0.1° grid → full AOI every day (the only source with guaranteed complete spatial coverage). |

**Implications the pipeline MUST encode (not optional):**

1. **Coverage is per (source, day, cell), not per day.** The per-cell
   `-9999`-placeholder path (`create_placeholder`) is the *normal* case for many
   (source, day) pairs, not an error. The model's masking is the designed
   mechanism for this; do not treat partial coverage as a pipeline failure.
2. **Per-source mosaic-before-crop is mandatory and per-day.** For each source
   and each day, mosaic **all** granules/scenes/tiles intersecting the AOI
   before cropping to any cell (already stated as non-negotiable in §9). A cell
   on a tile/swath seam draws from multiple inputs; a cell outside every
   footprint for that day gets `-9999`.
3. **The 8-day window is what makes partial daily coverage tolerable.** Each
   inference covers `[d-7, d]`; a cell rarely has the same source on all 8 days,
   but usually has *some* coverage across the window. Phase 0 MUST profile, per
   in-AOI cell, the fraction of the 8-day window each source actually populates
   — this is the realistic input-completeness distribution the model sees, and
   it bounds achievable FSC quality. Do not assume dense coverage.
4. **S1 sparsity is the dominant risk.** With S1 present on only ~16 days, many
   windows will have **zero** S1 timesteps. Quantify how often the S1 group is
   fully masked across the inference range in Phase 0; if it is the common case,
   surface it before committing compute — it materially changes what the
   high-res group contributes.
5. **`DailyMosaicWriter` heterogeneity is structural.** A daily output COG will
   have large `nodata` regions wherever no source covered that part of the AOI
   on that window — expected, not a bug. Record per-day AOI-coverage fraction as
   an output metric so downstream consumers know how complete each daily mosaic
   is.

### Archive Directory Formats and Structures

The clip stage (Stage 0→1) **preserves** these structures, formats, and nested
layouts — it only crops pixel extents (see `CLIPPING_PLAN.md`). So the formats
below describe **both** the raw archive (`data/bow_valley_selection_raw/`, the
clip stage's input) and the clipped archive
(`data/clipped_bow_valley_selection_raw/`, which the adapters read). Direct-source
ingestion must handle the following structures, files, and nested formats for the
9 modalities:

- **dem** (Copernicus DEM GLO-30):
  - Path: `data/bow_valley_selection_raw/dem/DEM1_SAR_DGE_30_[meta]/Copernicus_DSM_10_[tile]/`
  - Format: Nested SAFE directory. Under each tile's main directory, there is a `DEM/` subfolder containing a single GeoTIFF file (`..._DEM.tif`) representing elevation in meters above EGM2008 geoid.
- **era5** (ERA5-Land Daily Aggregates):
  - Path: `data/bow_valley_selection_raw/era5/`
  - Format: NetCDF (`.nc`) files. Monthly folders named `YYYYMM_ERA5LAND/` contain separate daily average files for wind and temperature variables: `10m_u_component_of_wind_0_daily-mean.nc`, `10m_v_component_of_wind_0_daily-mean.nc`, `2m_temperature_0_daily-mean.nc`, and `skin_temperature_0_daily-mean.nc`. Daily accumulated precipitation is stored as monthly files in the parent folder, e.g., `YYYYMM_ERA5LAND_totalprecip.nc`.
- **landsat8** / **landsat9** (Landsat Collection 2 Level 1 TOA):
  - Path: `data/bow_valley_selection_raw/landsat8/` and `data/bow_valley_selection_raw/landsat9/`
  - Format: `.tar` files or extracted directories containing individual band GeoTIFF files (`_B2.TIF` through `_B7.TIF` and `_B11.TIF`), a pixel QA band (`_QA_PIXEL.TIF`), and metadata text/JSON/XML files (`_MTL.json`, `_MTL.txt`, `_MTL.xml`) defining scaling coefficients.
- **sentinel1** (Sentinel-1 GRD):
  - Path: `data/bow_valley_selection_raw/sentinel1/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure. Inside the archive, measurements are in `.tiff` files (under `measurement/`) and metadata in `.xml` files.
- **sentinel2** (Sentinel-2 Level-1C):
  - Path: `data/bow_valley_selection_raw/sentinel2/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure. Granules contain JPEG2000 (`.jp2`) band files under `GRANULE/[granule_id]/IMG_DATA/`.
- **sentinel3** (Sentinel-3 OLCI Level-1 EFR):
  - Path: `data/bow_valley_selection_raw/sentinel3/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure for OL_1_EFR products. The radiance bands (e.g. `Oa17_radiance.nc`, `Oa21_radiance.nc`) and coordinate tie-points (`geo_coordinates.nc`) are stored as separate NetCDF files.
- **modis** (MOD09GA daily surface reflectance):
  - Path: `data/bow_valley_selection_raw/modis/`
  - Format: Standard HDF4 (`.hdf`) files representing MOD09GA tiles (e.g., `h10v03`) containing sinusoidal grid subdatasets.
- **worldcover** (ESA WorldCover v200):
  - Path: `data/bow_valley_selection_raw/worldcover/ESA_WorldCover_10m_2021_v200_[tile]_Map/`
  - Format: Categorical GeoTIFF file (`..._Map.tif`) under its respective tile directory.

### Temporal window

The default window is **derived from actual archive coverage**, not chosen for
seasonal convenience. A prior draft defaulted to `2024-02-01 → 2024-04-30`;
**there is no 2024 data in the archive** — every modality spans 2025-03 to
2025-06 (verified file listing below). The 2024 default would have produced an
all-`-9999` cube.

**Per-modality archive coverage (verified from `data/bow_valley_selection_raw`):**

| Modality | First acquisition | Last acquisition |
| --- | --- | --- |
| ERA5-Land | 2025-03 | 2025-05 |
| MODIS (`A2025060`–`A2025151`) | 2025-03-01 | 2025-05-31 |
| VIIRS (`A2025060`–`A2025151`) | 2025-03-01 | 2025-05-31 |
| Sentinel-3 OLCI | 2025-03-01 | 2025-06-09 |
| Landsat 8 | 2025-03-02 | 2025-05-28 |
| Landsat 9 | 2025-03-01 | 2025-05-29 |
| **Sentinel-1 GRD** | **2025-03-30** | 2025-05-31 |
| Sentinel-2 L1C | 2025-03-01 | 2025-05-30 |

**Binding constraints:**
- **S1 start (2025-03-30) is the latest-starting modality.** It dictates the
  earliest *fully-populated* 8-day window. With the non-negotiable 7-day
  prefill, the first inference day with S1 present across the whole window is
  **2025-04-06**.
- **S2 / Landsat 8 end (2025-05-28–30) is the earliest-ending optical modality.**
  It caps the inference range.

| Parameter | Value | Source |
| --- | --- | --- |
| Cube inference period | Configurable; **default `2025-04-06 → 2025-05-28`** | Archive coverage; S1-start-limited start, S2/L8-end-limited end |
| **Archive ingest period** | `start - 7 days → end` (default `2025-03-30 → 2025-05-28`) | Needed to fill the 8-day window for `d = start` |
| Timestep stride | 1 day (`DAYS_PER_TIMESTEP`) | `config.py` |
| Window per inference | 8 days (`NUM_TIMESTEPS`) | `config.py` |
| Prediction cadence | 1 prediction per cell per day, `d ∈ [start, end]` | sliding window |

The 7-day prefill is non-negotiable: the model needs 8 timesteps. Phase 0
archive audit MUST verify ingest coverage of `start − 7` through `end`, not
just the inference range.

**Earlier-start option:** if S1-absent days are acceptable as `-9999` for the
S1 group (the model masks them), the window may start as early as **2025-03-08**
(`start − 7 = 2025-03-01`, the common start of the other modalities). This
trades S1 coverage on the first ~3 weeks for a longer inference span. Decide in
Phase 0 against the audited S1 gap profile; do **not** silently assume S1 is
present before 2025-03-30.

### Filename convention (CONTRACT — resolved here)

The existing `LandsatEvalDataset` parses filenames at
`src/fsc/landsat_eval.py:171-176, 254-262`. Two branches:

- **Landsat-prefixed** (`LC*`, `LE09*`, `LC08*`, `PR*`): name format is
  `L0X_YYYYMMDD_LAT_LON_SC{cloud}.tif`. Month at `parts[1][4:6]`, lat at
  `parts[3]`, lon at `parts[4]`.
- **Non-Landsat** (default): month at `parts[0][5:7]` (i.e. `parts[0]` is
  shaped `XXXXX-MM-DD-...` or similar), lat at `parts[2]`, lon at `parts[3]`.

The non-Landsat branch is brittle and undocumented. **Decision:** the
`LocalSourceExporter` emits the **Landsat-style filename** for every cell:

```
dates={YYYYMMDD}_{window_start_YYYYMMDD}_{LAT}_{LON}_SC00.tif
```

Wait — that doesn't match. Re-reading `landsat_eval.py:171-176`: the
Landsat-style requires `parts[0]` to start with one of `LC|LE09|LC08|PR` and
`parts[1]` to be `YYYYMMDD`. So the exporter emits:

```
PR_{YYYYMMDD_window_end}_{LAT_DEG}_{LON_DEG}_SC00.tif
```

where `PR` is the recognized "synthetic / predicted" prefix already supported
in the parser, and `LAT_DEG`/`LON_DEG` are signed decimal degrees (`53.1234`,
`-115.6789`). **SPEC AC:** regex
`^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$` matches and
`prediction_month_from_file` returns the expected month for every exported tif.

The `PR` prefix's meaning in the existing code MUST be verified during FDD
phase (see §8 Q9) — if `PR` denotes PlanetScope rather than predictions, pick
a different recognized prefix or add a new one and patch the parser allowlist
in the same PR (touches `landsat_eval.py:172` — minimal additive change).

---

## 4. Architecture

Ports & Adapters. The existing `Dataset` is a *consumer* of a logical raster
stack. We introduce a `LocalSourceExporter` that produces the same stack from
local files, plus an `InferenceGridDriver` that orchestrates per-cell
inference. **Upstream of all of it sits the AOI clip stage** (§3 Pipeline
Stages); the adapters' archive root is the clipped output it produces.

```
                ┌──────────────────────────────────────────┐
                │   AOI Clip Stage (CLIPPING_PLAN.md)      │
                │   raw archive → clipped archive           │
                │   §2.0 intersect gate runs here, once     │
                └────────────┬─────────────────────────────┘
                             │ data/clipped_bow_valley_selection_raw
                             ▼
                ┌──────────────────────────────────────────┐
                │       InferenceGridDriver (new)          │
                │  AOI → 1 km grid → per-cell jobs          │
                └────────────┬─────────────────────────────┘
                             │ per (cell, day)
                             ▼
                ┌──────────────────────────────────────────┐
                │  LocalSourceExporter (new)               │
                │  Implements the same contract as          │
                │  EarthEngineExporter.create_ee_image     │
                └────────────┬─────────────────────────────┘
                             │ assembles dynamic + static
                             ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Source Adapters (new, one per modality)                 │
        │  S1GRDLocal • S2HarmonizedLocal • Landsat89Local •        │
        │  S3OLCILocal • MOD09GALocal • VNP09GALocal •              │
        │  ERA5LandLocal • CopernicusDEMLocal • ESAWorldCoverLocal │
        └────────────┬─────────────────────────────────────────────┘
                     │ GeoTIFF in canonical layout
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   LandsatEvalDataset (existing, unchanged)               │
        │   → MaskedOutput groups                                  │
        └────────────┬─────────────────────────────────────────────┘
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   EncoderWithHead (existing) → per-cell 10×10 FSC        │
        └────────────┬─────────────────────────────────────────────┘
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   DailyMosaicWriter (new) → daily FSC COG over AOI       │
        └──────────────────────────────────────────────────────────┘
```

### Module layout (new)

```
src/data/local_sources/
  __init__.py
  base.py             # LocalSourceAdapter ABC, GridCell dataclass, CellWindow
  s1.py               # Sentinel-1 GRD adapter
  s2.py               # Sentinel-2 L1C → harmonized DN
  landsat.py          # Landsat 8/9 C2 T1 TOA (encapsulates L9→L8 fallback)
  s3.py               # Sentinel-3 OLCI radiance
  modis.py            # MOD09GA (HDF/Mosaic); preserves -28672 fill value
  viirs.py            # VNP09GA (HDF/Mosaic) — emits both fine (per-pixel) and coarse (per-pixel raster, loader averages)
  era5.py             # ERA5-Land daily aggregates (NetCDF/GRIB); emits raw Kelvin
  dem.py              # Copernicus DEM GLO-30 + slope + aspect
  worldcover.py       # ESA WorldCover v200 (static, 2021 map; ignores day)
  cube_cache.py       # Per-modality per-(cell, day) numpy cache (.npz)
  exporter.py         # LocalSourceExporter — assembles multiband tif from cache
  grid.py             # AOI → 1 km grid generator (modes A and B)
  layout.py           # Canonical band order constants (re-exports from eo.py)

src/inference/
  __init__.py
  driver.py           # InferenceGridDriver
  mosaic.py           # DailyMosaicWriter (per-day COG, UTM 11N)
  windows.py          # Sliding 8-day window builder

scripts/
  export_bow_valley_cube.py     # build the cube from local archive
  infer_bow_valley_daily_fsc.py # run model + mosaic per day

configs/bow_valley/
  cube.yaml          # archive_root (clipped), processing_root, AOI bbox, date range, CRS, mode A/B, cache cap
  inference.yaml     # checkpoint path, batch size, output dir (default processing_root/daily_fsc), days
  cube_cells.csv     # generated cross-product CSV (cells × window days); emitted by grid.py

tests/test_local_sources/
  test_grid.py
  test_filename_contract.py    # exporter filenames pass LandsatEvalDataset parser
  test_<modality>_adapter.py   # one per adapter, value-domain + CRS asserts
  test_s1_parity.py            # GEE parity spike (Phase 3 step 0)
  test_s2_parity.py            # GEE parity spike (Phase 3 step 0)
  test_exporter_parity.py      # full-stack numeric parity vs GEE reference patches
  test_tracer_end_to_end.py    # one cell, one day → EncoderWithHead → assertions
  test_inference_driver.py
```

### Adapter contract

Every adapter implements:

```python
class LocalSourceAdapter(Protocol):
    bands_out: list[str]                       # exact names expected downstream
    spatial_kind: Literal["high","med","low","time","space","static"]
    native_fill: float | None                  # e.g. -28672 for MODIS; None if only -9999

    def fetch(
        self,
        cell: GridCell,                        # polygon, CRS, target transform, shape
        day: datetime.date | None,             # None for static layers
    ) -> np.ndarray:                           # shape (C, H, W); -9999 nodata
        ...
```

Rules enforced by `base.py`:
- Output reprojected to the cell's target grid (`EPSG:4326`, scale=10) using:
  - **bilinear** for continuous bands,
  - **nearest** for QA / categorical (WorldCover, cloud flags).
- **Nodata-aware bilinear (mandatory in the shared resampler).** Before a
  bilinear warp, fill/nodata pixels (`-9999`, and MODIS `-28672`) are masked to
  NaN so the interpolator never blends a valid value with a fill sentinel; the
  output restores `-9999`/`-28672` wherever the contributing source pixels were
  fill. Without this, a valid reflectance interpolated against `-28672` yields a
  garbage negative (e.g. `-5000`) that slips past the
  `CHANNEL_WISE_INVALID_DATA_THRESHOLDS` and pollutes model input. This is an
  edge-bleed risk for **every** continuous band near a nodata edge — MODIS fill,
  S2/S1 swath edges, Landsat scene borders — so it lives in `base.py`, not just
  the MODIS adapter.
- Missing acquisition → return `-9999` array of declared shape
  (`create_placeholder` equivalent).
- **Same-tile/date coalesce before mosaic-before-crop.** For scene/granule
  sources (S2, Landsat, and any swath source), an adapter resolving a `(cell,
  day)` MUST gather **all** products sharing the same (reference tile, date) — not
  `.first()` — and coalesce them per pixel: first valid (non-nodata, in-threshold)
  value wins, fall through to the next product where nodata, `-9999` only where
  all are nodata. Deterministic order = latest processing time first. This
  coalesce runs **per tile, before** the cross-tile mosaic-before-crop step. It is
  a valid-pixel union, not an average (preserves GEE value domain). Prevents
  false `-9999` when one product has a swath-edge/cloud gap that another product
  fills. (Non-negotiable §9; rationale in `DATA_ANALYSIS.md`.)
- MODIS adapter MUST preserve the native `-28672` fill value in addition to
  `-9999`. The downstream NDSI/NDVI computation at
  `src/fsc/landsat_eval.py:317, 331` asserts the fill value is *encountered*
  (sentinel for "MODIS data was actually present"). Stripping it will crash
  the loader.
- VIIRS coarse adapter emits a per-pixel raster at the cell grid. The loader
  averages spatially into `time_x`. Do not pre-average in the adapter.
- ERA5 adapter emits raw Kelvin / native units. The known temperature shift
  inconsistency lives in `Normalizer`, downstream of the adapter; adapter is
  not responsible for replicating it.
- Landsat adapter encapsulates the L9→L8 fallback internally (single
  `bands_out=["B2_landsat",..,"B7_landsat"]`), matching GEE behaviour.
- **Sentinel-2 harmonization**: Sentinel-2 data in GEE (both Level-1C and Level-2A) are harmonized to correct for the processing baseline baseline 04.00+ offset (+1000 DN). Direct Copernicus products do NOT have this harmonization. The local S2 adapter must check the processing baseline version of each input granule and subtract 1000 from the digital numbers if the baseline is `04.00` or later to ensure a harmonized time series matching the model's expectations.
- **ERA5-Land daily aggregation + precip day-shift**: the archive on disk is **already
  daily-aggregated** (one slice/day), so the adapter reads daily files directly — it does
  **not** re-aggregate hourly data. **`total_precipitation` is a forecast accumulation**
  (`GRIB_stepType=accum`, units m), and ERA5-Land stamps the total that closes day `i` at
  **`00:00` of day `i+1`**. The adapter MUST therefore read precip for day `i` from the
  **`i+1` `00:00` slice** (equivalently `tp[index] → day index−1`). The instantaneous
  temp/wind vars (`t2m`, `skt`, `u10`, `v10`) are daily means with **no** day shift. A
  naive label-based precip read is a silent off-by-one (passes shape/type checks). If
  daily files ever need regenerating from hourly CDS data: daily mean over the UTC day
  for temps/winds; for precip take the end-of-day accumulation (the next day's `00:00`),
  never a naive 24-value sum and never stopping at 23:00. See `DATA_ANALYSIS.md` → ERA5
  accumulation gotcha.
- **Copernicus DEM terrain metrics**: Slope and aspect are scale-sensitive. GEE
  (`src/data/earthengine/copernicus_dem.py:14-16`) computes `ee.Terrain.slope`/
  `aspect` on the DEM's **native grid** using true ground pixel dimensions
  (latitude-aware metres-per-pixel), *then* the export resamples to the 4326/
  scale=10 cell grid. The local DEM adapter must replicate that: compute slope/
  aspect with **latitude-correct metric pixel spacing** (supply the real
  metres-per-pixel in x/y at the cell's latitude to the Horn kernel), then
  resample DEM+slope+aspect to the 10 m cell grid. **Do NOT detour through
  EPSG:32611 to compute terrain** — the GEE reference patches were never computed
  in a UTM frame, so a UTM-computed slope/aspect fails parity (AC-21). The bug to
  avoid is running the kernel on a degree grid with unit (`1°≈1 m`) pixel
  spacing, which scales gradients by ~111,000× and forces all slopes toward 90° —
  that is a *pixel-spacing* error, not a projection error, and is fixed by
  passing correct metric spacing, not by changing CRS.
- **Sentinel-3 OLCI geolocation**: S3 OLCI SAFE products contain separate NetCDF files georeferenced by coordinate tie-point grids. The local adapter must use these geolocation arrays to precisely project OLCI radiance bands onto the target cell grid.
- WorldCover adapter ignores `day` and returns the v200 2021 map. Hardcoded.

### LocalSourceExporter

Mirrors `create_ee_image` exactly:
- Iterates `NUM_TIMESTEPS=8` days.
- Calls each time-varying adapter per day in the **fixed order**:
  `S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5
   + MODIS cloud flag + S2 cloud flag + Landsat cloud flag`.
- Appends static stack once: `DEM, slope, aspect, WorldCover Map`.
- Writes a multiband GeoTIFF per (cell, window-end-day) under the Landsat-style
  filename defined in §3.

### Cube cache layout (correctness fix vs prior draft)

The cache is **per-modality per-(cell, day)** numpy arrays (`.npz`), not
per-(cell, window-end-day) multiband tifs. Rationale: 8-day windows for
consecutive days overlap by 7 days. Caching at the multiband-tif level would
duplicate ~8× the storage. The exporter:

1. For each `(cell, day)` in the 8-day window, queries the cache
   `cube_cache.get(modality, cell_id, day)`.
2. On miss, calls the adapter, writes the array to `.npz`, returns it.
3. After all 8 days × all modalities are gathered, assembles the multiband tif
   in the canonical band order and writes it to
   `data/bow_valley_processing/cubes/` (the assembled-cube final destination).

The per-day `.npz` cache lives in `data/bow_valley_processing/cube_cache/`; both
roots derive from `processing_root` in `cube.yaml` (see §3 Directory layout).

**Cache directory sharding (filesystem-performance fix).** The cache is **sharded
one subdirectory per cell**:

```
cube_cache/{cell_id}/{day}_{modality}.npz
```

A flat `cube_cache/{cell_id}_{day}_{modality}.npz` layout would put **~300k
files** (mode A: ~344 cells × ~96 archive days × ~9 modalities) in a single
directory, degrading ext4/xfs directory indexing to O(N) on every lookup and
eviction scan. Per-cell sharding keeps each directory under ~1k entries
(~96 days × ~9 modalities ≈ 864 files/cell). Per-cell is enough — no hash-prefix
tier required at this scale.

Storage estimate (mode A, 500 cells × 90 inference days = 96 archive days):

- Per-cell per-day per-modality: ~100×100 × float32 × bands. S1 (3) + S2 (6+1
  QA) + Landsat (6+1 QA) + S3 (2) + MODIS (7+1 QA) + VIIRS fine (2) +
  VIIRS coarse (4) + ERA5 (5) ≈ 38 bands × 40 KB = 1.5 MB per (cell, day).
- 500 cells × 96 days × 1.5 MB ≈ **72 GB cache**, plus ~75 GB of assembled
  multiband tifs (500 × 90 × ~1.6 MB).

Cache (`data/bow_valley_processing/cube_cache/`) is FIFO-evicted with a size cap
(configurable, default 200 GB); it and `scratch/` are intermediate and cleanable
mid-run. `processing_root` is configured in `cube.yaml` (see §3 Directory
layout).

---

## 5. Inference Grid Driver

Pseudocode:

```python
grid = build_grid(aoi_bbox_utm, cell_size_m=1000, crs="EPSG:32611", mode=mode_A_or_B)
for day in daterange(start, end):                              # inference range
    cube_paths = []
    for cell in grid:                                          # parallel
        # window covers [day - 7, day], inclusive
        tif = exporter.export(cell, window_end=day)
        cube_paths.append((cell, tif))
    batches = batch_cells(cube_paths, batch_size=N)
    preds = run_encoder_with_head(model, batches)              # (B, 10, 10) FSC
    mosaic.write_day(day, preds, grid)                         # COG, EPSG:32611
```

Key design choices:
- **Cube reuse:** see §4 cache layout — per-modality per-day, not per-window.
- **Parallelism:** per-cell export is embarrassingly parallel
  (`multiprocessing.Pool`). Inference is GPU-batched across cells.
- **Mosaic:** `DailyMosaicWriter` writes one COG per day in EPSG:32611 to
  `data/bow_valley_processing/daily_fsc/` (the inference deliverable; overridable
  to object storage per §8 Q5). Each
  cell's 10×10 FSC patch maps to a 100 m pixel grid (`100 m × 10 px = 1 km`
  cell). Cells with all input groups masked fall back to `nodata`. The 10×10
  FSC raster from each cell is reprojected from EPSG:4326 (the loader's grid)
  to EPSG:32611 at mosaic-write time using nearest-neighbour (FSC is a
  prediction; bilinear would blend invalid neighbours).
- **Per-cell independence:** the model runs one forward pass per 1 km cell
  with no cross-cell spatial context (`patch_size_high_res=10`, input 100×100,
  output 10×10 at `src/fsc/patch_predict.py:26`). Edge effects on the 100 m
  output pixels at cell boundaries are a known *modelling* limitation, not a
  mosaic-stitching bug. Documented in §6.

---

## 6. Verification & FMEA (sketch — formalized in FDD)

| Risk | Mitigation |
| --- | --- |
| Filename contract mismatch with `LandsatEvalDataset` parser | `test_filename_contract.py` asserts every exporter-emitted filename parses correctly via `prediction_month_from_file` and yields expected `(month, lat, lon)`. Tested before any adapter is implemented. |
| Adapter value-domain drift vs GEE | Per-adapter parity test against GEE reference patch (numeric diff thresholds per source). S1 and S2 parity spikes run **first** (Phase 3 step 0) — highest interchange risk per `DATA_ANALYSIS.md`. |
| CRS / pixel-alignment mismatch | All adapters share one resampler in `base.py` driven by `GridCell.transform`. Golden-grid test asserts exact `transform`, `shape`, `crs`. |
| Band order regression | `layout.py` re-exports the canonical lists from `eo.py`; integration test asserts byte-for-byte band-name equality against `create_ee_image` output. |
| MODIS native fill stripped | MODIS adapter test asserts `-28672` is preserved in output where source had it. NDSI/NDVI assertions at `landsat_eval.py:317, 331` would crash otherwise. |
| Landsat L9→L8 fallback regression | Landsat adapter test exercises 3 scenarios: L9 present, L9 missing + L8 present, both missing → `-9999`. |
| VIIRS coarse pre-averaging breaks `time_x` | VIIRS coarse adapter returns shape `(4, 100, 100)`, not `(4,)`. Test asserts shape and that the loader's spatial mean reproduces GEE values. |
| Mosaic seams between adjacent cells | Cells are non-overlapping by design; verify via overlap=0 assertion in `grid.py` and a 2×2 mosaic visual test. Cross-cell context limitation documented separately. |
| **False `-9999` from picking one of several same-tile/date products** (S2 R070-vs-R113, S2A/S2B, reprocessing dupes; Landsat 9 dupes — verified in archive) | Adapter coalesces **all** same-(tile,date) products per pixel (first valid wins, fall through nodata), never `.first()`. Test: two synthetic same-tile/date products with complementary nodata masks → coalesced output has **zero** nodata where either input was valid, and the surviving value matches the deterministic-order winner. |
| Memory/IO blowup on full AOI sweep | Per-modality per-(cell, day) `.npz` cache, FIFO-evicted, size cap. S1 SAFE archives processed via windowed reads (read-only the cell footprint), not full-scene loads. Concrete sizing: §4. |
| Cloud flag bands dropped silently downstream | Emit them in the GeoTIFF anyway; loader will drop them. Documented. Keeps GEE byte-layout parity. |
| ERA5 normalization bug accidentally "fixed" | Adapter emits Kelvin; bug lives in `Normalizer` and stays as-is (out of scope per §2). |
| Per-cell modelling edge effects at 100 m output boundaries | Documented limitation: every cell is an independent forward pass. No mitigation planned in this work. Flag in `KNOWLEDGE.md`. |

### Tracer-bullet integration test (concrete assertions)

Export one cell for one window-end-day from local archive, run it through
`LandsatEvalDataset` + `EncoderWithHead`, assert:

1. `space_time_high_res_x.shape == (100, 100, 8, 15)`
2. `space_time_med_res_x.shape == (5, 5, 8, 2)` (note: docstring at
   `landsat_eval.py:236` says `(3, 3, T, C_STM)` but is stale; actual target
   is 5×5 per `DATA_ANALYSIS.md` §Compatibility caveats)
3. `space_time_low_res_x.shape == (2, 2, 8, 11)` (incl. NDSI + NDVI)
4. `time_x.shape == (8, 9)`
5. `space_x.shape == (100, 100, 14)`
6. `static_x.shape == (3,)`
7. FSC prediction shape `== (10, 10)`, values ∈ `[0.0, 1.0]`
8. `valid_data_mask_*` all set wherever inputs are `-9999` or below their
   `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`.
9. Filename parses correctly: `prediction_month_from_file` returns the month
   matching `window_end.month`.

These nine assertions are the SPEC's primary ACs.

### Tribal knowledge

`KNOWLEDGE.md` (create if missing) gets entries for:
- MODIS native fill `-28672` is sentinel for "data present" — don't strip.
- ERA5 normalization sign error is known and deliberately preserved.
- S3 normalization TODO is known; identity is intentional.
- `PR` filename prefix is recognized by the eval-dataset parser; verify its
  original meaning before reusing.
- Per-cell forward pass has no cross-cell spatial context — 100 m FSC pixels
  near cell boundaries may have edge effects.

---

## 7. Phased Delivery

Each phase ends with an explicit approval gate per CLAUDE.md workflow rules.

1. **Phase 0 — Archive Audit + Cube CSV + GEE Reference Patch Generation (no
   production adapters).**
   - Catalog every file we have for each modality: paths, formats (SAFE / HDF /
     NetCDF / COG), CRS, native scale, coverage of `[start-7, end]` × AOI,
     gaps.
   - **Generate the cube CSV** (§3 "Generated cube CSV"): stand up the geometry
     half of the grid generator (`grid.py`, pure CRS/polygon math — no adapters)
     and emit the full-cross-product CSV (in-AOI cells × inference-window days)
     in the canonical schema. This artifact drives both the sweep and the
     reference run.
   - Run the existing CSV-driven GEE exporter
     (`scripts/export_for_inference.py` → `EarthEngineExporterEval.export_from_csv_utm`,
     `src/data/earthengine/eo_eval.py:576` — **not** `export_for_eval.py`) over a
     **5–10 row sample of the generated cube CSV** to produce **GEE reference
     patches** used by every parity test in Phase 2/3. Sampling from the
     generated CSV guarantees each reference cell is in-AOI and in-archive by
     construction.
   - Verify the meaning of the `PR` filename prefix in `landsat_eval.py:172`.
   - Output: `docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md` +
     `configs/bow_valley/cube_cells.csv` (generated) +
     `tests/fixtures/gee_reference_patches/`.
2. **Phase 0.5 — AOI Clip (prerequisite stage, per `CLIPPING_PLAN.md`).**
   Produces `data/clipped_bow_valley_selection_raw`, the archive root every
   adapter reads (§3 Pipeline Stages, §4). Must complete before any adapter
   work in Phase 3 — the cube cannot be built from the raw archive.
   - Implement `scripts/developer_scripts/clip_dataset.py` (Typer CLI) and its
     validation script per `CLIPPING_PLAN.md §3`.
   - Run the §2.0 intersect gate per product; emit the per-source clip manifest.
   - **Exit gate (mandatory):** the post-run audit asserts
     `data/clipped_bow_valley_selection_raw` contains **zero** all-nodata /
     zero-valid-pixel outputs, and the clip manifest accounts for every input
     product (`CLIP | SKIP_NO_OVERLAP | SKIP_DEGENERATE_OVERLAP`). Phase 0's
     static-layer coverage assertion (DEM/WorldCover mosaics reach `lat 52.31`)
     is re-checked against the clipped output. Approval gate.
3. **Phase 1 — FDD.** Formal Design Document per planning skill, including the
   tracer test of §6 with the nine concrete assertions. Approval gate.
4. **Phase 2 — SPEC.** Acceptance criteria as test sentences. Per-adapter
   value-domain assertions, per-adapter parity thresholds (numeric diff
   tolerance) vs GEE reference patches generated in Phase 0. Approval gate.
5. **Phase 3 — Tasks (vertical slices).** Re-ordered to de-risk high-impact
   adapters early.
   - **Step 0 — Parity spikes (throwaway).** Stand up minimal S1 and S2 GRD/L1C
     download + reprojection scripts. Compare to GEE reference patches.
     Quantify drift. *Decision point:* if drift is too large to recover with
     processing, escalate before sinking effort into the full ports/adapters
     stack.
   - **Step 1 — Contract.** `base.py` + `GridCell` + `grid.py` (productionize the
     geometry half built in Phase 0: cross-product CSV emission, kept/dropped
     manifest, mode A/B) + `layout.py` + `cube_cache.py`. `test_grid.py`,
     `test_filename_contract.py` pass.
   - **Step 2 — Placeholder exporter + tracer test.** `LocalSourceExporter` +
     placeholder adapters returning `-9999`. `test_tracer_end_to_end.py` passes
     with all-`-9999` cubes (FSC will be degenerate but pipeline plumbed).
   - **Step 3 — Adapters, in difficulty / parity-risk order:**
     1. `worldcover.py` (static, easy)
     2. `dem.py` (static, easy)
     3. `era5.py` (low parity risk, well-defined NetCDF)
     4. `modis.py` (mind the `-28672` fill)
     5. `viirs.py`
     6. `s3.py`
     7. `landsat.py` (with L9→L8 fallback test)
     8. `s2.py` (parity spike already done, now production)
     9. `s1.py` (parity spike already done, now production)
   - **Step 4 — `InferenceGridDriver` + `DailyMosaicWriter`.**
   - **Step 5 — `scripts/export_bow_valley_cube.py` and
     `scripts/infer_bow_valley_daily_fsc.py`.**

### Tooling per task

Detected from repo state (confirm in FDD):
- Linting: `ruff` (default per CLAUDE.md if `.pre-commit-config.yaml` does not
  specify otherwise).
- Type checking: `mypy` (default per CLAUDE.md).
- Tests: `pytest`.

Every Phase 3 step must pass `ruff check`, `mypy`, and the relevant test set
before approval.

---

## 8. Open Questions (need user input before FDD)

1. **Archive locations & data-flow contract. [RESOLVED]** Raw archive under the
   symlink `data/bow_valley_selection_raw` → `/archive/data/ai4snow/bow_valley_selection_raw/`,
   with subfolders `dem`, `era5`, `landsat8`, `landsat9`, `modis`, `sentinel1`,
   `sentinel2`, `sentinel3`, `viirs`, `worldcover`. The AOI clip stage
   (`CLIPPING_PLAN.md`, §7 Phase 0.5) writes a same-layout **clipped** mirror at
   the symlink `data/clipped_bow_valley_selection_raw`. **Contract resolved: the
   `LocalSource*` adapters read the clipped archive, not the raw one** (see §3
   "Pipeline Stages & Data Flow"). The clip stage is therefore a mandatory
   on-path prerequisite, and `data/bow_valley_inference_aoi.geojson` is the single binding extent
   end-to-end.
2. **Date window. [RESOLVED]** Default set to **`2025-04-06 → 2025-05-28`**,
   derived from verified archive coverage (see §3 Temporal window). The earlier
   `2024-02-01 → 2024-04-30` draft had **no archive data** and is discarded.
   **Note:** the CSV-recorded dates (2024-01-05 → 2025-12-22, cited in a prior
   draft) are *cell-sampling* metadata, **not** archive acquisition dates — do
   not use them to scope ingestion. See Q4 for what the CSV date column means.
3. **Sweep mode. [PARTIALLY RESOLVED]** Default **(A)** sample-only, over the
   **in-AOI** CSV cells (~344 after the `data/bow_valley_inference_aoi.geojson` centre-in filter, see
   §3). Mode (B) tiles `data/bow_valley_inference_aoi.geojson`, not the wider cell-sampling bbox.
   Remaining input: confirm A vs B for the production run (drives compute).
4. **CSV `date` column semantics. [RESOLVED]** The `date` column in
   `sampled_cells_bow_river_with_dates.csv` is **training/evaluation sampling
   metadata** — the day each cell was drawn for label pairing in the existing
   train/eval pipeline (`LandsatEvalDataset` matches an input tif to a label tif
   by date+coords). **It is NOT a per-cell prediction day for this inference
   run.** This plan is an *inference* job: the driver iterates the configured
   window from §3 (`2025-04-06 → 2025-05-28`, every day, every in-AOI cell) and
   **does not read the CSV `date` at all**. The CSV is consumed here for **cell
   geometry only** (`center_x/y` and bounds) when building the mode-A grid; the
   `date` column is ignored. Consequences: the driver loop is "all in-AOI cells ×
   every day in the configured window" (no per-cell date intersection), and the
   2024–2025 span of the CSV dates is irrelevant to ingestion scoping. The
   earlier worry that "cells whose date falls outside the archive cannot be
   served" was a category error — it imported train/eval label-pairing semantics
   into an inference run.
5. **Output destination.** Daily COGs default to local disk at
   `data/bow_valley_processing/daily_fsc/` (see §3 Directory layout). Remaining
   input: confirm local disk vs object storage for the production run
   (`inference.yaml` output path is the switch).
6. **Checkpoint.** Which finetuned `EncoderWithHead` checkpoint feeds
   inference? Path?
7. **Compute budget.** GPU count and wall-clock target. Mode (A): ~45 k
   forwards, hours on one GPU. Mode (B): ~2.7 M forwards, needs multi-GPU.
8. **Sentinel-2 product level. [RESOLVED]** Verified: all 116 archive granules
   are **L1C** (`MSIL1C`, zero `MSIL2A`), processing baseline **`N0511`**
   (= 04.00+) across the board. Matches `S2_HARMONIZED` value domain. The
   −1000 DN harmonization (§3, DATA_ANALYSIS §S2) is **required for every
   granule**. Tiles present: `T11UNS, T11UNT, T11UPS, T11UPT` (the 2×2 grid).
9. **`PR` filename prefix meaning. [RESOLVED]** Inspected the codebase and existing files. `data/eval_tifs` only contains `LC09` files. The prefix `PR` is unused on disk but supported in `src/fsc/landsat_eval.py` parser, making it fully safe to use as the prefix for our synthetic/predicted direct-source input files to ensure correct downstream coordinate parsing.
10. **Cloud-flag emission.** Keep emitting (default; preserves GEE byte
    layout, dropped downstream) or skip to save IO? Recommend keep.

---

## 9. Non-Negotiables (from `DATA_ANALYSIS.md`)

- Dynamic band order per timestep is fixed.
- Nodata is `-9999`. MODIS additionally preserves native
  `MODIS_FILL_VALUE=-28672` — both must survive into the exported tif.
- Per-cell export grid is `EPSG:4326`, `scale=10`, ~100×100 px per cell.
- WorldCover stays as a single `Map` band; the loader one-hot encodes.
- Normalization constants are tuned to GEE-exported numeric ranges — do not
  "fix" the ERA5 temperature sign or S3 identity normalization in this work.
- Categorical / QA → nearest-neighbor resampling only.
- VIIRS coarse bands are exported as per-pixel rasters on the cell grid; the
  loader (not the adapter) does the spatial mean into `time_x`.
- Landsat L9→L8 fallback is encapsulated inside the Landsat adapter.
- **Mosaicing overlapping daily scenes is mandatory**: Any scene overlaps or swath edge boundaries must be composite-mosaiced prior to cropping to avoid artificial nodata boundaries within a single 1 km grid cell.
- **Same-tile/date multi-product valid-pixel coalescing is mandatory** (distinct from the cross-tile mosaic above): when more than one product covers the *same* reference tile on the *same* date (different orbit/satellite/reprocessing — verified for S2 and Landsat 9 in this archive), the adapter must coalesce them per pixel — take the first product with a valid (non-nodata, in-threshold) value, fall through to the next where nodata, and emit `-9999` only where **all** same-tile-date products are nodata. Deterministic product order (latest processing time first) settles ties. **No value blending** (coalesce, not average) to preserve the GEE value domain. Replacing GEE's `.first()` with this coalesce is what prevents false `-9999` from a swath-edge/cloud gap in one product when another product has the pixel. See `DATA_ANALYSIS.md` → "Same-tile/date multi-product overlap".

