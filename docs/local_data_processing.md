# Local Data Processing — Bow Valley Direct-Source Pipeline

How to recreate the Bow Valley inference data pipeline from raw archive to
clipped archive. Run steps in order. Each task appends its own section here.

> **Living doc.** Future tasks (TASK-003+) add their steps below. Keep it
> concise: commands, paths, and gotchas — not narrative.

---

## 0. Prerequisites

- **Data is downloaded manually (for now).** No automated fetch yet. You must
  place the raw archive at `data/bow_valley_selection_raw/` yourself, one
  subdirectory per source, before any step runs. Sources, formats, and counts
  are cataloged in `docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md`. The
  pipeline is read-only on this directory — it never writes back into it.
- **Environment.** `uv` manages deps. Run everything with `uv run …`. System
  GDAL 3.8+ (`gdalinfo`, `gdal_translate`) must be on `PATH` — rasterio's bundled
  GDAL lacks the HDF4 driver MODIS needs.
- **Earth Engine (optional, TASK-001 reference patches only).** Set `EE_PROJECT`
  in a repo-root `.env` (see `.env.example`); it overrides the default project.

---

## 1. Raw archive layout

```
data/
├── aoi.geojson                      # authoritative clip + inference boundary (EPSG:4326)
├── bow_valley_selection_raw/        # raw input — placed here manually
│   ├── dem/                         # Copernicus DEM, nested SAFE; *_DEM.tif tiles
│   ├── worldcover/                  # ESA WorldCover; *_Map.tif tiles
│   ├── era5/                        # ERA5-Land NetCDF (.nc)
│   ├── landsat8/  landsat9/         # Collection-2 .tar (band GeoTIFFs, EPSG:32612)
│   ├── modis/                       # MOD09GA HDF4 (.hdf), sinusoidal
│   ├── viirs/                       # VNP09GA HDF5 (.h5), sinusoidal
│   ├── sentinel1/                   # S1 GRD .zip (range geometry, GCPs)
│   ├── sentinel2/                   # S2 L1C .zip (JP2, EPSG:32611)
│   └── sentinel3/                   # S3 OLCI .zip (NetCDF tie-point grids)
└── clipped_bow_valley_selection_raw/  # OUTPUT of the clip stage (starts empty)
```

The clipped archive is the **single root every downstream adapter reads**. AOI
authority lives in `aoi.geojson` — do not hardcode bounds elsewhere.

---

## 2. TASK-001 — Phase 0: audit, cube CSV, GEE reference patches

Produces the sweep enumeration and parity fixtures every later task consumes.

```bash
# Emit the generated cube CSV (cells × inference-window days, full cross-product)
uv run python -m src.data.local_sources.grid --emit-csv --mode A \
    --window-start 2025-04-06 --window-end 2025-05-28

uv run pytest tests/test_local_sources/test_grid.py tests/test_local_sources/test_cube_csv.py -q
```

**Outputs:**
- `configs/bow_valley/cube_cells.csv` — 18 232 rows (344 in-AOI cells × 53 days),
  schema `date,crs,center_x,center_y,min_x,min_y,max_x,max_y`, all `EPSG:32611`.
- `configs/bow_valley/cell_filter_manifest.csv` — 500 cells, 344 KEEP / 156 DROP.
- `docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md` — full archive catalog.
- `tests/fixtures/gee_reference_patches/` — 6 GeoTIFFs (308 bands) for parity.

**Gotchas:**
- Grid math is EPSG:32611. AOI filter reprojects cell centres to 4326. 156 of
  500 cells (31%) fall outside the AOI and are dropped by design.
- The legacy `date` column in `sampled_cells_bow_river_with_dates.csv` is never
  read — only cell geometry is reused; dates come from the inference window.

---

## 3. TASK-002 — Phase 0.5: AOI clip stage

Crops every raw source to `aoi.geojson`, non-destructively, into
`data/clipped_bow_valley_selection_raw/`. A two-stage intersect gate skips
products that miss the AOI (no output file). Per-source manifest records every
decision.

```bash
# Dry-run: gate only, no pixels decoded, no writes — sanity check first
uv run python scripts/developer_scripts/clip_dataset.py clip-all --dry-run

# Real clip (all sources). Or one: clip-source worldcover
uv run python scripts/developer_scripts/clip_dataset.py clip-all

# Post-run audit: zero all-nodata outputs, static mosaics reach lat 52.31
uv run python scripts/developer_scripts/clip_audit.py --root data/clipped_bow_valley_selection_raw

uv run pytest tests/test_clip_dataset.py -q
```

**Code lives in the package**, not the scripts: `src/data/local_sources/clip/`
(`settings`, `gate`, `footprints`, `gdal_io`, `clippers`, `manifest`,
`orchestrator`). The two CLIs are thin entrypoints.

**Outputs:**
- `data/clipped_bow_valley_selection_raw/<source>/…` — clipped products, native
  CRS/format/pixels preserved.
- `data/clipped_bow_valley_selection_raw/<source>/clip_manifest.csv` +
  combined `clip_manifest.csv` at the root.

**Gotchas:**
- Intersect gate is the **one** place footprint-vs-AOI filtering happens.
  Adapters must not re-implement it. `CLIP_MIN_AOI_OVERLAP_AREA_KM2` (default
  1 km²) is env-overridable.
- MODIS/VIIRS output is **per-grid GeoTIFFs** (`<grid>__<band>.tif`), one per
  subdataset. The 500 m grid clips to ~2× the 1 km grid — indexed from each
  grid's own geotransform, never a hardcoded 1200 clamp.
- Landsat stays EPSG:32612, S2 stays EPSG:32611. Cross-zone reprojection to the
  cell grid is the adapter's job (TASK-012), not the clip stage.
- Expected dry-run verdict on the curated archive: ~531 CLIP / 2 SKIP_NO_OVERLAP
  (the two W120 WorldCover tiles sit west of the AOI).
---

## 4. Spatiotemporal alignment & 8-day cube assembly

Ensures spatial location and date matching are preserved from raw clipped sources when building 308-band assembled cubes.

### Location & Date Safeguards
- **Spatial Grid (`cube_cells.csv`):** Authoritative cell geometry defined in UTM 11N (`EPSG:32611`). Bounding boxes (`min_x`, `min_y`, `max_x`, `max_y`) map to target cell coordinates.
- **Date Target:** Row `date` defines sliding window end day $d$; sliding window spans 8 days $[d-7, d]$.
- **Filename Contract:** Assembled cubes written to `data/bow_valley_processing/cubes/` with standard format `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (signed decimal degrees of cell center and window-end date). Validated via `test_filename_contract.py` to match `LandsatEvalDataset` parser.

### Pipeline & Data-Flow Integration
1. **Clip Stage Footprint manifest (`combined_clip_manifest.csv`):** Keeps record of geographic bounds, overlaps, and pixel validity of clipped source archives on disk.
2. **Adapter Date Parsing:** `LocalSource*` adapters scan clipped directory for files matching specific target days $i \in [d-7, d]$. Empty days filled with `-9999` placeholder. Sentinel-2 baseline version DN offset (-1000) applied for baseline 04.00+.
3. **Reprojection & 10 m Resampling:**
   - Adapter reprojects target UTM 11N bounding box to native source CRS:
     - Landsat: `EPSG:32612` (UTM 12N)
     - Sentinel-2: `EPSG:32611` (UTM 11N)
     - Sentinel-1: GCP range geometry window (GCP-based pixel slice)
     - MODIS/VIIRS: Sinusoidal projection (`+proj=sinu +R=6371007.181`)
   - Resampled to $100 \times 100$ pixel grid via robust resampler in `base.py`.
   - Coalesces overlapping orbit/scene pixels (first valid wins) to prevent false `-9999` nodata.
4. **Intermediate Cache Sharding:** Per-day per-cell arrays cached under `data/bow_valley_processing/cube_cache/{cell_id}/{day}_{modality}.npz`. Avoids 8x storage duplicate of sliding windows and filesystem indexing limits.
5. **Cube Assembly (`LocalSourceExporter`):** Composites 8 daily cached arrays in exact canonical band order (308 bands).

---

## Testing baseline

The suite is **already red on a clean checkout** (6 pre-existing failures, see
`docs/agents/planning/raw-data-ingestion/tasks/TEST_BASELINE.md`). Judge work by
**delta** — never `pytest -x` at the suite level. New work must add zero new
failures.
