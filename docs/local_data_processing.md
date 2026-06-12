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
├── bow_valley_inference_aoi.geojson                      # authoritative clip + inference boundary (EPSG:4326)
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
authority lives in `bow_valley_inference_aoi.geojson` — do not hardcode bounds elsewhere.

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

Crops every raw source to `bow_valley_inference_aoi.geojson`, non-destructively, into
`data/clipped_bow_valley_selection_raw/`. A two-stage intersect gate skips
products that miss the AOI (no output file). Per-source manifest records every
decision.

```bash
# Dry-run: gate only, no pixels decoded, no writes — sanity check first.
# NOTE: a dry-run can hide footprint-reader bugs (see Gotchas). Don't treat its
# CLIP/SKIP tally as proof of correctness.
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py clip-all --dry-run

# Real clip (all sources, serial).
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py clip-all

# ...or run sources in parallel (they are independent processes). Example:
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py clip-source sentinel2 &
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py clip-source sentinel3 &
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py clip-source viirs &
wait
# Parallel clip-source jobs each write only their own per-source manifest; the
# combined root manifest is NOT produced. Regenerate it (header once, then every
# per-source body) before running the audit:
{ head -1 data/clipped_bow_valley_selection_raw/dem/clip_manifest.csv; \
  for m in data/clipped_bow_valley_selection_raw/*/clip_manifest.csv; do \
    tail -n +2 "$m"; done; \
} > data/clipped_bow_valley_selection_raw/clip_manifest.csv

# Sentinel-1 is processed, NOT clipped — process it from RAW through ESA SNAP into the
# per-granule AOI-wide dB+angle cache (offline, hours; idempotent; raw granules are
# read-only). This is the SINGLE S1 product: both the cube S1Adapter AND the viewer's S1
# quicklook read it. (clip-all does not touch S1 — there is no raw-DN clipped S1.)
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py process-s1

# ...or do the whole raw → read-roots pipeline in one go (process-s1 FIRST, then clip-all
# of every other modality — the process-then-clip order):
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py process-all

# Post-run audit: zero all-nodata outputs, static mosaics reach lat 52.31, and the S1
# SNAP cache covers every AOI-overlapping raw granule.
# (Single-command Typer app — no subcommand. --root defaults to the clipped dir.)
uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_audit.py
```

**CLI flags** (both `clip-all` and `clip-source`): `--input-dir`
(default `data/bow_valley_selection_raw`), `--output-dir`
(default `data/clipped_bow_valley_selection_raw`), `--aoi`
(default `data/bow_valley_inference_aoi.geojson`), `--dry-run`. `clip-all` also takes `--only
a,b,c` to run a comma-separated subset **serially** (and it *does* write the
combined manifest for that subset — unlike separate `clip-source` jobs).

**Runtime.** Budget roughly 60–90 min for a full serial `clip-all` on this
archive; **Sentinel-2 alone is ~45 min** (JP2 decode per band) and is the long
pole — it is not hung. MODIS/VIIRS are fast per granule but emit thousands of
per-grid GeoTIFFs. Running the four heavy sources (S1/S2/S3/viirs) in parallel
roughly halves wall-clock.

**Code lives in the package**, not the scripts: `src/data/local_sources/clip/`
(`settings`, `gate`, `footprints`, `gdal_io`, `clippers`, `manifest`,
`orchestrator`). The two CLIs are thin entrypoints.

**Outputs:**
- `data/clipped_bow_valley_selection_raw/<source>/…` — clipped products, native
  CRS/format/pixels preserved.
- `data/clipped_bow_valley_selection_raw/<source>/clip_manifest.csv` +
  combined `clip_manifest.csv` at the root.

**Manifest schema** — one row per input product, columns:
`product_id, source, footprint_bbox, intersects, aoi_overlap_km2,
valid_pixel_count, action, output_path`. The `action` is one of `CLIP`,
`SKIP_NO_OVERLAP` (footprint disjoint from AOI), or `SKIP_DEGENERATE_OVERLAP`
(overlap below `CLIP_MIN_AOI_OVERLAP_AREA_KM2`, or post-clip zero valid pixels).
Skips have an empty `output_path` and no file on disk.

**Real run result (2026-06-02):** 533 products → **531 CLIP / 2 SKIP_NO_OVERLAP**
(the 2 skips are the W120 WorldCover tiles, west of the AOI). Audit passed; full
test suite at baseline (0 new failures). Per-source counts: dem 9, worldcover 2,
era5 15, landsat8 19, landsat9 29, modis 92, sentinel1 32, sentinel2 116,
sentinel3 125, viirs 92.

**Parallelism.** Sources are independent (separate input/output dirs and
manifests), so `clip-source <name>` jobs run in parallel safely. Sentinel-2 is
the long pole (JP2 decode per band); MODIS/VIIRS are CPU-light but emit many
files. There is no combined-manifest step when you run sources separately —
regenerate the root `clip_manifest.csv` by concatenating the per-source ones.

**Gotchas:**
- Intersect gate is the **one** place footprint-vs-AOI filtering happens.
  Adapters must not re-implement it. `CLIP_MIN_AOI_OVERLAP_AREA_KM2` (default
  1 km²) is env-overridable.
- **`--dry-run` proves the gate runs, not that footprints are read correctly.**
  A footprint reader that returns `None` shows up as a legitimate-looking
  `SKIP_NO_OVERLAP`. If an in-coverage modality skips ~100 %, suspect the reader,
  not the geography. (The first real run caught three such bugs — see below.)
  Diagnose by dumping one footprint and comparing its bounds to the AOI:
  ```python
  from pathlib import Path
  from src.data.local_sources.clip.settings import load_aoi_polygon
  from src.data.local_sources.clip import orchestrator as o
  aoi = load_aoi_polygon(Path("data/bow_valley_inference_aoi.geojson"))
  fn = o.MODALITIES["sentinel1"].gate_footprint          # the reader for that source
  fp = fn(sorted(Path("data/bow_valley_selection_raw/sentinel1").glob("*.zip"))[0])
  print(fp)                                              # None => reader bug, not geography
  print(fp.bounds, fp.intersects(aoi))                   # else compare to aoi.bounds
  ```
- MODIS/VIIRS output is **per-grid GeoTIFFs** (`<grid>__<band>.tif`), one per
  subdataset. The 500 m grid clips to ~2× the 1 km grid — indexed from each
  grid's own geotransform, never a hardcoded 1200 clamp.
- Landsat stays EPSG:32612, S2 stays EPSG:32611. Cross-zone reprojection to the
  cell grid is the adapter's job (TASK-012), not the clip stage.

**Footprint/subdataset parsing — modality quirks (fixed, regression-tested):**
- **Sentinel-1** manifest `<gml:coordinates>` is `"lat,lon lat,lon"` (comma
  within each pair); the parser normalises commas to whitespace.
- **Sentinel-3** footprint lives in `<gml:posList>`, not `<gml:coordinates>`.
- **VIIRS** HDF5 subdataset descriptor is `HDF5:"path"://…/GRID/…/BAND` (group
  path after `://`); grid/band are parsed by splitting on `/`, unlike the MODIS
  HDF4 `…:"path":GRID:BAND` (`:`-delimited) form. Both go through system GDAL.
---

## 4. Spatiotemporal alignment & 8-day cube assembly

Ensures spatial location and date matching are preserved from raw clipped sources when building 308-band assembled cubes.

### Location & Date Safeguards
- **Spatial Grid (`cube_cells.csv`):** Authoritative cell geometry defined in UTM 11N (`EPSG:32611`). Bounding boxes (`min_x`, `min_y`, `max_x`, `max_y`) map to target cell coordinates.
- **Date Target:** Row `date` defines sliding window end day $d$; sliding window spans 8 days $[d-7, d]$.
- **Filename Contract:** Assembled cubes written to `data/bow_valley_processing/cubes/` with standard format `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (signed decimal degrees of cell center and window-end date). Validated via `test_filename_contract.py` to match `LandsatEvalDataset` parser.

### Pipeline & Data-Flow Integration
1. **Clip Stage Footprint manifest (root `clip_manifest.csv`):** Keeps record of geographic bounds, overlaps, and pixel validity of clipped source archives on disk (schema in §3).
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
