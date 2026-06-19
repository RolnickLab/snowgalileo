# Local Data Processing — Bow Valley Direct-Source Pipeline

How to recreate the Bow Valley inference pipeline end to end — from the raw
download archive to daily fractional-snow-cover (FSC) COGs — without Earth
Engine. Run the stages in order. Each task appends its own section here.

> **Living doc.** Future tasks (TASK-003+) add their steps below. Keep it
> concise: commands, paths, and gotchas — not narrative.

## Stage order at a glance

Every operator script lives in
`scripts/developer_scripts/bow_valley_inference_local/`; each is a thin Typer CLI
over package code, run with `uv run python …` (the viewer with `uv run solara
run …`). Run the stages top to bottom:

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 0 | Grid + reference patches (§2) | `python -m src.data.local_sources.grid --emit-csv` | `configs/bow_valley/cube_cells.csv`, parity fixtures |
| 1 | Process raw → read roots (§3) | `process_raw_dataset.py process-all` | clipped archive + `sentinel1_snap/` cache |
| 1a | (S1 only, standalone) | `process_raw_dataset.py process-s1` **or** `build_bow_valley_s1_cache.py` | `sentinel1_snap/s1_grd_<granule>.tif` |
| 1b | Audit stage 1 | `process_raw_audit.py` | exit 0 = clean |
| 2 | Assemble 308-band cubes (§5) | `export_bow_valley_cube.py` | `processing_root/cubes/PR_*.tif` |
| 3 | Daily FSC inference (§6) | `infer_bow_valley_daily_fsc.py` | `processing_root/daily_fsc/*.tif` |
| 4 | Inspect / QA (§7) | `solara run data_viewer.py` | Clip / Cube / Daily-FSC tabs |

**Key ordering rule (stage 1):** Sentinel-1 is **processed, never clipped** — it
must go through ESA SNAP *before* anything reads it. `process-all` enforces this:
it runs `process-s1` **first** (raw S1 → SNAP cache), then `clip-all` for every
other modality. The single S1 product (`sentinel1_snap/`) is read by both the
cube `S1Adapter` and the viewer. There is no raw-DN clipped-S1 product.

---

## 0. Prerequisites

- **Data is downloaded manually (for now).** No automated fetch yet. You must
  place the raw archive at `data/bow_valley_selection_raw/` yourself, one
  subdirectory per source, before any step runs. Sources, formats, and counts
  are cataloged in `docs/agents/planning/bow_valley/000-analysis/002-archive-audit.md`. The
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
- `docs/agents/planning/bow_valley/000-analysis/002-archive-audit.md` — full archive catalog.
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

# Verify the SNAP cache before assembling cubes. An INTERRUPTED build can publish a
# truncated sliver tif that the idempotent cache-hit check then refuses to overwrite,
# silently dropping S1 from every cube over the missing area. The build now guards
# against this (extent check before atomic publish), but verify regardless:
uv run python scripts/spikes/verify_s1_cache.py   # per-granule extent-ratio + valid-pixel

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
     - Sentinel-1: reads the per-granule SNAP `sentinel1_snap/` cache
       (EPSG:32611, terrain-corrected dB+angle), windowed per cell — **not** a
       raw GCP slice (that path was removed; see §3)
     - MODIS/VIIRS: Sinusoidal projection (`+proj=sinu +R=6371007.181`)
   - Resampled to $100 \times 100$ pixel grid via robust resampler in `base.py`.
   - Coalesces overlapping orbit/scene pixels (first valid wins) to prevent false `-9999` nodata.
4. **Intermediate Cache Sharding:** Per-day per-cell arrays cached under `data/bow_valley_processing/cube_cache/{cell_id}/{day}_{modality}.npz`. Avoids 8x storage duplicate of sliding windows and filesystem indexing limits.
   - **Invalidation:** a `.cache_version` stamp at the root holds `cube_cache.CACHE_VERSION`. On construction, a dir whose stamp differs is **force-cleared** (a code change to any adapter `fetch`/clip logic must bump `CACHE_VERSION` in the same diff). The `--cache-policy` flag (§5) is the manual backstop for the forgot-to-bump case. **All clearing happens only in the parent process**, never in a worker (concurrent clears would race).
   - **Eviction (Mode B scale):** `prune_before_day` runs once per day in the parent before the pool spawns. It is **lazy** (a no-op while entries ≤ `cache_max_entries`, so Mode A behaves as if uncapped) and **day-frontier** — past the cap it drops only entries provably dead under the sliding window (`day < current_day − 7`), never the live window. `cache_max_entries` is set in `cube.yaml` (default 200 000; 3 000 000 for the full Mode B sweep).
5. **Cube Assembly (`LocalSourceExporter`):** Composites 8 daily cached arrays in exact canonical band order (308 bands).

---

## 5. Stage 2 — assemble the cubes (`export_bow_valley_cube.py`)

Builds the in-AOI grid and writes one canonical **308-band** cube tif per
`(cell, window_end)` into `processing_root/cubes/`, using the **real-adapter**
exporter (`LocalSourceExporter`, `placeholder=False`) over the read roots from
stage 1. Additive: composes `build_grid` + the parallel exporter, touches no
GEE-path code.

> **Invocation note:** this script is now a **multi-command** Typer app — the cube
> build is the `export` subcommand (it used to be the bare script). There is also a
> `clean-cache` subcommand (below).

```bash
# Smoke run — 4 cells, default window from cube.yaml.
uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \
    export --config configs/bow_valley/cube.yaml --limit 4

# Full sweep (all 344 in-AOI cells), explicit window-end, 8 workers.
uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \
    export --config configs/bow_valley/cube.yaml --window-end 2025-05-28 --workers 8

# Wipe the cube cache on demand (reports entries removed) — for the "did my
# clips/adapters change?" case the version stamp can't catch.
uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \
    clean-cache --config configs/bow_valley/cube.yaml
```

**`export` flags:** `--config` (default `configs/bow_valley/cube.yaml`), `--limit`
(cap cells for a smoke run; `None` = all), `--window-end` (`YYYY-MM-DD`; default
from `cube.yaml`), `--workers` (default ~8, clamped to cores/cells),
`--verify-s1-cache` / `--no-verify-s1-cache`, `--cache-policy`
(`prompt` | `reuse` | `overwrite`, default `prompt`).

**Cube-cache policy (`--cache-policy`).** The per-(modality, cell, day) `cube_cache`
is reused across runs. `prompt` (default) asks before reusing a non-empty cache and
**errors on a non-TTY** — so a batch/SLURM run never silently builds cubes from a
possibly-stale cache. `reuse` keeps it without asking; `overwrite` clears it **once,
in this parent process**, before any worker spawns (workers never clear — that would
race). Use `overwrite` after an adapter or clip change that the `CACHE_VERSION` stamp
didn't catch (a version bump force-clears automatically). A code change to any adapter
`fetch`/clip logic should bump `cube_cache.CACHE_VERSION` in the same diff; the stamp
then invalidates stale caches everywhere on next construction.

**S1 pre-flight (on by default).** `--verify-s1-cache` checks the per-granule
SNAP cache covers each cell's window and **fails loud** if a needed
`sentinel1_snap/s1_grd_*.tif` is missing — the cube exporter never runs SNAP
inline (it is the heavy offline stage 1a). Pass `--no-verify-s1-cache` only to
deliberately produce S1-free cubes.

**Outputs:** `processing_root/cubes/PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (filename
contract in §4); intermediate per-day/per-cell arrays cached under
`processing_root/cube_cache/` (§4 item 4).

---

## 6. Stage 3 — daily FSC inference (`infer_bow_valley_daily_fsc.py`)

Runs the pretrained model over the sweep and writes one daily fractional-snow-
cover **COG per inference day** into `processing_root/daily_fsc/`. Reads
`cube.yaml` (sweep window/mode/roots) **and** `inference.yaml` (checkpoint, eval
config, batch, device); builds `EncoderWithHead` via the **same** load path as
`scripts/eval_only.py` (`Encoder(**enc_cfg)` → `EncoderWithHead` →
`load_state_dict`), then drives `InferenceGridDriver`.

```bash
# Smoke run — 4 cells.
uv run python scripts/developer_scripts/bow_valley_inference_local/infer_bow_valley_daily_fsc.py \
    --cube-config configs/bow_valley/cube.yaml \
    --config configs/bow_valley/inference.yaml --limit 4
```

**Flags:** `--cube-config` (default `configs/bow_valley/cube.yaml`), `--config`
(default `configs/bow_valley/inference.yaml`), `--limit`, `--cache-policy`
(`prompt` | `reuse` | `overwrite`, default `prompt`; same cube-cache semantics as §5).

**Checkpoint is required — fails loud if absent.** A missing checkpoint aborts
the run rather than silently initializing random weights (a random sweep would
emit a plausible-looking but meaningless COG). No downstream/GEE code is touched.

**Outputs:** one daily FSC COG per inference day in
`processing_root/daily_fsc/` (the value domain is 0–1).

---

## 6.5. Full Mode-B AOI sweep (`cube_full_run.yaml` / `inference_full_run.yaml`)

The Mode-A run (§5–§6, `cube.yaml`) sweeps only the ~344 in-AOI **sample** cells.
The full run tiles the **entire AOI** into a 1 km lattice (mode B) and runs the
same two stages over it, writing into an **isolated** processing root so the
Mode-A deliverables are never touched.

**Config pair (sibling to the Mode-A configs):**

| File | Differences from Mode-A config |
|------|--------------------------------|
| `configs/bow_valley/cube_full_run.yaml` | `mode: B`, `mode_b_inset_m: 5000`, `processing_root: data/bow_valley_processing/full_run` |
| `configs/bow_valley/inference_full_run.yaml` | `export_workers: 16` (vs 8); everything else identical |

**Grid: 5 km negative buffer.** `mode_b_inset_m: 5000` erodes the AOI inward by
5 km (a `buffer(-5000)` in UTM 11N, `grid.py:_tile_aoi_to_cells`) **before** tiling,
dropping the outer 5 km ring of output cells: **25,078 → 21,985 cells (−12%)**. This
removes edge cells whose 8-day source windows may have degraded coverage near the AOI
boundary. It does **not** add a data margin around retained cells — every cell still
reads whatever clipped source overlaps it. The buffer is a *code capability* defaulting
to `0.0`; it is active **only** because this config sets it.

**Isolated output root.** `data/bow_valley_processing` is a symlink to
`/archive/data/ai4snow/bow_valley_processing` (7.8 TB free), so `full_run/` resolves
to `/archive/data/ai4snow/bow_valley_processing/full_run/`. The `cube_cache/`, `cubes/`,
`daily_fsc/`, `manifests/`, and `scratch/` subdirs all derive from this root — created
on first write, never overlapping the Mode-A tree.

```bash
# Stage 2 — assemble the full-AOI cubes (21,985 cells × 21 days = 461,685 cubes).
# 'reuse' is wrong here (fresh isolated cache); 'overwrite' clears the empty full_run
# cache up front and runs non-interactively (safe for a long batch run).
uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \
    export --config configs/bow_valley/cube_full_run.yaml --workers 16 --cache-policy overwrite

# Stage 3 — daily FSC inference over the full grid (writes one COG per inference day).
uv run python scripts/developer_scripts/bow_valley_inference_local/infer_bow_valley_daily_fsc.py \
    --cube-config configs/bow_valley/cube_full_run.yaml \
    --config configs/bow_valley/inference_full_run.yaml --cache-policy overwrite
```

> The inference driver re-exports each day's cubes into its own cache as it runs, so
> stage 3 can be run on its own (it does not require stage 2 to have run first). Running
> stage 2 separately is only useful to materialise/QA the cubes before inference.

**Scale & runtime (21-day window, 5 km inset = 21,985 cells, 16 workers):**

| Resource | Mode A (measured) | Full Mode B (projected) |
|----------|-------------------|--------------------------|
| Cells | 344 | 21,985 |
| Cube exports | 7,224 | 461,685 |
| Wall-clock | 31 min (8 workers) | **~33 h measured** (14–16 workers; see breakdown below) |
| `cubes/` on disk (~12.3 MB each) | ~88 GB | **~5.7 TB** (stays on `/archive`) |
| `cube_cache/` (~130 KB/entry) | ~few GB | **~250–390 GB** (on SSD — see below) |
| `daily_fsc/` COGs | 21 × ~0.35 MB | 21 × a few MB (on SSD) |

The wall-clock is **export-bound and CPU-bound** (not I/O- or GPU-bound). Day 1 measured
**~1.2 cubes/s**; profiling one cold cube export (`cProfile`, cache off) shows the per-cube
CPU split:

| Source | Share | Cost |
|--------|-------|------|
| **Sentinel-2** | **~42%** | JP2 decode (`rasterio.open`, ~60 opens/cube) — single-threaded, heavy |
| **ERA5** | ~20% | NetCDF `open_dataset` (h5netcdf, ~40 opens/cube) |
| **Sentinel-3** | ~19% | `scipy.griddata` swath warp (irreducible per the S3 design) |
| Landsat / MODIS / VIIRS | ~19% | cheap (MODIS/VIIRS sinusoidal warp is NOT a hot spot) |

Both disks sit **<15% util** during the sweep (all 16 cores pegged, load avg ~32, zero
workers in I/O-wait) — the export bottleneck is **CPU**, specifically JP2 decode + repeated
file opens, not disk.

The sweep has **two sequential stages per day**, and each dominates at a different point:

1. **Export** (CPU-bound). **Day 1 is fully cold (~5 h)**; the `cube_cache` then amortises
   the repeated opens — **day 2 measured ~9.4 cubes/s, ~8× the cold ~1.2** (it reuses 7/8 of
   the sliding window), so days 2–21 export in **~35–40 min each**.
2. **Inference + mosaic** (GPU forward over all 21,985 cells + assembling the AOI-wide
   1360×1690 COG). Measured **~50 min/day** and **constant** — it does NOT benefit from the
   cube cache, so once export goes warm this stage *dominates*: ~50 min × 21 ≈ **~18 h**.

Net end-to-end ≈ **~33 h** (≈ 5 h cold day-1 export + ~20 × ~38 min warm export + ~21 ×
~50 min inference, stages serial per day). The earlier flat "~50 h" (extrapolating the cold
day-1 export rate) and "~20–30 h" (ignoring the per-day inference stage) were both wrong —
day 1 is the slow export, but **inference+mosaic is the steady-state floor**.

**Worker count is a CPU lever.** 16 workers on a 16-core host slightly oversubscribe
(parent + GPU thread + per-worker rasterio/h5py helpers → ~14% system time, 20k
ctx-switch/s); `inference_full_run.yaml` sets `export_workers: 14` to leave the
coordinator headroom. There is **no I/O lever** — the SSD split below is correct hygiene
(keeps small-file churn off the HDD, frees `/archive` seeks for cube writes) but does not
change throughput, because I/O was never the limiter.

**I/O split across two drives (hygiene, not a speedup).** `cubes/` (~5.7 TB) must stay on
`/archive` (only it has the room). The `cube_cache/` is millions of tiny `.npz` files; put
them (and the trivially-small `daily_fsc/`) on a separate SSD so that small-file churn
leaves the HDD. NOTE: **moving an existing cache HDD→SSD is not worth it** — 1.9 M tiny
files copy at ~50/file/s (~12 h, HDD-random-read-bound) regardless of `mv`/`tar`; start a
fresh cache on the SSD instead (`--cache-policy overwrite`).

```bash
# data/bow_valley_fast_processing -> /storage/data/bow_valley_fast_processing (SSD).
SSD=data/bow_valley_fast_processing/full_run ; RUN=data/bow_valley_processing/full_run
mkdir -p "$SSD/cube_cache" "$SSD/daily_fsc"
# If a cache already exists, MOVE it (don't recreate empty — preserves the warm window):
mv "$RUN/cube_cache" "$SSD/cube_cache" 2>/dev/null || true
ln -s "$(readlink -f "$SSD/cube_cache")" "$RUN/cube_cache"
ln -s "$(readlink -f "$SSD/daily_fsc")"  "$RUN/daily_fsc"
```

The source reads + cube writes stay HDD-bound, so this is a partial (not 3×) speedup —
it removes the cache-thrash contention, the part the SSD actually fixes.

**Gotchas:**
- **Run in the background / detached.** A multi-day sweep must survive disconnects —
  run under `nohup`/`tmux`, tee stdout into `full_run/scratch/`, and pass an explicit
  `--cache-policy` (never the `prompt` default — it errors on a non-TTY by design, §5).
  Use `overwrite` for a clean start; use **`reuse`** to **resume** on top of a warm
  cache without re-fetching (e.g. after a crash — `export()` rewrites the day's cubes
  but every modality fetch is a cache hit, so the resume is cache-fast).
- **Edge-cell guard (Mode B only).** AOI-edge tiles can produce a sub-pixel scene read
  window that rounds to a 0-px axis; before the `cell_window`/`reproject_to_cell` guards
  (`_scene_ops.py`, `base.py`) this raised `CPLE_AppDefinedError: Invalid dataset
  dimensions : 0 x N` inside a worker and killed the whole pool. Interior cells (Mode A,
  small smoke slices) never hit it — smoke-test the **tail** cells (`grid[-300:]`) when
  validating a Mode-B change, not just `--limit N` (which takes the head).
- **`cache_max_entries: 3000000`** is sized for the Mode-B live window (~1.58 M) plus
  the day-boundary transient (~1.78 M). At ~130 KB/entry that is ~250–390 GB on disk —
  fits the SSD with headroom. Do not lower it for this run or the over-cap prune fires
  inside the live window.
- The 5 km inset follows the AOI's true shape; concave notches narrower than 10 km may
  close. Verify the kept-cell extent in the viewer (§7) before trusting the output AOI.

---

## 7. Inspect / QA — the data viewer (`data_viewer.py`)

A developer/QA Solara app; each tab is a leafmap map with the AOI outline
overlaid. Read-only on the archive and `processing_root`; writes only transient
decimated GeoTIFFs to a temp dir.

```bash
uv run solara run scripts/developer_scripts/bow_valley_inference_local/data_viewer.py
```

Three tabs, one per pipeline output:

- **Clip** — pick a clipped product from the manifest; see its quicklook on a
  basemap plus clip-stage metadata (overlap km², valid-pixel count, action). S1
  here renders from the `sentinel1_snap/` cache, not the clip manifest.
- **Cube** — inspect an assembled per-cell cube (stage 2): pick prediction date →
  cell → variable, then step the timestep slider. Variable/timestep selection is
  **availability-filtered** (only var/timestep combinations that actually exist
  in that cube are selectable).
- **Daily FSC** — step a date slider through the daily FSC COGs (stage 3); the
  selected day renders colormapped (0–1) on the map.

See `docs/agents/planning/bow_valley/060-viewer/060-viewer-plan.md`, `CONTRACT.md`, and
`PLAN-V2-CUBE-FSC-TABS.md`.

---

## Testing baseline

The suite is **already red on a clean checkout** (6 pre-existing failures, see
`docs/agents/planning/bow_valley/020-data-ingestion/tasks/test-baseline.md`). Judge work by
**delta** — never `pytest -x` at the suite level. New work must add zero new
failures.
