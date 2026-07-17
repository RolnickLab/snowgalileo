# Hybrid Inference Pipeline — AOI Cubes via GEE + Local FSC Inference

How to produce daily fractional-snow-cover (FSC) COGs over an arbitrary AOI using
the **hybrid** pipeline: cubes are downloaded from Google Earth Engine (GEE), then
the finetuned model is run **locally** over them. This is the counterpart to the
fully-local `docs/local_data_processing.md` pipeline — use this one when you do not
have the raw sensor archive on disk but do have GEE access.

> **Two commands, one hand-off.** Step 1 downloads cubes; step 2 runs inference over
> them. They are split on purpose — download is slow, network-bound and GEE-throttled;
> inference is local, fast, and re-run constantly (new checkpoint, new threshold, mosaic
> fix). **The cubes on disk plus the cube CSV are the entire contract between the two.**

Scripts live in `scripts/developer_scripts/fortress_mountain_basin/`. Both are thin
Typer CLIs run with `uv run python …`.

## At a glance

| #   | Stage           | Script                       | Output                                                                   |
| --- | --------------- | ---------------------------- | ------------------------------------------------------------------------ |
| 1   | Build cubes     | `build_aoi_cubes_gee_url.py` | cube CSV + `data/<tifs-folder>/PR_<date>_<lat>_<lon>.tif_EPSG:32611.tif` |
| 2   | Daily FSC infer | `infer_aoi_cubes.py`         | one `fsc_YYYYMMDD.tif` per day (100 m px, EPSG:32611, COG, nodata −9999) |

```
build_aoi_cubes_gee_url.py  --aoi … --start-date … --end-date … --tifs-folder … --out-csv …
        │  cube CSV  (date, crs, center_lat, center_lon, min_x, min_y, max_x, max_y)
        │  cube dir  (data/<tifs-folder>/PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif)
        ▼
infer_aoi_cubes.py          --cube-csv … --cube-dir … --out-dir …
        │  fsc_YYYYMMDD.tif  (one stitched COG per day)
```

The cube CSV is the **single source of truth** for step 2: the grid, each cube's
filename, and the cell centre in true degrees are all derived from it. Inference never
re-tiles the AOI, so the mosaic grid cannot drift from the grid the cubes were built on.

## Requirements

### 1. Python environment

`uv` manages the project. All commands are `uv run python …` — no manual venv activation.

### 2. Model checkpoint (step 2 only)

Step 2 loads the finetuned `EncoderWithHead` named in `configs/bow_valley/inference.yaml`:

```
checkpoint: logging_checkpoints/snowgalileo_finetune/clouds_pretrained_42_lfdciemu.pth
```

The checkpoint is **required** — if it is absent the script fails loudly with
`FileNotFoundError` rather than silently initialising random weights (an all-random sweep
produces a plausible-looking but meaningless COG). Confirm it resolves before running:

```
ls -l logging_checkpoints/snowgalileo_finetune/clouds_pretrained_42_lfdciemu.pth
```

If it lives elsewhere, override rather than edit the YAML:
`INFER_CHECKPOINT=/path/to/model.pth uv run python …`.

### 3. Google Cloud + Earth Engine (step 1 only)

Step 1 calls `ee.Initialize(project=EE_PROJECT, …)`. Two things must be true:

- **An Earth-Engine-enabled GCP project.** Set via the `EE_PROJECT` env var; it defaults
  to `ee-marlena` (`src/snow_galileo/data/config.py:232`). The account you authenticate
  with must be **registered for Earth Engine** and have access to that project
  (https://console.cloud.google.com/earth-engine).

  ```
  export EE_PROJECT=your-ee-enabled-project     # skip to use the ee-marlena default
  ```

- **Credentials.** `get_ee_credentials()` (`src/snow_galileo/data/earthengine/utils.py:21`)
  picks one of two paths:

  1. **Service account** — if `GCP_SA_KEY` holds a service-account key JSON, it is used
     directly (best for headless/server runs).

  2. **Application Default Credentials (ADC)** — otherwise it falls back to the persistent
     `gcloud` credential. Set this up once, interactively:

     ```
     gcloud auth application-default login      # in a Claude Code session: ! gcloud auth application-default login
     gcloud config set project your-ee-enabled-project
     ```

  Sanity check auth + project before a full run:

  ```
  uv run python -c "import ee; ee.Initialize(); print(ee.Number(1).getInfo())"   # prints 1 when good
  ```

Downloaded cubes land under `DATA_FOLDER/<tifs-folder>`, where `DATA_FOLDER` is the
repo's `data/` directory (`config.py:244`). `--tifs-folder` is a folder **name** (it may
contain subdirs, e.g. `fortress_mountain_basin/cubes`), not an absolute path.

## Step 1 — build cubes

Tiles the AOI into a gapless 1 km EPSG:32611 lattice (the **Mode B** tiler), writes the
cube CSV, then downloads one 8-day, 308-band cube per `(cell, day)` in GEE `url` mode.

```
uv run python scripts/developer_scripts/fortress_mountain_basin/build_aoi_cubes_gee_url.py \
  --aoi data/fortress_mountain_basin_aoi.geojson \
  --start-date 2025-04-01 --end-date 2025-04-30 \
  --tifs-folder fortress_mountain_basin/cubes \
  --out-csv configs/aoi_cubes/cube_cells.csv
```

Key options:

| Option          | Default                            | Notes                                                               |
| --------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `--aoi`         | (required)                         | GeoJSON, single Polygon, any declared CRS (reprojected to lon/lat). |
| `--start-date`  | (required)                         | First window-end day, `YYYY-MM-DD`, inclusive.                      |
| `--end-date`    | (required)                         | Last window-end day, `YYYY-MM-DD`, inclusive.                       |
| `--tifs-folder` | `aoi_cubes`                        | Download folder **name** under `data/`.                             |
| `--out-csv`     | `configs/aoi_cubes/cube_cells.csv` | Where the cube CSV is written.                                      |
| `--mode`        | `url`                              | `url` (download) / `cloud` / `drive`. `url` is the hybrid default.  |
| `--max-workers` | `4`                                | Parallel download threads (`url` only). Keep low — GEE throttles.   |
| `--limit`       | `None`                             | Cap CSV rows for a smoke run.                                       |
| `--check-gcp`   | `False`                            | Skip already-exported tifs (needs a configured bucket).             |
| `--dry-run`     | `False`                            | Write the CSV only; do not touch Earth Engine.                      |

**Smoke first.** Verify the CSV before spending quota:

```
# CSV only, no download:
uv run python scripts/developer_scripts/fortress_mountain_basin/build_aoi_cubes_gee_url.py \
  --aoi data/fortress_mountain_basin_aoi.geojson \
  --start-date 2025-04-01 --end-date 2025-04-01 \
  --tifs-folder fortress_smoke --out-csv configs/aoi_cubes/fortress_smoke.csv --dry-run

# then one real cube:
… --start-date 2025-04-01 --end-date 2025-04-01 --tifs-folder fortress_smoke --limit 1
```

## Step 2 — run inference

Reads the cubes, runs the model batched, and stitches the per-cell FSC patches into one
daily COG with `DailyMosaicWriter`.

```
uv run python scripts/developer_scripts/fortress_mountain_basin/infer_aoi_cubes.py \
  --cube-csv configs/aoi_cubes/cube_cells.csv \
  --cube-dir data/fortress_mountain_basin/cubes \
  --out-dir data/fortress_mountain_basin/daily_fsc \
  --device cuda
```

Key options:

| Option         | Default                             | Notes                                                                                          |
| -------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------- |
| `--cube-csv`   | `configs/aoi_cubes/cube_cells.csv`  | The CSV step 1 wrote (single source of truth).                                                 |
| `--cube-dir`   | `data/aoi_cubes`                    | Where step 1 downloaded the cubes.                                                             |
| `--out-dir`    | `data/outputs/aoi_fsc`              | Daily FSC COG output dir.                                                                      |
| `--config`     | `configs/bow_valley/inference.yaml` | Checkpoint, eval config, batch size, device.                                                   |
| `--device`     | config value (`cuda`)               | Override, e.g. `cpu`. **No automatic CPU fallback** — set `cpu` explicitly on a GPU-less host. |
| `--batch-size` | config value (`16`)                 | Cells per forward pass; override for VRAM.                                                     |
| `--limit-days` | `None`                              | Process the first N days only (smoke run).                                                     |

**Smoke on CPU first** to isolate plumbing bugs from CUDA/VRAM ones:

```
… --device cpu --batch-size 5 --limit-days 1
```

Output: one `fsc_YYYYMMDD.tif` per day — a single-band float32 COG, EPSG:32611, 100 m px,
nodata −9999. For the Fortress AOI (25 cells over a 5×5 km area) each mosaic is 50×50 px.

## Worked example — Fortress Mountain Basin (verified run)

The pipeline was run end to end, both scripts back to back, over
`data/fortress_mountain_basin_aoi.geojson` for `2025-04-01 .. 2025-04-30`:

- **Cubes** (`data/fortress_mountain_basin/cubes/`): 25 cells × 30 days = **750** cubes,
  each 100×100 px, 308 bands, EPSG:32611.
- **FSC** (`data/fortress_mountain_basin/daily_fsc/`): **30** daily COGs, one per day,
  50×50 px. Spot check of `fsc_20250401.tif`: all 2500 px valid (no interior nodata),
  values in `[0.002, 1.000]`, mean ≈ 0.78 — in range and terrain-plausible for an April
  alpine basin.

## Things to know (gotchas)

1. **Cubes are always 100×100 (no export buffer).** An oversized cube is randomly cropped
   to 100×100 by the loader's **unseeded** `np.random.choice` offset
   (`dataset.subset_image`), so a 108×108 cube yields a prediction for a randomly shifted
   window — silently misregistered by up to 80 m, differently on every tile and every day.
   Step 1 pins the export buffer to 0 for this reason (the underlying exporter's 40 m
   default is overridden), and step 2 guards it: `_check_cube_shape` refuses to run on any
   cube that is not exactly 100×100.

2. **A missing cube is a hard failure, not a hole.** If any `(cell, day)` cube is absent
   from `--cube-dir`, step 2 aborts that run rather than writing a partial day. A mosaic
   with a nodata hole renders as a plausible COG that nothing downstream can tell is
   incomplete. Re-run step 1 for the missing date.

3. **Inputs are normalized — do not regress this.** The local inference path applies the
   same `Normalizer(std=True, …)` (from `configs/normalizing_dict.json`) that the GEE/eval
   path uses. An earlier version skipped normalization and fed the encoder raw physical
   units against a std-normalized checkpoint — every FSC output was invalid-but-plausible.
   The Fortress run above post-dates that fix. If you regenerate `normalizing_dict.json`
   for a new dataset, you silently mis-normalize this (and every) existing checkpoint — the
   dict is a frozen training-time constant, not fitted at inference.

4. **Filename mismatch is handled by read-only symlinks.** The GEE exporter names cubes
   `PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif` (doubled extension), which the
   loader cannot parse. Step 2 symlinks each cube (read-only, into a `TemporaryDirectory`
   deleted on exit) to a loader-parsable name whose lat/lon come **from the CSV**. No cube
   on disk is renamed or moved.

5. **GEE throttles `url` downloads.** `getDownloadURL` returns 429s under a large thread
   pool. `--max-workers 4` is the sweet spot; raising it costs, not saves, wall-clock.

6. **The seam guard is a bare `assert`.** `DailyMosaicWriter` asserts tiles are
   non-overlapping (`mosaic.py`). Do not run step 2 under `python -O` — `-O` strips asserts.

## Output QC — do not skip

Green plumbing does not mean a correct result. Before trusting a mosaic, check:

- **Values in `[0, 1]`.** Anything outside means the sigmoid/head config is wrong.
- **No `-9999` blocks in the interior.** A nodata block is a fully-masked cell — investigate,
  don't accept it.
- **Extent matches the AOI**, and the snow pattern is physically plausible against terrain
  (snow on the basin and upper slopes, less in the treed valley). A plausible-looking but
  *spatially shifted* pattern is the failure mode gotcha #1 warns about.
