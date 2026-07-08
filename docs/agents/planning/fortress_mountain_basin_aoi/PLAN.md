# Fortress Mountain Basin AOI — Cube Build via GEE URL Pipeline

Status: **code complete & validated; blocked on GEE/GCP credentials for the live download.**
Last updated: 2026-07-07

## Goal

Build the FSC input cubes over the Fortress Mountain Basin AOI by driving the original
GEE data pipeline in **URL mode** (download). Operator-facing entry point takes a GeoJSON
AOI and a date range — arguments only, no YAML config.

- AOI: `data/fortress_mountain_basin_aoi.geojson` (single Polygon, CRS `OGC:CRS84` = WGS84
  lon/lat; ~50.81–50.85 N, -115.25– -115.18 E).

## What was done

### 1. New operator script — `scripts/developer_scripts/build_aoi_cubes_gee_url.py`

Typer CLI, **arguments only** (modelled on `bow_valley_inference_local/infer_bow_valley_daily_fsc.py`
but with no config file). It:

1. Loads the AOI CRS-aware (`_load_aoi_geographic`): reads the GeoJSON's declared CRS and
   reprojects to EPSG:4326 if needed, so the tiler always receives lon/lat (CRS is law — a
   UTM AOI fed in as degrees would produce garbage tiles).
2. Tiles the AOI into a gapless 1 km **EPSG:32611** lattice via the canonical Mode-B tiler
   (`grid._tile_aoi_to_cells`) — the seamless-coverage prerequisite for later stitching.
3. Builds the cube CSV via `grid.build_cube_csv_for_gee_utm` (the reader dialect — see §2).
4. Writes the CSV, then (unless `--dry-run`) runs
   `EarthEngineExporterEval.export_from_csv_utm` in URL mode to download one 8-day cube per
   `(cell, day)`.

Key options: `--aoi`, `--start-date`, `--end-date` (YYYY-MM-DD, inclusive), `--tifs-folder`,
`--out-csv`, `--mode` (url/cloud/drive), `--inset-m`, `--limit` (smoke cap), `--check-gcp`,
`--dry-run`.

### 2. Fixed the `build_cube_dataframe` column-dialect bug

Two CSV dialects existed and mismatched:

- **Canonical UTM** (`grid.build_cube_dataframe`): `center_x`/`center_y` = eastings/northings.
  Round-trips with `load_cells` + fixtures + `CellGeometry`. **Left untouched** (its
  vocabulary is load-bearing downstream).
- **GEE UTM reader** (`eo_eval.export_from_csv_utm`): expects `center_lat`/`center_lon`
  (true decimal degrees), used only to name the output file
  `PR_{date}_{lat:.16f}_{lon:.16f}.tif`.

Feeding a canonical frame to the reader raised `KeyError`. Fixed with a Ports-and-Adapters
split (no rename of canonical columns):

- **`grid.py`**: added `GEE_UTM_CSV_COLUMNS` and `build_cube_csv_for_gee_utm()` — wraps
  `build_cube_dataframe` and reprojects the centre (EPSG:32611 → EPSG:4326) to emit
  `center_lat`/`center_lon` as **true degrees** (not mislabelled eastings — the loader
  parses these back into location bands, so they must be real lat/lon).
- **`eo_eval.py` `export_from_csv_utm`**: made the centre-column read tolerant of both
  dialects (`center_lat`/`center_lon`, else `center_x`/`center_y`, else `ValueError`),
  mirroring the existing `export_from_csv_wgs84` pattern. **Minimal touch**: existing local
  variable names (`center_x`/`center_y`) and the filename line were left unchanged — only
  the dataframe column-key lookups became flexible.

### 3. Tests — `tests/test_local_sources/test_cube_csv.py`

Fixed a test that had encoded the bug (it asserted the canonical CSV satisfied the reader's
columns and cited stale line numbers). Now:

- `test_gee_utm_schema_is_reader_dialect` — adapter emits exactly `GEE_UTM_CSV_COLUMNS`.
- `test_canonical_csv_lacks_reader_centre_columns` — regression guard: canonical CSV must
  NOT carry `center_lat`/`center_lon` (guards against a silent rename reintroducing the bug).
- `test_gee_exporter_column_contract` — repointed at the adapter.
- `test_gee_utm_centre_is_true_degrees` — `center_lat`/`center_lon` land in the AOI's
  geographic range (49–53 N, -117– -113 E), never UTM metre magnitudes.

### Validation done

- ruff + mypy: clean on all changed files.
- `tests/test_local_sources/test_cube_csv.py`: 10 passed (was 7).
- Broader suite (`grid`/`cube_csv`/`eo_eval`/`export`): 44 passed, 6 skipped (env-gated).
- Dry-run over Fortress AOI: 25 cells × 2 days = 50 rows, correct schema, centres in degrees.
- Live single-date attempt (2025-04-01): **CSV built correctly (25 rows), exporter
  constructed and invoked** — failed only on stale Google credentials (see §Remaining).

### Uncommitted changes (as of this writing)

```
 M src/snow_galileo/data/earthengine/eo_eval.py
 M src/snow_galileo/data/local_sources/grid.py
 M tests/test_local_sources/test_cube_csv.py
?? scripts/developer_scripts/build_aoi_cubes_gee_url.py
?? configs/aoi_cubes/            # generated CSVs (cube_cells / fortress_smoke)
```

Nothing has been committed — pending review.

## What remains to be done

### A. Provision GEE / GCP credentials on this machine (BLOCKER)

The live export failed with `RefreshError: invalid_grant: Bad Request`. Earth Engine
authenticates on this machine via **gcloud Application Default Credentials (ADC)**, not the
usual `~/.config/earthengine/` token:

- `~/.config/gcloud/application_default_credentials.json` exists but is **dated 2024-11-15**;
  its refresh token is expired/revoked.
- `~/.config/earthengine/` does **not** exist.
- No `GOOGLE_*` / `EARTHENGINE_*` env vars are set.
- `gcloud` (`/usr/bin/gcloud`) and `earthengine` (venv) CLIs are both present.

**Steps (interactive — a browser/OAuth flow, must be run by the operator):**

1. Refresh the ADC token EE actually uses:

   ```
   gcloud auth application-default login
   ```

   In this Claude Code session you can run it inline by typing:
   `! gcloud auth application-default login`

2. (If EE then complains about a missing/again-invalid token, initialise the EE-specific
   credential as well:)

   ```
   earthengine authenticate
   ```

3. Ensure a GCP project is associated (EE now requires one). Either:

   ```
   gcloud config set project <YOUR_EE_ENABLED_PROJECT>
   ```

   or confirm the project the exporter/`EarthEngineExporterEval` initialises with is
   Earth-Engine-enabled (https://console.cloud.google.com/earth-engine). The account must
   be **registered for Earth Engine** and have access to the target project.

4. Sanity check before a full run:

   ```
   earthengine authenticate --quiet   # or:
   uv run python -c "import ee; ee.Initialize(); print(ee.Number(1).getInfo())"
   ```

   A printed `1` means auth + project are good.

### B. Live smoke test (after A)

Run a **single cube** first (`--limit 1`) before the full AOI:

```
uv run python scripts/developer_scripts/build_aoi_cubes_gee_url.py \
  --aoi data/fortress_mountain_basin_aoi.geojson \
  --start-date 2025-04-01 --end-date 2025-04-01 \
  --tifs-folder fortress_smoke --out-csv configs/aoi_cubes/fortress_smoke.csv \
  --limit 1
```

Verify one `.tif` lands in the configured `DATA_FOLDER/fortress_smoke/`, then drop
`--limit 1` for the full single-date run (25 cubes), and finally the real date range.

### C. Downstream stitching seam (out of scope here, noted for later)

The GEE URL output is named `PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif`, which does
**not** match the inference driver's `build_cube_filename`
(`PR_{date}_{lat}_{lon}_SC00.tif`). A future stitcher must parse these filenames (or map by
the CSV's UTM bounds / `cell_id`) itself. Cube download is unaffected.

### D. Commit

Once the live smoke passes, commit the change set (per review discipline — one commit per
logical unit; the script + grid adapter + eo_eval tolerance + tests are one coherent unit,
or split if preferred).
