# Bow Valley Archive Audit (Phase 0 — TASK-001)

*Formerly `ARCHIVE_AUDIT.md`.*

Catalog of `data/bow_valley_selection_raw`, the generated cube CSV, ingest-window
coverage, the S1 sparsity profile, and the static-layer (DEM/WorldCover) AOI
coverage assertion. Produced by TASK-001; consumed by every later task.

> **Method:** all figures below are measured from the on-disk archive
> (`gdalinfo`, filename date parsing, `pyproj`/`shapely` geometry), not copied
> from the planning docs. Where the planning docs and the archive disagree, the
> archive wins and the discrepancy is flagged.

---

## 1. Modality catalog

| Modality | Dir | Files | On-disk size | Format | Native CRS |
| --- | --- | ---: | ---: | --- | --- |
| DEM (Copernicus GLO-30) | `dem/` | **9** `*_DEM.tif` tiles | 632 MB | nested SAFE GeoTIFF | EPSG:4326 |
| ERA5-Land | `era5/` | 16 `.nc` (3 monthly folders) | 4.4 MB | NetCDF-4 | EPSG:4326 (0.1°) |
| Landsat 8 | `landsat8/` | 19 `.tar` | 24 GB | C2 L1 TOA GeoTIFF | **EPSG:32612** |
| Landsat 9 | `landsat9/` | 30 `.tar` | 36 GB | C2 L1 TOA GeoTIFF | **EPSG:32612** |
| MODIS MOD09GA | `modis/` | 94 `.hdf` | 12 GB | HDF4 sinusoidal (2 grids) | Sinusoidal |
| Sentinel-1 GRD | `sentinel1/` | 32 `.zip` | 53 GB | SAFE, **range geometry + GCPs** | none (GCPs, WGS84) |
| Sentinel-2 L1C | `sentinel2/` | 116 `.zip` | 75 GB | SAFE JP2 | EPSG:32611 |
| Sentinel-3 OLCI | `sentinel3/` | 125 `.zip` | 112 GB | SAFE NetCDF + tie-points | tie-point geolocation |
| VIIRS VNP09GA | `viirs/` | 94 `.h5` | 13 GB | HDF5 sinusoidal (2 grids) | Sinusoidal |
| WorldCover v200 | `worldcover/` | **4** `*_Map.tif` tiles | 377 MB | categorical GeoTIFF | EPSG:4326 |

**Total ≈ 326 GB.**

### ⚠️ Discrepancies vs planning docs (corrected in the docs)

The planning docs conflated **file counts** with **tile counts** for the static
layers. The load-bearing number for mosaic coverage is the elevation/Map tile
count:

- **DEM:** **9 `*_DEM.tif` elevation tiles** (`N50–N52 × W115–W117`, 1°×1°), in a
  nested SAFE layout with ~196 files total (KML/XML/PDF + auxiliary
  FLM/EDM/HEM/WBM rasters; 99 tifs, only 9 are elevation). The docs' "126 tiles"
  was a stale, wrong figure.
- **WorldCover:** **4 `*_Map.tif` tiles** (+ 4 `*_InputQuality.tif` companions =
  8 tifs). The docs' "8 tiles" was actually the *file* count; there are 4 Map
  tiles.
- **Corrected in** SPEC §3, CLIPPING_PLAN §1/§2.1, DATA_ANALYSIS §catalog/§spatial
  (this pass). The coverage assertion they guard still holds (§4), so no AC was
  ever blocked.

---

## 2. Generated cube CSV (FR-19b, AC-11b)

- **Path:** `configs/bow_valley/cube_cells.csv`
- **Schema (canonical, fixed by `eo_eval.py:577-585`):**
  `date, crs, center_x, center_y, min_x, min_y, max_x, max_y`
- **Rows:** **18 232** = 344 in-AOI cells × 53 window days (full cross-product).
- **CRS:** every row `EPSG:32611`.
- **Window:** default `2025-04-06 → 2025-05-28` (53 days inclusive); per-row
  `date` is the window-end day.
- **Cell filter manifest:** `configs/bow_valley/cell_filter_manifest.csv` —
  500 rows, **344 KEEP / 156 DROP** (centre-in rule).

### Cell AOI filter (AC-10) — verified counts

| Rule | Kept | Dropped | Sum |
| --- | ---: | ---: | ---: |
| centre-in (`data/bow_valley_inference_aoi.geojson`) | **344** | 156 | 500 |
| `--require-fully-inside` | **338** | 162 | 500 |

Reproduced from the real legacy CSV + AOI; matches PLAN §3.

---

## 3. Temporal coverage of the ingest window

Ingest window = `start − 7 → end` = **`2025-03-30 → 2025-05-28`** (the 7-day
prefill is non-negotiable; the model needs 8 timesteps for `d = start`).

| Modality | First acq. | Last acq. | # dates | Covers 03-30 → 05-28? |
| --- | --- | --- | ---: | --- |
| Landsat 8 | 2025-03-02 | 2025-05-28 | 9 | ✅ |
| Landsat 9 | 2025-03-01 | 2025-05-29 | 16 | ✅ |
| Sentinel-1 | 2025-03-30 | 2025-05-31 | **16** | ✅ (start-limiting) |
| Sentinel-2 | 2025-03-01 | 2025-05-30 | 37 | ✅ |
| Sentinel-3 | 2025-03-01 | 2025-05-31 | 84 | ✅ |
| MODIS | 2025-03-01 | 2025-05-31 | 92 | ✅ |
| VIIRS | 2025-03-01 | 2025-05-31 | 92 | ✅ |
| ERA5-Land | 2025-03 | 2025-05 | (monthly) | ✅ |

**Every modality covers the full ingest window.** S1's 2025-03-30 first
acquisition is what fixes the default inference start at 2025-04-06.

---

## 4. Static-layer AOI coverage assertion (AC-5)

AOI extent (`data/bow_valley_inference_aoi.geojson`, EPSG:4326):
`lon [-116.562, -114.528]`, `lat [50.730, 52.307]` (reaches `lat_max = 52.31`).

Mosaic extents measured via `gdalinfo` `wgs84Extent`:

| Layer | Tiles | Mosaic lon | Mosaic lat | Covers AOI to lat 52.31? |
| --- | ---: | --- | --- | --- |
| DEM | 9 | `[-117.0, -114.0]` | `[50.0, 53.0]` | ✅ |
| WorldCover | 4 | `[-120.0, -114.0]` | `[48.0, 54.0]` | ✅ |

**Both static mosaics fully contain the AOI**, including the northern edge at
lat 52.31. The corrected tile counts (9 DEM, 4 WC) are sufficient.

---

## 5. S1 sparsity profile (AC-4, dominant risk)

S1 acquisition dates (16 total): 03-30, 04-01, 04-06, 04-13, 04-18, 04-23,
04-25, 04-30, 05-05, 05-07, 05-12, 05-17, 05-19, 05-24, 05-29, 05-31.

**S1-fully-masked-window rate over the inference range** (`window = [d−7, d]`,
`d ∈ 2025-04-06 … 2025-05-28`):

| Inference days | Windows with **zero** S1 timestep | Windows with ≥1 S1 |
| ---: | ---: | ---: |
| 53 | **0 (0 %)** | 53 (100 %) |

**Result: no fully-masked-S1 window at the default start.** S1's 16 dates are
spaced ≤ 8 days across the inference span, so every 8-day window catches at least
one S1 acquisition. The dominant-risk scenario flagged in PLAN §3 / FDD §3 (many
windows with no S1 at all) **does not materialize** at the `2025-04-06` start —
it would only appear if the window were pushed earlier (toward the
`2025-03-08` earlier-start option) into the pre-03-30 S1 gap.

> Per-cell per-source 8-day-window *spatial* completeness (each scene footprint ∩
> each cell) is deferred to TASK-003, where `GridCell` footprints exist; the
> temporal S1 metric above is the load-bearing one for the compute go/no-go.

---

## 6. `PR` filename prefix (subtask 6 — RESOLVED)

`src/fsc/landsat_eval.py` parses filenames in two branches; the Landsat branch
accepts prefixes `LC | LE09 | LC08 | PR`. `data/eval_tifs` on disk contains only
`LC09` files — **`PR` is parser-supported but unused on disk**, so it is safe to
adopt as the synthetic/predicted direct-source prefix.

**Two filename forms coexist (by design):**
- GEE reference patches (`export_from_csv_utm`, `eo_eval.py:599`):
  `PR_{date}_{center_x:.16f}_{center_y:.16f}.tif` — 3 fields, UTM coords.
- `LocalSourceExporter` contract (FR-18): `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif`
  — 5 fields, signed degrees.

Both parse via the `PR` branch (`parts[1][4:6]` → month). Parity matching is by
shared cube-CSV row, not filename string (SPEC AC-27). Filename ownership is
resolved in TASK-004. (See `docs/agents/KNOWLEDGE.md`.)

---

## 7. GEE reference patches (subtask 7, AC-6) — ✅ DONE

`tests/fixtures/gee_reference_patches/` now holds **6 GEE reference patches**,
exported via `scripts/export_for_inference.py` →
`EarthEngineExporterEval.export_from_csv_utm` (url mode, direct
`getDownloadURL` → `requests.get`, no GCP bucket — `EE_BUCKET_TIFS=None`).

**Earth Engine setup (resolved this session):**
- Created GCP project **`bow-valley-inference`** (`gcloud projects create`), linked
  billing, enabled `earthengine.googleapis.com`, registered the project for EE
  (Console). Authed as `pelletier.f@gmail.com`.
- `EE_PROJECT` is now **env-overridable** (`src/data/config.py`, via `os.environ`
  + optional `.env`/`python-dotenv`) so EE calls bill `bow-valley-inference`
  instead of the hardcoded `ee-marlena`. `.env` is gitignored; see `.env.example`.

**Sample rows** (6, spread across the window and cells), from
`configs/bow_valley/cube_cells.csv`:

| date | center_x | center_y | month |
| --- | --- | --- | --- |
| 20250406 | 562863.85 | 5653083.79 | 04 |
| 20250414 | 540863.85 | 5745083.79 | 04 |
| 20250423 | 595863.85 | 5666083.79 | 04 |
| 20250502 | 556863.85 | 5622083.79 | 05 |
| 20250510 | 639863.85 | 5744083.79 | 05 |
| 20250519 | 625863.85 | 5760083.79 | 05 |

**Validation:** every patch is **308 bands** (35×8 dynamic + 3×8 cloud + 4 static,
matches `create_ee_image` / `KNOWLEDGE.md`), `EPSG:32611`, ~103×101 px (the
~100×100 cell, slightly larger pre-crop as expected). Filenames normalized to the
canonical GEE form `PR_{date}_{center_x}_{center_y}.tif` (3-field UTM), parsing
via the `PR` branch. Parity ACs (AC-14, AC-15, AC-21, AC-27) now have fixtures to
diff against; the adapter parity work (TASK-005+) is **unblocked**.

To regenerate or extend the set:

```bash
head -1 configs/bow_valley/cube_cells.csv > /tmp/ref_rows.csv
awk 'NR>1 && (NR-1)%3000==1' configs/bow_valley/cube_cells.csv | head -6 >> /tmp/ref_rows.csv
EE_PROJECT=bow-valley-inference uv run python scripts/export_for_inference.py \
    --mode url --tifs_folder gee_reference_patches --path_to_csv /tmp/ref_rows.csv
# then strip the "_EPSG:32611.tif" filename artifact and move into
# tests/fixtures/gee_reference_patches/
```

---

## 8. AC status

| AC | Description | Status |
| --- | --- | --- |
| AC-1 (SPEC AC-10) | 344 centre-in / 338 fully-inside; manifest sums to 500 | ✅ |
| AC-2 (SPEC AC-11) | grid cells pairwise non-overlapping | ✅ |
| AC-3 (SPEC AC-11b) | cube CSV schema + row count + EPSG:32611 + GEE column contract | ✅ |
| AC-4 (SPEC AC-30) | S1-fully-masked-window rate reported (= 0 %) | ✅ |
| AC-5 | DEM/WorldCover mosaics reach lat 52.31; window covered every modality | ✅ |
| AC-6 | 5–10 GEE reference patches | ✅ 6 patches, 308 bands each (§7) |
| AC-7 | ruff + mypy clean | ✅ |
| AC-8 | no new suite failures vs `TEST_BASELINE.md` | ✅ (see commit notes) |
