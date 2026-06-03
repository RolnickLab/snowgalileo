Working branch: ablations (https://github.com/marlens123/presto-v3/tree/ablations) 
- A single GeoTIFF consists of input products for a 1 km x 1 km ground area collected over a time series of 8 days.
- During export, all products are resampled to 10 m spatial resolution, so that each product will have H = 100 and W = 100 (https://github.com/marlens123/presto-v3/blob/9591fec0a91a9f0e061aedd17beea78e673b8fef/src/data/earthengine/eo_eval.py#L442). In reality, they won’t strictly have a shape of (100, 100) after export (but larger, with more distortion the further to the poles). We need to crop this to the right shape for model processing. For pre-training and inference, the model takes care of this automatically in the dataset class (https://github.com/marlens123/presto-v3/blob/9591fec0a91a9f0e061aedd17beea78e673b8fef/src/data/dataset.py#L448). For fine-tuning and evaluation, the inputs additionally need to be aligned with the labels (which are already in the right shape), we currently crop to the label bounds in a separate pre-processing step that needs to be executed separately (https://github.com/marlens123/presto-v3/blob/9591fec0a91a9f0e061aedd17beea78e673b8fef/scripts/developer_scripts/eval_crop_bounds.py#L15, more description https://github.com/marlens123/presto-v3/tree/main/data#evaluation-data)
- We divide the products into a) time-varying products, and b) static-in-time products. b) are ESA Worldcover and Copernicus DEM, a) is everything else. So every time-varying contributes [H * W * 8 timesteps * channels per product] and each static-in-time product contributes [H * W * channels per product] to the exported GeoTIFF (https://github.com/marlens123/presto-v3/blob/e456bfc877c351d02b7cf4f1f9424c373573f52b/src/data/earthengine/eo.py#L410).
- The exported GeoTIFF has a shape of [all_combined_bands, H, W]. all_combined_bands will consist of all time-varying products interleaved for all 8 timesteps, followed by the static-in-time products (https://github.com/marlens123/presto-v3/blob/e456bfc877c351d02b7cf4f1f9424c373573f52b/src/data/earthengine/eo.py#L410).
- This results in each exported GeoTIFF having 35 (all time-varying channels) * 8 (timesteps) + 3 (cloud info bands) * 8 timesteps + 4 (static-in-time channels) = 308 bands in total.
- The cloud info bands are currently used for analyzing the exported tiles, but they are not required as model input. So we don’t really need them for inference
- The structure of each GeoTIFF will be the same. During export, missing data (e.g. due to a lower revisit time) will be flagged as -9999 (https://github.com/marlens123/presto-v3/blob/9591fec0a91a9f0e061aedd17beea78e673b8fef/src/data/earthengine/utils.py#L32). This value is important because it identifies which data needs to be masked out during modeling.
- We additionally clip each product to a valid range, depending on what is physically plausible, or stated by earthengine documentation (https://github.com/marlens123/presto-v3/blob/977c38a73d521e4a0cb5a86610db98832707b157/src/data/config.py#L149). Since earthengine (I think) preprosses some data in their own way, these values might need to change with a new export platform!
- We use both Landsat 9 and Landsat 8 for a revisit time of 8 days.
- We mix the projections a little: during pre-training, we use Galileo’s approach and export all data in WGS84 (https://github.com/marlens123/presto-v3/blob/9591fec0a91a9f0e061aedd17beea78e673b8fef/src/data/earthengine/eo.py#L620), for fine-tuning, we use the original projection of our snow labels, which are in UTM (code link). For inference, we ideally want to follow the second approach I would say, but this is not implemented yet.
- Sidenote: We always use the Google Earth Engine URL mode for downloading (has some restrictions, but because it’s fast and for free).

## Bow Valley direct-source pipeline — conventions

- **DataFrames: use `pandas`, not `polars`.** The repo already depends on
  `pandas` (8+ modules) and uses zero `polars`. The GEE exporter boundary we must
  feed, `EarthEngineExporterEval.export_from_csv_utm`
  (`src/data/earthengine/eo_eval.py:576`), reads the cube CSV with `pd.read_csv`.
  Adding `polars` would introduce a new dependency purely for the Bow Valley grid
  generator with no benefit and a type-mismatch seam at exactly the contract
  boundary. Planning docs (PLAN/SPEC/TASK-00x) mention `polars` as a default
  preference; that preference is **overridden here** — pandas is the project
  standard for this pipeline. (Decided 2026-06-01.)
- **Generated cube CSV schema is fixed by the GEE exporter** — exactly
  `date, crs, center_x, center_y, min_x, min_y, max_x, max_y`, read column-by-column
  at `eo_eval.py:577-585`. The legacy `sampled_cells_bow_river_with_dates.csv`
  already uses this same 8-column schema; the grid generator reuses cell geometry
  (`center_x/y`, bounds, `crs=EPSG:32611`) and rewrites only the `date` column to
  the inference-window cross-product. The legacy `date` (all `20250515` /
  label-sampling metadata) is never read.
- **GEE export filename vs LocalSourceExporter filename differ (known, by design).**
  `export_from_csv_utm` emits `PR_{date}_{center_x:.16f}_{center_y:.16f}.tif`
  (3 fields, UTM coords) for the reference patches; the new `LocalSourceExporter`
  emits `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (5 fields, signed degrees). Both parse
  through the `PR` branch of `LandsatEvalDataset` (`src/fsc/landsat_eval.py:171-176`,
  month at `parts[1][4:6]`). Parity matching is by shared cube-CSV row, not filename
  string (SPEC AC-27). Filename ownership is resolved in TASK-004.

### Clip stage (Phase 0.5 / TASK-002)

- **The clip logic lives in the package, the CLIs are thin entrypoints.** The
  importable package is `src/data/local_sources/clip/` (`settings`, `gate`,
  `footprints`, `clippers`, `gdal_io`, `manifest`, `orchestrator`) — sibling to
  `grid.py`, since this is pipeline domain code, not a side script. The two Typer
  CLIs `scripts/developer_scripts/clip_dataset.py` (`clip-source`, `clip-all`,
  `--dry-run`) and `scripts/developer_scripts/clip_audit.py` only do argument
  parsing + `from src.data.local_sources.clip ...` imports (run via `uv run`, which
  uses the editable install). The old flat `scripts/developer_scripts/clip_dataset.py`
  + `scripts/.../test_clip_dataset.py` prototype was **removed** — it had no intersect
  gate, crashed (degenerate-size `assert`) instead of skipping non-overlapping tiles,
  and hardcoded a `min(1200,…)` MODIS clamp that truncated the 500 m science grid.
  Pytest tests live at `tests/test_clip_dataset.py`.
- **The §2.0 intersect gate is the one place footprint filtering happens.** Two
  stages: (1) metadata-only footprint∩AOI polygon test → `SKIP_NO_OVERLAP`;
  (2) overlap area < `CLIP_MIN_AOI_OVERLAP_AREA_KM2` (pydantic-settings,
  default 1 km²) **or** post-clip zero valid pixels → `SKIP_DEGENERATE_OVERLAP`.
  Skips write **no output file**. Adapters must NOT re-implement this. The real
  full-archive clip-all run (2026-06-02) produced **531 CLIP / 2 SKIP_NO_OVERLAP**
  (533 products; the 2 skips are the W120 WorldCover tiles west of lon −116.56),
  and the post-run audit passed.
- **Do not trust a `--dry-run` gate tally as proof of correctness.** The dry-run
  evaluates the *same* footprint readers the real run uses; a footprint reader
  that silently returns `None` makes the gate emit `SKIP_NO_OVERLAP`, which in a
  dry-run looks like a legitimate geographic skip. The first real run exposed
  three footprint/subdataset readers that were silently wrong (see next bullet) —
  the prior "531 CLIP / 2 SKIP dry-run" figure had masked them because nobody
  tallied the skips *per source*. Always sanity-check that an in-coverage
  modality is not skipping 100 %.
- **Three footprint/subdataset parsing bugs the first real clip-all surfaced**
  (all fixed + regression-tested, commit `735d92d8`):
  - **Sentinel-1** GML `<gml:coordinates>` is comma-within-pair
    (`"lat,lon lat,lon"`), not whitespace scalars. The parser `.split()` on
    whitespace yielded too few tokens → `None` → all 32 S1 wrongly skipped.
    Fix: normalise commas to spaces in `_parse_gml_coordinates`.
  - **Sentinel-3** stores the footprint in `<gml:posList>`, not
    `<gml:coordinates>` → all 125 S3 would skip. Fix: posList fallback regex.
  - **VIIRS** HDF5 subdataset descriptor `HDF5:"path"://group/.../band` was
    parsed with the MODIS HDF4 `:`-split, leaking quotes/slashes into the output
    filename and crashing `gdal_translate` on the first product. Fix:
    `gdal_io._parse_grid_band` splits the HDF5 group path on `/`, leaves the
    HDF4 `:`-form unchanged. MODIS output was unaffected (verified identical
    tokens), so MODIS did **not** need re-clipping.
- **MODIS/VIIRS clip output = per-grid GeoTIFFs, one per subdataset**, written to
  `<out>/modis/<granule_stem>/<GRID>__<band>.tif`, preserving native sinusoidal
  CRS+geotransform via `gdal_translate`. Both MODIS (HDF4) and VIIRS (HDF5)
  subdataset enumeration + extraction go through system `gdalinfo`/`gdal_translate`
  (`clip/gdal_io.py`), because rasterio's GDAL build lacks the HDF4 driver and the
  two descriptor dialects need format-aware grid/band parsing (`_parse_grid_band`):
  HDF4 is `…:"path":GRID:BAND` (`:`-delimited), HDF5 is `HDF5:"path"://…/GRID/…/BAND`
  (group path after `://`, `/`-delimited). The per-grid GeoTIFFs are each 1200² (1 km)
  / 2400² (500 m) at the native tile extent; cropping is by **AOI geometry**, see next
  bullet.
- **MODIS/VIIRS clip crops by AOI _geometry_, NOT a reprojected-corner index window
  (sinusoidal-shear trap).** `_clip_sinusoidal_subdataset` uses
  `rasterio.mask.mask(crop=True)` against the AOI reprojected into the subdataset's
  Sinusoidal CRS — identical in spirit to `_clip_geotiff_to`. The earlier approach
  (the one CLIPPING_PLAN §2.7 originally prescribed) reprojected the AOI's four
  lon/lat **corners** to sinusoidal and built an axis-aligned pixel window from their
  bbox. That is wrong: in MODIS Sinusoidal `x = R·λ·cos φ`, a lon/lat rectangle
  **shears** into a parallelogram, so its bounding window is ~5× too wide in X — the
  clip kept a ~10°-wide block of real data (100 % fill) instead of the AOI's ~2°-wide
  diagonal band. Geometry masking yields the correct band (~33.7 % fill over this
  AOI; nodata in the sheared corners). **Verify clip correctness by per-row valid-col
  span (≈ AOI width in km), never by the output bounding box** — the bbox of a
  diagonal band is legitimately ~10° wide even when the data is correct. Found by the
  clip-viewer (visual QA), not a unit test; the per-grid ratio test
  (`test_modis_per_grid_index_ratio`) only checks the 500 m grid is ~2× the 1 km grid,
  which both approaches satisfy, so it did **not** catch the shear. Re-clip after such
  a change with `clip-source modis` / `clip-source viirs` (~10 min, 92 products each),
  then rebuild the **combined** `clip_manifest.csv` by concatenating all 10 per-source
  manifests in `orchestrator.SOURCES` order — `clip-all --only modis,viirs` would
  truncate the combined manifest to just those two sources.
- **Landsat clips stay native EPSG:32612, S2 stays EPSG:32611.** The clip queries
  each band's CRS dynamically (no hardcoded zone) and reprojects the AOI to it. The
  cross-zone 32612→4326 reprojection is the Landsat adapter's job (TASK-012), not the
  clip stage. S1 measurement TIFFs are range-geometry (GCPs, no affine) → sliced by
  the AOI-overlapping GCP pixel window with shifted GCPs (defensive CRS+transform
  fast-path if a future pull ships orthorectified UTM).

