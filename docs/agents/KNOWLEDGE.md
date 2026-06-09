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
- **Per-cell target grid is EPSG:32611 (UTM 11N) @ 10 m, 100×100 — NOT 4326
  scale=10.** Empirically confirmed against real on-disk GEE cubes:
  `data/eval_tifs/LC09_*` (the inference/filename-style path our pipeline mimics)
  are **UTM (EPSG:32646/32638…), 100×100, 10 m resolution**. So `GridCell.crs="EPSG:32611"`,
  `shape=(100,100)`, `transform=from_origin(min_x, max_y, 10, 10)`.
  - **Two GEE export paths exist — do not conflate them (this was the planning bug).**
    The PLAN/SPEC "EPSG:4326 scale=10, ~159×100" is **real but describes the
    *training/label* export** (`export_for_labels`, filename `min_lat=…_dates=…`):
    verified `data/tifs_test/min_lat=…tif` is genuinely EPSG:4326, ~101×149,
    8.983e-05°. Our pipeline feeds the **inference** export
    (`export_from_csv_utm` → `_export_for_polygon`, which passes the CSV `crs`
    (`EPSG:32611`) + `scale=10` m into the EE download → UTM 11N, 100×100). The 4326
    grid is correct *for the training path*; it is wrong for the inference path.
  - **The loader's tensor-assembly path reads neither the tif CRS nor transform**
    — `_tif_to_array` uses only `data.values` (+ lat/lon from the **filename**), and
    crops to 100×100 (`subset_image`, requires H,W ≥ 100). The **prediction-write**
    path (`landsat_eval.py:1345-1346`) *does* read `data.rio.crs`/`.transform()`, but
    only to **copy them through** onto `*_with_preds.tif` — it asserts nothing about
    CRS, so UTM is safe and preserved. (Earlier "reads neither CRS nor transform"
    phrasing was imprecise: tensor path ignores them; write path copies them.)
  - **Filename lat/lon is a SEPARATE EPSG:4326 channel, independent of the UTM
    pixel grid.** `to_cartesian` (`dataset.py:256-257`) asserts `-90≤lat≤90` /
    `-180≤lon≤180` ("Make sure you are in EPSG:4326") and feeds `static_x`. So the
    filename MUST carry signed **degrees** even though the grid is UTM — the two are
    different consumers. Do not "align" the filename to UTM to match the grid CRS;
    `to_cartesian` would crash.
  - **Band-count trap:** `data/eval_tifs/*` are **316-band** cubes (pre-commit
    `5e2920f9 "remove viirs cloud flag"`). The **current** contract is **308**
    (`38 dynamic × 8 timesteps + 4 static`, `layout.TOTAL_BANDS`). Those on-disk
    files confirm the *grid* (UTM/100×100/10 m) but NOT the band layout — do **not**
    use them as current-layout fixtures; the loader's `num_timesteps` assert would fail.
  - Decided 2026-06-04 (user-ruled); re-validated against on-disk cubes same day.
- **DEM terrain (TASK-007): compute slope/aspect in the DEM's NATIVE frame, then
  resample to the UTM cell grid — two distinct CRS roles, do not collapse them.**
  `ee.Terrain.slope/aspect` (`copernicus_dem.py:14-16`) runs on the native GLO30
  DEM (latitude-aware true metric pixel spacing); `create_ee_image`'s export then
  resamples `[DEM,slope,aspect]` to the cell grid. The adapter must do the same:
  Horn kernel with latitude-correct metres-per-pixel **in the native frame**, then
  `reproject_to_cell` to **EPSG:32611** 100×100. Do NOT compute terrain in any
  projected grid (would diverge from `ee.Terrain`), and do NOT run the kernel on a
  raw degree grid with `1°≈1 m` spacing (gradients ×111,000 → all slopes ≈90°).
  **Caveat for readers of the older planning prose:** REVIEW_AUDIT #1 / earlier
  FR-15 / PLAN drafts justified "don't compute in UTM" with "the reference patches
  were never in a UTM frame" — that reasoning is **stale** (the cell grid IS UTM
  now). The *conclusion* (compute in native frame) stands; the *reason* is "because
  `ee.Terrain` is native," not "because the export is 4326." The terrain RESAMPLE
  target is UTM; the terrain COMPUTATION frame is native. (Flagged 2026-06-04.)

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
- **Sentinel-2 archive includes S2C, not just S2A/S2B (planning docs missed it).**
  Sentinel-2C became operational in 2025 and the Bow Valley archive carries S2C L1C
  products (e.g. the 2025-04-08 R113 `T11U**` granules integrated by TASK-013b). S2C
  shares the L1C SAFE structure (`manifest.safe`, `MTD_MSIL1C.xml`, `IMG_DATA/*.jp2`,
  N0511 baseline → −1000 DN), so the **clip stage handles it transparently** (purely
  structural, no satellite-token gating) and the **adapter regex is `S2[ABC]`**
  (`s2.py:_GRANULE_RE`, mirrored by the test's `_archive_acq_dates` regex). The unit
  letter is parity-irrelevant; gating on `S2[AB]` would have silently dropped every S2C
  granule. (Flagged + fixed 2026-06-08; earlier PLAN/SPEC/TASK prose says "S2A/S2B"
  only — read it as "S2A/S2B/S2C".)
- **The clip stage was lossy-recompressing every Sentinel-2 JP2 (FIXED 2026-06-08).**
  `_clip_geotiff_to` (`clip/clippers.py`) crops with `rasterio.mask.mask` (values preserved
  in memory) but wrote via GDAL's `JP2OpenJPEG` driver copying `src.profile`, which
  **defaults to LOSSY**. Raw SAFE JP2s are `COMPRESSION_REVERSIBILITY=LOSSLESS`; clipped
  ones came out `LOSSY` → reflectance corrupted ±~2 DN (B04: 12 % exact, max 12 DN over a
  patch) and the categorical `MSK_CLASSI` mask class-flipped (patch opaque 1→0). **Only S2
  is affected** — it is the only JP2 source; DEM/WorldCover/Landsat (DEFLATE/LZW/none) and
  S1/MODIS/VIIRS (uncompressed/NetCDF) are already lossless (audited 2026-06-08). Fix:
  `_clip_geotiff_to` forces `REVERSIBLE=YES, QUALITY=100` when the output is `.jp2`. After
  the fix, clipped B04 over the patch is **100 % bit-exact** to raw. **Re-clip S2** after
  this change (`clip-source sentinel2`) and rebuild the combined manifest additively.
  Guard: `test_clip_dataset.py::test_sentinel2_clip_is_lossless`. **TASK-013's "bit-exact B4
  parity" had been a false-green** — its assertion was *signed-median == 0*, which lossy
  ±2 DN noise preserves. See [[s2-clip-lossy-jp2-bug]].
- **Sentinel-2 `QA60` IS reconstructable — a deterministic MSK_CLASSI repack (TASK-013c,
  S2CloudAdapter).** GEE's `COPERNICUS/S2_HARMONIZED` rebuilds QA60 (post-2024-02-28) as
  `MSK_CLASSI_OPAQUE<<10 | MSK_CLASSI_CIRRUS<<11`, **opaque precedence, snow excluded** →
  value domain exactly `{0,1024,2048}`. **Verified by a direct GEE pull** (project
  `bow-valley-inference`, persistent creds): for `…_T11UNT` 2025-04-08 over PR_20250414 the
  GEE bands give QA60=1024, OPAQUE=1, CIRRUS=0 — and the **raw** SAFE MSK_CLASSI opaque
  matches (=1). The earlier "infeasible / different algorithm" conclusion was **WRONG**: it
  read the *lossy-corrupted* clipped mask (opaque flipped to 0). Once the clip is lossless,
  the repack reproduces GEE. (NOT a separate cloud algorithm; the SNAP-route idea does not
  apply — QA60 is an ESA L1C band, not a SNAP product.) `S2CloudAdapter` (`s2.py`) packs it
  with nearest 60 m→10 m reproject + the same coalesce/mosaic path as `S2Adapter`. See
  [[s2-qa60-reconstructed-from-msk-classi]].
- **Landsat clips stay native EPSG:32612, S2 stays EPSG:32611.** The clip queries
  each band's CRS dynamically (no hardcoded zone) and reprojects the AOI to it. The
  cross-zone 32612→4326 reprojection is the Landsat adapter's job (TASK-012), not the
  clip stage. S1 measurement TIFFs are range-geometry (GCPs, no affine) → sliced by
  the AOI-overlapping GCP pixel window with shifted GCPs (defensive CRS+transform
  fast-path if a future pull ships orthorectified UTM).
- **S1 adapter runs ESA SNAP once per (granule, cell) into a cached dB+angle GeoTIFF
  (TASK-014); the clip stage does NOT preprocess S1.** The clip is a range-geometry
  pixel *crop* (GCPs, no CRS), so the clipped scene's geographic extent is still the
  whole ~250 km swath. SNAP terrain-correction over the full AOI bbox (840 NPEs, 3.3 GB)
  OR the full clipped scene (1060 NPEs, 651 Mpx) both **NPE-corrupt** on swath-empty
  regions — only a small **per-cell `Subset` geoRegion** runs clean. So `s1_snap.py`
  (graph `s1_grd_graph.xml`) builds an offline per-cell cache (`s1_grd_<granule>_cell{id}.tif`,
  3-band: Sigma0_VH, Sigma0_VV linear, ellipsoid-incidence angle); `s1.py` reads it
  (pure raster, no SNAP), `10·log10` for VV/VH, angle passthrough, `< -30 dB` edge mask
  (VV/VH only). **dB is done in the adapter, not SNAP** — `LinearToFromdB` scoped to σ⁰
  drops the angle band; the cache stores linear σ⁰. **`angle` = ellipsoid incidence**
  (`saveIncidenceAngleFromEllipsoid`, matches reference patches ≤0.4°), NOT local
  incidence. **Band order pinned by index** (BigTIFF persists no descriptions). Parity
  proven on PR_20250519 (VV 0.38 / VH 0.40 dB, angle 0.24°); PR_20250406 is a
  GEE-pull-confirmed single-scene anomaly (GEE VV −2.63 vs our −12.7 dB, *same*
  acquisition — not a pipeline bug), PR_20250423 a SNAP "Empty region!" quirk. See
  [[s1-adapter-snap-cache-and-angle]], [[xarray-sentinel-s1c-regex-bug]]. Build:
  `python -m src.data.local_sources.s1_snap`.
- **S3 OLCI lat/lon are CF-scaled int32 — apply `scale_factor` before the AOI mask
  (or every radiance band clips to (0,0)).** `geo_coordinates.nc` stores `latitude`
  / `longitude` as `int32` with `scale_factor ≈ 1e-6` (raw `49896598` means
  `49.896598°`). The S3 clip masks the swath to the AOI bbox from these grids; the
  original code compared the **raw integers** against degree bounds, so the mask was
  always empty → `r0,r1,c0,c1 = 0,0,0,0` → every `Oa*_radiance.nc` sliced to `(0,0)`.
  The failure was **silent**: `_slice_s3_netcdf` counts valid pixels over *every* 2D
  dataset matching the geo grid, and non-grid datasets (`removed_pixels`,
  `instrument_data`) are full-copied, so the manifest still reported ~33 M "valid
  pixels" while the science bands held nothing. Fix: `_cf_scaled()` decodes
  `scale_factor`/`add_offset` before the mask; radiance now clips to the real swath
  window (e.g. `(734, 722)`, varying per overpass geometry). **Found by the
  clip-viewer Phase-4 probe**, not a test — same lesson as the sinusoidal shear:
  a units/projection mismatch that produces a wrong window is invisible to a
  valid-pixel-count gate when unrelated datasets pad the count. Re-clip with
  `clip-source sentinel3` (~125 products) and rebuild the combined manifest.

