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
  at `eo_eval.py:577-585`. The legacy `tests/fixtures/sampled_cells_bow_river_with_dates.csv`
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
  CLIs `scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py`
  (`clip-source`, `clip-all`, `process-s1`, `process-all`, `--dry-run`) and
  `scripts/developer_scripts/bow_valley_inference_local/process_raw_audit.py` only do
  argument parsing + `from src.data.local_sources ...` imports (run via `uv run`, which
  uses the editable install). They were named `clip_dataset.py` / `clip_audit.py` until
  S1 gained a SNAP step (`process-s1`), making "clip" too narrow. The old flat `scripts/developer_scripts/bow_valley_inference_local/clip_dataset.py`
  - `scripts/.../test_clip_dataset.py` prototype was **removed** — it had no intersect
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
  ±2 DN noise preserves. See \[[s2-clip-lossy-jp2-bug]\].
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
  \[[s2-qa60-reconstructed-from-msk-classi]\].
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
  \[[s1-adapter-snap-cache-and-angle]\], \[[xarray-sentinel-s1c-regex-bug]\]. Build:
  `python -m src.data.local_sources.s1_snap`.
- **A truncated SNAP cache tif silently dropped S1 from cubes — guarded since
  commit `b90a8955`.** An *interrupted* offline build can exit 0 yet publish a tiny
  truncated sliver (~0.4 % of the AOI); the idempotent cache-hit check
  (`out_tif.exists() and not overwrite`) then **skips** it on the next build, so the
  sliver masquerades as a valid entry and every cube over the missing area gets an
  all-`-9999` S1 block. `s1_snap._output_extent_is_plausible` now runs **before the
  atomic publish**: it rejects an output whose UTM area is `< _MIN_EXTENT_RATIO`
  (0.25) of the expected AOI∩footprint, and rejects unreadable/corrupt outputs
  (`rasterio.errors.RasterioIOError`). A rejected output stays a `.partial` (retried
  next build), never a false cache hit. **This was NOT a path/manifest/border-noise/
  coverage bug** — those were all ruled out by A/B diagnostic. **If S1 looks missing
  again, run `scripts/developer_scripts/bow_valley_inference_local/spikes/verify_s1_cache.py` first** (per-granule extent-ratio +
  valid-pixel) — a sliver/truncated tif is the prime suspect; sparse-across-timesteps
  S1 is otherwise **EXPECTED** (S1 only on ~16 acquisition dates, and only on cells a
  granule footprint covers — a full-window scan of 7223 cubes shows valid S1 on
  exactly the 7 in-window acq dates, 0 everywhere else). Guard tests:
  `tests/test_local_sources/test_s1_snap_extent_guard.py`. See
  \[[s1-truncated-snap-cache-silent-dropout]\].
- **Inference driver/mosaic (TASK-015) mosaics FSC by DIRECT UTM placement — NO reproject.**
  The per-cell cube grid is already EPSG:32611 (not 4326), so each cell's 10×10 FSC
  prediction is already UTM 11N at 100 m/px. `DailyMosaicWriter` (`src/inference/mosaic.py`)
  places each patch into the daily AOI COG by **exact integer pixel offset** (block copy on
  the shared 100 m lattice), so placement is bit-identical — no interpolation, no invented
  FSC. The SPEC FR-22 / AC-29 / TASK-015 §2 wording "reproject each 10×10 FSC patch from
  EPSG:4326 with nearest-neighbour" is **stale** (predates the 2026-06-04 cell-grid CRS
  correction); the patch is not in 4326. Non-overlapping cells → disjoint blocks → a
  double-write assertion is the seam guard. All-masked cell (every loader valid-mask all-zero)
  → `None` → stays `nodata`; `aoi_coverage_fraction` is a COG tag. **Downstream is sacred:**
  the driver edits no `src/fsc/*` or `src/data/earthengine/*` — it injects a ready
  `EncoderWithHead` + `LocalSourceExporter` and drives the unchanged loader inference path
  through one read-only shim `src/inference/_loader_bridge.py` (the `__new__`-bypass
  tracer-test trick, isolated). The GEE `_predict_and_store_output` runner is untouched and
  runs in parallel. Checkpoint/model build + entry-point script + multiprocessing tuning are
  TASK-016.
- **Slow real-archive parity tests are serialized on one xdist worker — a parity "failure"
  in a full `-n auto` run may be I/O oversubscription, not a regression.** The suite runs
  `-n auto --dist loadgroup`; every `@pytest.mark.slow` test is paired with
  `@pytest.mark.xdist_group("slow_archive")` so the heavy GDAL-decoding real-archive tests
  (S2/Landsat parity, `test_clip_dataset`, S2 spike) run serialized on a single worker
  instead of competing 16-wide for disk/GDAL I/O. Before the group, a 7m51s full run
  false-failed `test_parity_b4_against_gee[PR_20250414/PR_20250423]` while they are
  deterministic (`PR_20250423` = 96.0 % bit-exact every run, > the 0.90 gate) and pass
  standalone / under `-n 4` / on a 3m49s full run. **xdist's `loadgroup` ignores a group
  added dynamically in a collection hook** — the marker must be static on each test
  (`tests/conftest.py` only documents the `SLOW_XDIST_GROUP` constant). **Triage rule:** a
  real-archive parity failure is only real if it reproduces **isolated** (`pytest <nodeid> -p no:xdist`); otherwise it is scheduling noise. See TEST_BASELINE.md.
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
- **S3 OLCI parity does NOT improve with SNAP orthorectification — do not re-attempt it
  (investigated and rejected 2026-06-09).** GEE terrain-orthorectifies OLCI in SNAP, so
  the §10 note flagged "SNAP ortho is the closer for the S3 residual" as a follow-up
  pairing with S1 (TASK-014). It was re-tested with SNAP's correct optical ortho path —
  `Reproject orthorectify=true` + SRTM 1Sec (the SAR `Terrain-Correction` /
  `Ellipsoid-Correction` ops reject an OLCI product) — and it went the **wrong direction**
  vs the production `scipy.griddata` swath-warp on the same patch/day/cell (10403 co-valid
  px): Oa17 corr 0.666→0.658, Oa21 0.783→0.774. The residual is **not** terrain distortion
  (else DEM ortho would have closed it) — it is **sampling geometry**: the patch is ~3 OLCI
  pixels wide (~300 m px on a ~1 km cell), so corr ~0.67 is a few edge pixels, and
  `spatial_kind="med"` 5×5-downsamples away any sub-pixel difference before the model sees
  it. Bonus blocker: SNAP's netCDF reader can't open the **clipped** `.nc` (HDF5 dim-scale
  refs — the same landmine the adapter avoids via `h5py`; `IllegalStateException: DataObject doesnt start with OHDR`), so ortho would force sourcing the raw product for zero gain.
  **Keep the swath-warp; the open S3 lever is the identity-normalization TODO, not
  geolocation.** Evidence kept: `src/data/local_sources/parity/s3.py` (logic),
  `scripts/developer_scripts/bow_valley_inference_local/spikes/run_s3_parity.py` +
  `s3_olci_ortho_graph.xml`. See PARITY_SPIKE_NOTES §10.1, \[[s3-snap-ortho-rejected]\].

### Cube cache (`cube_cache.py`) — invalidation & eviction

- **`CACHE_VERSION` versions the PROCESSING CODE, not the data — keep it a code
  constant, never config.** It is bumped *in the same diff* that changes any
  adapter `fetch`/clip logic; a cache dir whose `.cache_version` stamp differs is
  **force-cleared on construction** (so a known-incompatible cache can never be
  reused). A fresh dir (no stamp) is reconciled, never spuriously cleared. Putting
  it in `cube.yaml` would let one cluster's stale value silently reuse another's
  incompatible cache — exactly the disaster the stamp prevents. See
  \[[cube-cache-version-stamp-invalidation]\].
- **The `--cache-policy {prompt|reuse|overwrite}` flag backstops the forgot-to-bump
  case.** `prompt` (default) asks if the cache is non-empty and **errors on a
  non-TTY** (never silently reuses a possibly-stale cache in a batch job); `reuse`
  keeps it; `overwrite` clears once. **Clearing happens ONLY in the single parent
  process** (`resolve_cache_policy` before any worker spawns, or the `clean-cache`
  command) — a worker constructing with `overwrite=True` would wipe a sibling's
  fresh entries mid-run. Workers never clear. `export_bow_valley_cube.py` is now a
  **multi-command** Typer app: `export …` (was the bare script) **and**
  `clean-cache …`. `infer_bow_valley_daily_fsc.py` also takes `--cache-policy`.
- **Eviction is day-frontier, lazy, parent-only — not FIFO-eager.** `prune_before_day`
  is the **only** eviction path, called once per day in the parent before that day's
  pool spawns. It exploits the day-ordered sweep invariant (a cube for day D reads
  only `[D − window_days … D]`), so any entry with `day < D − window_days` is provably
  dead. **Lazy:** a no-op while `len ≤ max_entries` (Mode A is behaviour-identical to
  no eviction); only past the cap does it drop the dead frontier — giving a wide margin
  so a many-worker cluster never races to evict a still-live entry. It **never** evicts
  the live window; if still over cap after pruning (live window alone exceeds it) it
  logs `cube_cache_over_cap_after_prune` and returns. `cache_max_entries` is configurable
  (`cube.yaml`; `DEFAULT_MAX_ENTRIES = 200_000`, set to `3_000_000` for Mode B's full
  18 232-row sweep). The old `_evict_to_cap` FIFO path was **removed**. See
  \[[cube-cache-day-frontier-eviction]\].

## TASK-016 — downstream value-domain & inference invariants (AC-4)

These five are *preserved-as-is* contracts the direct-source pipeline must not "fix"
(they are model-numeric-domain or downstream-loader concerns, out of scope per SPEC §6):

- **MODIS `-28672` native fill is load-bearing — never strip or blend it.** The loader
  treats `-28672` as a "data present" sentinel (`landsat_eval.py:317,331` NDSI/NDVI);
  the MODIS adapter must preserve it in addition to `-9999`, and the nodata-aware
  resampler masks it to NaN before any bilinear so it never bleeds into a valid pixel.
  See \[[modis-fill-28672-load-bearing]\].

- **ERA5 temperature-shift sign is preserved, not corrected.** The known temp-sign quirk
  is a model-numeric-domain concern; the adapter emits raw Kelvin/native units and does
  **not** "fix" the sign (SPEC §6 out of scope). Separately, `total_precipitation` carries
  the ERA5-Land day-shift (day `i` ← `i+1` 00:00 accum slice); temps/winds are unshifted.
  See \[[era5-precip-accumulation-day-shift]\].

- **S3 identity-normalization is intentional.** The S3 OLCI radiances are passed through
  with identity normalization on purpose (downstream concern, out of scope to change); the
  open S3 lever is this norm TODO, **not** geolocation/ortho (\[[s3-snap-ortho-rejected]\]).

- **`PR` filename prefix is supported and currently unused on disk.** The loader's filename
  parser accepts the `PR_` prefix (`PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif`); the exporter emits
  exactly that. If the prefix meaning ever reopens, the **only** permitted downstream change
  is an additive allowlist patch at `landsat_eval.py:172` in the same PR (RESOLVED — no patch
  needed today).

- **Per-cell inference has NO cross-cell context.** Each cell is an independent
  `EncoderWithHead` forward on its own 308-band cube; the driver batches cells only for GPU
  throughput, never to share spatial context across cells. The daily mosaic stitches the
  independent 10×10 predictions by exact UTM pixel offset (\[[inference-driver-direct-utm-mosaic]\]).

- **Scene adapters (S2/Landsat) windowed-read the cell footprint, never the full UTM tile.**
  The clip keeps the full tile; reading a whole band is ~900 MB float64 → multi-GB OOM on a
  sweep. `_scene_ops.cell_window` reads only the cell neighbourhood (+4 px margin); output is
  bit-identical to the full read. See \[[s2-landsat-windowed-read-oom]\].
