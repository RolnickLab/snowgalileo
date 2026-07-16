# Fortress Mountain Basin AOI — Cube Build + FSC Inference via GEE URL Pipeline

Status: **Step 1 (cube build) done & committed — 75/75 tiles, seamless, buffer 0.
Step 2 (inference) written, lints/typechecks clean, NOT yet run — blocked on a model
checkpoint (none on this workstation). Resume on the compute server.**

> **⚠️ 2026-07-13 — the inference path never normalized its inputs (§6, now FIXED).** Every
> FSC output the local direct-source pipeline has produced to date is **invalid** and must be
> regenerated — Bow Valley Mode A, the ~33 h full Mode-B sweep, and anything Fortress. The
> encoder was fed raw physical units against a std-normalized checkpoint. Two follow-ons:
> `InferenceGridDriver` **will** be reused after all (§5 decision reversed), and no test built
> on placeholder cubes can ever catch a normalization bug (§6).
>
> **↳ 2026-07-13 (update) — Bow Valley Mode A is now regenerated (§7).** New normalized run in
> `data/normalized_daily_fsc/`; old invalid run preserved at `data/bow_valley_processing/daily_fsc/`.
> Old-vs-new diverges materially (MAD 0.24, corr 0.70), worst at window start — the expected
> signature. Still to regenerate: the ~33 h Mode-B sweep and Fortress. QC-on-own-merits (§E.4)
> still pending for the new Mode A mosaic.

Last updated: 2026-07-13

## Goal

Two operator commands over the Fortress Mountain Basin AOI:

1. **Build cubes** — drive the original GEE data pipeline in **URL mode** (download) to
   produce one 8-day, 308-band cube per `(cell, day)`.
2. **Run inference** — read those cubes, run the finetuned `EncoderWithHead`, and stitch
   the per-cell FSC patches into one daily COG.

- AOI: `data/fortress_mountain_basin_aoi.geojson` (single Polygon, CRS `OGC:CRS84` = WGS84
  lon/lat; ~50.81–50.85 N, -115.25– -115.18 E).

### Why two commands, not one

Their cost profiles are opposite. Cube download is slow, network-bound, GEE-throttled and
quota-consuming; inference is local, fast, and re-run constantly (new checkpoint, new
threshold, mosaic fix). Fusing them would force a "skip existing cubes" cache — which is
this same split, implemented worse. **Cubes on disk are the hand-off boundary**, and the
cube CSV is the contract between the two:

```
build_aoi_cubes_gee_url.py  --out-csv configs/aoi_cubes/cube_cells.csv --tifs-folder aoi_cubes
        │  cube CSV  (date, crs, center_lat, center_lon, min_x, min_y, max_x, max_y)
        │  cube dir  (data/aoi_cubes/PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif)
        ▼
infer_aoi_cubes.py          --cube-csv ... --cube-dir data/aoi_cubes --out-dir ...
        │  fsc_YYYYMMDD.tif  (100 m px, EPSG:32611, COG, nodata -9999)
```

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

### 4. Seamless native-UTM export + parallelism (2026-07-09)

The first live run exposed ~20 m seams: a 0-valued border sliver around each tile from
the EPSG:4326 round-trip (grid convergence tilts the UTM box → 0-filled slivers). Fixed by
exporting in the cell's **native UTM CRS** instead of round-tripping through 4326.

- **`eo_eval.py` `export_from_csv_utm_native`** — builds the request rectangle directly in
  the row's UTM `crs`, grows each cell by a `buffer_m` halo (default 40 m → neighbours
  overlap 80 m, for a downstream mean-mosaic to reconcile), and exports at
  **`crs=<UTM>` + `scale=10`**. EE resamples every band onto one shared 10 m lattice snapped
  to the UTM origin → axis-aligned, no tilt, no 0-border.
- **Dead end that was reverted:** an explicit `crs_transform` affine (to pin the origin)
  failed on ~2/3 of dates with `Expression evaluates to an image with inconsistent bounding boxes` — the 308-band composite has bands with mismatched native footprints, and without a
  `scale` to trigger auto-resample EE cannot form one grid. `crs=UTM + scale=10` sidesteps
  it entirely and produces the **same** origin/grid. `clip()` was a partial red herring (it
  only masked to region, didn't unify projections) and was removed.
- **Parallelism** — `url` mode is network-bound (blocking `getDownloadURL` + download per
  cube), so rows run through a `ThreadPoolExecutor` (`--max-workers`, default 4).
  cloud/drive stay serial (batch submits mutate shared `ee_task_list`). Workers capped low:
  GEE throttles `getDownloadURL` → a big pool returns 429s.

**Live verification (3-day AOI run, 2025-04-01..03, 25 cells/day):** 75/75 tiles, 0 errors;
all 108×108; every origin on a common 10 m grid (`origin mod 10 == 0`); **zero 0-border
pixels** on any tile; adjacent tiles overlap exactly 80 m.

Committed: `d4e34c38` (exporter fix), `871b54f4` (parallelism).

### 4b. Buffer default dropped to 0 — and why it must stay 0 (2026-07-13)

`d3a49ccb` changed the `--buffer-m` default to `0`. The 108×108 tiles described above were
the *buffer-40* run (`data/fortress_par3day/`, now superseded). The **current** cubes in
`data/aoi_cubes/` are the buffer-0 run and are what everything downstream assumes:

- 75 tiles (25 cells × 3 days, 2025-04-01..03), **100×100 px**, 10 m, 308 bands.
- Origins land exactly on the 1 km UTM lattice; tiles are adjacent and **non-overlapping**.

**Buffer > 0 is now a correctness bug, not just wasted bytes.** `dataset.subset_image`
(`src/snow_galileo/data/dataset.py:497-516`) crops an oversized cube down to 100×100 using
an **unseeded `np.random.choice`** offset. A 108×108 cube therefore yields a prediction for
a *randomly shifted* window — silently misregistered by up to 80 m, differently on every
tile and every day. The mosaic would look plausible and be wrong. `infer_aoi_cubes.py`
guards this explicitly (`_check_cube_shape`) and refuses to run on non-100×100 cubes.

Note this also means the "overlapping halo + mean-mosaic" idea sketched in §4 is a dead
end unless the loader's random crop is addressed first — `DailyMosaicWriter` asserts
non-overlap anyway (`mosaic.py:142`).

### 5. Inference step — `scripts/developer_scripts/infer_aoi_cubes.py` (2026-07-13, NOT YET RUN)

New Typer CLI, step 2 of the pipeline. Reuses the canonical leaf pieces and adds no new
classes or abstractions:

```
cube CSV ──▶ grid (GridCell.from_utm_bounds, from the CSV's UTM bounds)
         ──▶ cube path (filename reconstructed deterministically — never globbed)
         ──▶ symlink to a loader-parsable name (lat/lon from the CSV)
         ──▶ masked_output_for_tif  (the unchanged loader bridge)
         ──▶ EncoderWithHead        (batched forward, patch 10/1/1)
         ──▶ DailyMosaicWriter.write_day  ──▶  fsc_YYYYMMDD.tif
```

Options: `--cube-csv`, `--cube-dir`, `--out-dir`, `--config` (defaults to
`configs/bow_valley/inference.yaml`), `--device`, `--batch-size`, `--limit-days`.

#### Decision: the cube CSV is the single source of truth

The grid, the cube filename, and the cell centre in true degrees are **all** derived from
the CSV. Inference does **not** re-tile the AOI (it takes no `--aoi`), so the grid it
mosaics onto cannot drift from the grid the cubes were exported on.

#### Decision: `InferenceGridDriver` is deliberately NOT reused — ❌ REVERSED (2026-07-13)

> **This decision was wrong and is being undone.** It rested on a false premise: that the
> driver "has no 'read a cube directory' mode" and that satisfying it "would mean a
> fake-exporter stub with one call site." Both claims are incorrect — see below. The
> reversal is the next work item; the original reasoning is kept for the record.

~~The existing driver (`src/snow_galileo/inference/driver.py`) is wired to export cubes on
the fly through `LocalSourceExporter` and has no "read a cube directory" mode. Satisfying
it would mean a fake-exporter stub with one call site. Instead the script reuses the leaf
pieces that carry the real logic — `masked_output_for_tif`, `EncoderWithHead`,
`DailyMosaicWriter` — and duplicates only the ~30-line batching loop. `_build_model` is
likewise copied from the Bow Valley script (~20 lines) rather than importing script-to-script.~~

**Why it was wrong.** The driver's *entire* coupling to cube building is one line
(`driver.py:215`): `tif = self.exporter.export(cell=cell, window_end=day)`. It already
degrades gracefully for any injected object that is not a `LocalSourceExporter`:

- `run():122` — `cache = getattr(self.exporter, "_cache", None)` → `None` → no prune.
- `_pre_export_day():171-176` — `not isinstance(self.exporter, LocalSourceExporter)` →
  returns `{}` → `_tif_for_cell` is empty → the serial `.export()` fallback runs.

That fallback is documented in the driver itself as "the path the stub-exporter tests
exercise" (`driver.py:169`). So the driver already supports an injected cube source whose
whole contract is `.export(cell=..., window_end=...) -> Path`, and it does not care whether
that method *builds* a cube or *finds* one on disk. The "one call site" objection is also
false: there would be three implementors (real exporter, test stub, prebuilt-cube source).

**What the duplication actually cost.** `infer_aoi_cubes.py` copy-pasted ~120 lines whose
own comments admit it ("Mirrors `inference.driver._MASK_INDICES`", "Mirrors
`inference.driver`"): `_MASK_INDICES`, `_PATCH_SIZE_*`, `_is_fully_masked`, the whole
`_predict_day` batching/stack/forward/rearrange loop, and the `DailyMosaicWriter` day loop.
Two copies of the inference path means every fix needs landing twice.

**The reversal (next work item).** Add a ~25-line `PrebuiltCubeSource` whose `.export()`
does the CSV lookup → `_gee_cube_name` → `_check_cube_shape` → `_loader_safe_link` → return
the link. Inject it into `InferenceGridDriver`. The driver changes **by one type
annotation** (`exporter: LocalSourceExporter` is already a lie — the tests pass stubs and
the driver duck-types around them twice; a small `typing.Protocol` with just `export()`
makes the existing informal contract explicit). Net diff is negative.

`infer_bow_valley_daily_fsc.py` is **behaviourally untouched**: it keeps injecting
`LocalSourceExporter` and keeps the fused single-pass build+infer, the parallel pre-export,
and the day-frontier cache prune. The two scripts simply inject different cube sources into
the same driver — no cube/inference separation is required.

Carry-overs to preserve when reversing:

1. **Days.** The driver derives days from `inference_days(window_start, window_end)` — a
   contiguous range — and deliberately never reads a CSV `date` column (SPEC AC-31). The
   script derives them from `sorted(frame["date"].unique())`. On the current CSV these are
   identical (verified: `configs/aoi_cubes/cube_cells.csv` is 3 days, 20250401–20250403,
   step 1, 25 cells, 75 rows), so pass `window_start=min(days)`, `window_end=max(days)` —
   plus a one-line contiguity assert, or a future sparse CSV would make the driver iterate a
   day that has no cubes.
2. **Missing-cube fail-loud.** Keep the pre-flight check in `main()` *before* `driver.run()`
   (`infer_aoi_cubes.py:427`). Otherwise the failure moves later, into the shim's `.export()`.
3. **Symlink temp dir.** Move the `with tempfile.TemporaryDirectory()` out one level to wrap
   `driver.run()`; the shim holds `link_dir`.

#### Decision: symlinks, and NOT touching `src/` (operator's call, 2026-07-13)

The two pipelines have incompatible filename conventions. **Harmonising them is explicitly
out of scope for now.**

- The GEE exporter emits `PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif`. The doubled
  extension comes from two layers each appending one: `eo_eval.py:762` builds an identifier
  already ending in `.tif`, then `eo_eval.py:389` appends `_{crs}.tif`.
- **The loader parses the cell centre straight out of the filename** —
  `landsat_eval.py:257-266` does `float(tif_path.stem.split("_")[3])` and feeds the result
  to `to_cartesian()` for the model's location channels. On a real cube that call raises
  `ValueError: could not convert string to float: '-115.1902100076054865.tif'`. **Verified
  against a real file.** Every cube is unreadable today.
- Rejected: patching `landsat_eval._tif_to_array` — genuinely shared with training and
  evaluation. Not worth the blast radius.
- Rejected (for now): patching `eo_eval.py:762`. It is a one-line fix inside
  `export_from_csv_utm_native`, whose only caller in the repo is our own build script, so
  training/eval are provably unaffected — but the operator chose to leave `src/` frozen and
  keep this work script-local. **Revisit when harmonising the two pipelines.**
- **Chosen:** the inference script symlinks each cube (read-only, into a `TemporaryDirectory`,
  deleted on exit) to `PR_{date}_{lat:.16f}_{lon:.16f}_SC00.tif`, with **lat/lon taken from
  the CSV**, not scraped back out of the filename. No cube on disk is renamed or moved, and
  the location channels are exact rather than a lossy round-trip. Verified: a real cube
  under the corrected name loads through `masked_output_for_tif` and yields the expected
  13-tensor bundle (`(100,100,8,15)`, `(5,5,8,2)`, `(2,2,8,11)`, …).

#### Other choices

- **Checkpoint:** same as Bow Valley — `configs/bow_valley/inference.yaml`
  (`clouds_pretrained_42_lfdciemu.pth`, `fsc_inference_bow_river_tiny.json`, `finetune`).
  Reused as-is; no new config file.
- **Output:** stitched daily mosaic only (one COG per day), via `DailyMosaicWriter`.
- **Missing cube → hard failure.** A hole in the mosaic that renders as a plausible COG is
  worse than a crash, so the script refuses to write a partial day.
- **Fully-masked cell → nodata**, not a fabricated value (mirrors the driver's behaviour).

### 6. FIXED — the inference path never normalized its inputs (2026-07-13)

Open risk #1 (§F) was real. **Every FSC output produced by the local direct-source pipeline
so far is invalid** — Bow Valley Mode A, the ~33 h full Mode-B sweep, and anything Fortress
would have produced. They must be regenerated.

**The bug.** `_loader_bridge.py:54` set `ds.normalizer = None`. In `landsat_eval.py:596`,
`normalizer is None` takes the branch returning the raw `DatasetOutput` — no shift, no
divide. The encoder was fed **raw physical units**: reflectance in DN, S1 in dB, DEM in
metres (~1500–3000), ERA5 in Kelvin (~270) — against a checkpoint finetuned on
std-normalized inputs. Predictions were wrong-but-plausible; nothing failed loudly.

**Why the "mirrors the GEE runner" claim in the bridge's docstring was false.** The GEE
runner reaches its inference dataset through `_get_dataset` (`landsat_eval.py:1239-1249`),
which *always* assigns a normalizer, defaulting to `normalization="std"`
(`landsat_eval.py:823`, `876-883`). Verified that **every** caller omits the kwarg and so
takes that default: `finetune.py:121` (which produced the checkpoint), `eval_only.py:145`,
`run_inference.py:102` (the GEE inference runner). The local path was the only one that
skipped it.

**The fix.** `ds.normalizer = _inference_normalizer()` — a new `lru_cache(maxsize=1)`
builder returning `Normalizer(std=True, normalizing_dicts=<configs/normalizing_dict.json>)`,
i.e. exactly what `_get_dataset` builds. Cached because the bridge is called once per cube
(461,685 times in a full Mode-B sweep) and it reads JSON off disk. Docstring corrected.

**Is `configs/normalizing_dict.json` the right artifact at inference time?** Yes — and the
opposite would be the bug. Challenged on the grounds that the dict is fitted per-dataset and
so can't be available when inferring day-by-day as new data arrives. It isn't fitted at
inference; it is a **frozen training-time constant**, like ImageNet's mean/std:

- `compute_normalization.py`'s own argparse description: *"should be executed once before
  training"*.
- `Normalizer`'s class comment (`dataset.py:90-93`): the std bands use *"the pre-training
  population statistics"*.
- The dict is git-tracked with a fixed `total_n: 152559` — one population, one snapshot.
- `Normalizer(std=False)` is **dead code**: it leaves `normalizing_dicts=None`, and
  `__call__` (`dataset.py:182`) then raises `NotImplementedError`. `std=True` + this dict is
  the only normalization that works at all.
- Decisive: `run_inference.py` (the GEE runner) **already** loads this exact file while doing
  precisely the day-by-day-on-new-data thing. That case is already the served case.

Re-fitting the scaler on inference data would be train/serve skew and textbook leakage. The
constants must stay frozen so day 365 is comparable to day 1.

**⚠️ The finding that matters most — why no test caught this.** `Normalizer._normalize` is
`np.where(valid_data_mask, (x - shift) / div, NO_DATA_VALUE)` (`dataset.py:167`). On a cube
where **every** pixel is `-9999`, normalization is a *genuine no-op* — output is bit-identical
whether or not a normalizer is attached. The placeholder exporter emits exactly such a cube,
and `test_tracer_end_to_end.py` drives the loader with it (and hand-rolls the same
`normalizer = None` at line 174, which is how the bridge's copy looked legitimate).

> **No test built on placeholder cubes can detect a normalization bug.** "The tracer test
> passes" is not evidence that the model-input path is correct. A regression test must first
> overwrite the fill with valid pixels.

**Tests added** — `tests/test_local_sources/test_loader_bridge_normalizer.py` (2 tests):

1. The bridge's normalizer is element-wise identical to the one `_get_dataset` builds.
2. End-to-end: bridge output differs from the same loader driven with `normalizer=None`.
   Uses a `valid_cube` fixture that overwrites the placeholder `-9999` fill with valid
   pixels — without that, the test is vacuous (it failed exactly this way on first run).

**Validation.** `ruff check`, `ruff format`, `mypy` clean. `tests/test_local_sources/`:
263 passed, 17 skipped (missing-archive / missing-SNAP env guards), **0 failed** — zero new
failures against the documented baseline.

**Related footgun, NOT fixed (pre-existing).** `configs/normalizing_dict.json` is referenced
by bare path and is **not bound to the checkpoint**. Re-running `compute_normalization.py`
for a new dataset silently mis-normalizes every existing checkpoint — the same
wrong-but-plausible failure class. Affects `finetune.py`, `eval_only.py`, and
`run_inference.py` identically. Belongs in `KNOWLEDGE.md`.

### 7. Bow Valley Mode A regenerated with the normalizer fix (2026-07-13)

The §6 fix mandated regenerating every prior local-pipeline FSC output. **Bow Valley Mode A
is now regenerated** — the first of the three (Mode A, full Mode-B sweep, Fortress) to be
redone. The ~33 h Mode-B sweep and Fortress remain to regenerate.

**Run.** `infer_bow_valley_daily_fsc.py`, Mode A, on this workstation (checkpoint
`clouds_pretrained_42_lfdciemu.pth` **is** present here — the §D blocker is Fortress-specific,
tied to `infer_aoi_cubes.py`, not this script). 344 cells × 21 days
(2025-04-06..26), CUDA, warm cube cache (`--cache-policy reuse`), ~36 min, exit 0,
`aoi_coverage_fraction=1.0`, 34400 valid px/day.

- **Output redirected, old run preserved.** `INFER_OUT_DIR=data/normalized_daily_fsc`
  (env override of `InferenceSettings.out_dir`) writes the **new** run there; the **old,
  invalid** run stays intact at `data/bow_valley_processing/daily_fsc/` for the comparison.
  No config file edited.

**Comparison — new (normalized) vs old (raw-unit, invalid).**
`scripts/developer_scripts/bow_valley_inference_local/compare_fsc_runs.py` (new; per-day +
aggregate over the finite-valid overlap: mean each side, MAD, max abs diff, Pearson corr).

| metric        | value  |
| ------------- | ------ |
| old mean FSC  | 0.4402 |
| new mean FSC  | 0.3700 |
| mean abs diff | 0.2443 |
| max abs diff  | ~1.0   |
| Pearson corr  | 0.697  |

- **The fix changed real behaviour, as expected.** Mean FSC 0.44→0.37; per-pixel MAD 0.24
  (¼ of the [0,1] range); some pixels flip end-to-end (max_ad ≈ 1.0). Not a harmless offset.
- **Divergence is time-structured.** Window-start days (04-06..09) are worst — corr 0.37–0.56,
  old mean pinned high (~0.66–0.69) — settling to corr ~0.70–0.86 from 04-10 on. The old run
  over-predicted snow at window start: the wrong-but-plausible signature §6 describes.
- **New values sane.** Per-day new means all in [0,1]; no head/sigmoid misconfig sign.

**NOT done — QC on its own merits.** This is old-vs-new *only*. New ≠ correct. The §E.4 QC
(interior-nodata blocks; terrain-plausible & not spatially shifted vs Bow Valley) has **not**
been run against the new mosaic. Corr 0.70 says related-but-rescaled — it cannot confirm the
new field is physically right.

**Artifacts (uncommitted).** `compare_fsc_runs.py`; `data/normalized_daily_fsc/` (21 COGs +
`_run.log`).

## What remains to be done

### A. Provision GEE / GCP credentials on this machine (DONE)

Resolved — ADC refreshed via `gcloud auth application-default login`; live exports run.
Historical steps retained below for reference.

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

### B. Live smoke test (DONE — seamless 3-day run verified; see §4)

Historical smoke protocol. Run a **single cube** first (`--limit 1`) before the full AOI:

```
uv run python scripts/developer_scripts/build_aoi_cubes_gee_url.py \
  --aoi data/fortress_mountain_basin_aoi.geojson \
  --start-date 2025-04-01 --end-date 2025-04-01 \
  --tifs-folder fortress_smoke --out-csv configs/aoi_cubes/fortress_smoke.csv \
  --limit 1
```

Verify one `.tif` lands in the configured `DATA_FOLDER/fortress_smoke/`, then drop
`--limit 1` for the full single-date run (25 cubes), and finally the real date range.

### C. Downstream stitching seam (RESOLVED — see §5)

Superseded. The filename mismatch is handled in `infer_aoi_cubes.py` by joining on the cube
CSV (UTM bounds → `cell_id`) and symlinking to a loader-parsable name. Neither pipeline's
naming was changed. Kept as a known debt: see "Revisit when harmonising" in §5.

### D. BLOCKER — no model checkpoint on this workstation

`infer_aoi_cubes.py` has never been executed. `logging_checkpoints/` holds only
`__init__.py`; there is **no `.pth` anywhere in the repo**, so
`clouds_pretrained_42_lfdciemu.pth` (referenced by `configs/bow_valley/inference.yaml`)
cannot be loaded here. The script fails loudly with `FileNotFoundError` by design — it never
random-inits.

**This is why the work is being moved to the compute server, where the checkpoints live.**

### E. Resume protocol (on the compute server)

Everything below is unverified-at-runtime. Do these in order.

1. **Confirm the checkpoint resolves:**

   ```
   ls -l logging_checkpoints/snowgalileo_finetune/clouds_pretrained_42_lfdciemu.pth
   ```

   If it lives elsewhere, override rather than edit the YAML:
   `INFER_CHECKPOINT=/path/to/model.pth`.

2. **Smoke one day, on CPU first** (isolates plumbing bugs from CUDA/VRAM ones):

   ```
   uv run python scripts/developer_scripts/infer_aoi_cubes.py \
     --cube-csv configs/aoi_cubes/cube_cells.csv \
     --cube-dir data/aoi_cubes \
     --out-dir data/outputs/fortress_fsc \
     --device cpu --batch-size 5 --limit-days 1
   ```

   Expect one `fsc_20250401.tif`. The 25 cells span a 5×5 km AOI at 100 m px → a **50×50**
   raster, EPSG:32611, nodata −9999.

3. **Then the full 3 days** (drop `--limit-days`, `--device cuda`).

4. **QC the mosaic — do not skip this.** The plumbing being green does not mean the result
   is right. Check, at minimum:

   - FSC values are in `[0, 1]` (anything outside means the sigmoid/head config is wrong).
   - No `-9999` blocks in the interior (a nodata block = a fully-masked cell; investigate,
     don't accept).
   - The mosaic's spatial extent matches the AOI, and the snow pattern is physically
     plausible against terrain (Fortress Mountain: snow on the basin/upper slopes, less in
     the treed valley). A plausible-looking but *shifted* pattern is the failure mode §4b
     warns about.

### F. Open risks (carried into the next session)

1. ~~**`normalizer=None` in the loader bridge** (`_loader_bridge.py:54`).~~ ✅ **RESOLVED
   2026-07-13 — the risk was real. See §6.** The bridge now builds the same
   `Normalizer(std=True, …)` the GEE/eval path uses. **Consequence: every FSC output the
   local pipeline has produced to date is invalid and must be regenerated** (Bow Valley
   Mode A, the ~33 h full Mode-B sweep, anything Fortress). Also learned: no test built on
   all-`-9999` placeholder cubes can detect a normalization bug — §6 explains why.
2. **The seam guard is a bare `assert`** (`mosaic.py:142`) — stripped under `python -O`.
   Don't run the mosaic writer with optimisations on.
3. `infer_aoi_cubes.py` is lint/typecheck-clean (ruff + mypy) but has **no test and no
   runtime execution** behind it. The first real run is also its first test. (The
   `InferenceGridDriver` reversal in §5 shrinks this surface: the batching/mask/mosaic path
   becomes the driver's, which *is* tested.)
4. **`configs/normalizing_dict.json` is not bound to the checkpoint** (pre-existing, repo-wide).
   Re-running `compute_normalization.py` for a new dataset silently mis-normalizes every
   existing checkpoint — same wrong-but-plausible failure class as #1. Affects `finetune.py`,
   `eval_only.py`, `run_inference.py` equally. Should be recorded in `KNOWLEDGE.md`.

### G. Commit

Not yet committed. Uncommitted at handoff:

```
?? scripts/developer_scripts/infer_aoi_cubes.py                    # step 2 — inference (§5)
?? configs/aoi_cubes/                                              # generated cube CSVs
?? tests/test_local_sources/test_loader_bridge_normalizer.py       # normalizer regression (§6)
 M src/snow_galileo/inference/_loader_bridge.py                    # THE NORMALIZER FIX (§6)
 M docs/agents/planning/fortress_mountain_basin_aoi/PLAN.md
```

**`src/` is no longer untouched.** The original "keep this work script-local" stance
(§5, "Decision: symlinks, and NOT touching `src/`") held for the *filename* problem — that
one is still solved with symlinks, and `eo_eval.py` / `landsat_eval.py` remain frozen. But
the normalizer bug (§6) is *in* `src/snow_galileo/inference/_loader_bridge.py` and cannot be
fixed from a script; leaving it would mean knowingly shipping invalid predictions. The
operator has since approved extending `src/snow_galileo/inference/` and
`src/snow_galileo/data/local_sources/`, provided the
`scripts/developer_scripts/bow_valley_inference_local/` scripts keep working with minimal
changes — which the §5 reversal preserves (`infer_bow_valley_daily_fsc.py` is behaviourally
unchanged).
