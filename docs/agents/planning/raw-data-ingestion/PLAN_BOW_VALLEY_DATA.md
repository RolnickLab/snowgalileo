# Plan — Bow Valley Direct-Source Data Cube & Daily Snow Cover Inference

## 1. Goal

Adding a direct-source pipeline for the Bow Valley (Alberta, west of Calgary) to 
supplement Google Earth Engine ingestion to :

1. Build a spatiotemporal data cube matching the exact tensor contract defined
   in `DATA_ANALYSIS.md` (band order, shapes, masks, normalization, `-9999`
   nodata).
2. Run the pretrained SnowGalileo encoder + FSC head over the cube as a
   **1 km × 1 km grid sweep, per day**, producing a daily fractional snow cover
   raster mosaic for the entire AOI.

The repository's downstream code (`Dataset`, `LandsatEvalDataset`,
`EncoderWithHead`, `LandsatEval`, metrics) must remain untouched. We add two
new modules behind the existing contracts: a direct-source ingestion adapter
and an inference-grid driver.

---

## 2. Scope & Non-Goals

**In scope**
- Local-file adapters for S1, S2, Landsat 8/9, S3 OLCI, MODIS MOD09GA,
  VIIRS VNP09GA, ERA5-Land, Copernicus DEM GLO-30, ESA WorldCover v200.
- Producer of per-cell GeoTIFFs matching `create_ee_image` layout (dynamic
  stack + static stack, `-9999` nodata, `EPSG:4326`, scale=10, dims ~100×100,
  `MODIS_FILL_VALUE=-28672` preserved in MODIS bands).
- A 1 km grid generator covering the Bow Valley AOI in EPSG:32611.
- A daily-stride inference driver that, for each day `d` in the configured
  window, builds an 8-day input window ending at `d`, runs `EncoderWithHead`
  over every grid cell, and mosaics the per-cell 10×10 FSC predictions into a
  daily COG.
- Validation harness comparing direct-source patches against GEE reference
  patches **generated as part of Phase 0** by running the existing GEE pipeline
  over a small held-out sample (5–10 cells × 3 days).

**Out of scope**
- Retraining or fine-tuning the model.
- Changing band order, normalization constants, mask semantics, or tensor
  shapes.
- Adding new modalities (cloud-flag decoding, VIIRS QF1, etc.).
- Real-time / streaming ingestion. Batch job over a static archive.
- Fixing the known ERA5 temperature shift sign or the S3 identity normalization
  TODO — these are model-numeric-domain concerns, addressed in a separate
  migration.

---

## 3. AOI, Grid, and Temporal Window

### AOI — computed from `sampled_cells_bow_river_with_dates.csv`

500 sampled cells in EPSG:32611. Convex-bbox of all cell extents:

| Bound | Value (EPSG:32611, m) |
| --- | --- |
| `min_x` | `518363.85` |
| `max_x` | `705363.85` |
| `min_y` | `5599583.79` |
| `max_y` | `5761583.79` |
| width  | `187_000` (187 km) |
| height | `162_000` (162 km) |
| max-tile grid | `187 × 162 = 30_294` 1 km cells if fully tiled |

**Decision required before FDD (see §8 Q3):** sweep mode.
- **(A) Sample-only:** infer over the 500 CSV cells only. Cheap (~500 cells).
- **(B) Full tile:** infer over all ~30 k cells in the AOI bbox. Storage and
  compute are ~60× larger; needs explicit GPU budget.

Plan assumes **(A)** as default. (B) is a configuration switch on the grid
generator.

### Grid + CRS

| Parameter | Value | Source / Rationale |
| --- | --- | --- |
| Grid math CRS | `EPSG:32611` (UTM 11N) | Matches CSV cell extents; preserves 1 km cell metric. |
| Per-cell export CRS | `EPSG:4326`, `scale=10` | Matches `create_ee_image`; downstream loader assumes this. |
| Daily mosaic CRS | `EPSG:32611` | Mosaic stays in metric CRS for analysis. **Per-cell rasters in 4326, mosaic in UTM is intentional** — per-cell tifs feed the loader unchanged; mosaic is a separate output product. Reprojection happens once, at mosaic-write time, on 10×10 FSC outputs (low IO). |
| Grid cell size | 1000 m × 1000 m (≈ 100×100 px @ 10 m) | `EXPORTED_HEIGHT_WIDTH_METRES` |
| Cell layout | Non-overlapping; centred on CSV `center_x, center_y` for mode (A) or tiled from `min_x, min_y` for mode (B) | Matches existing CSV semantics |

**CRS is law** — every cell carries an explicit `transform`, `crs`, and `shape`
triple that all adapters must conform to.

### Fixed Extent Mosaic & Scene Coverage Complexity

A fixed spatial extent is provided for the full Bow Valley AOI desired daily mosaic. This introduces significant operational complexity:
- **Multi-Scene Composition**: The full AOI bbox is approximately 187 km × 162 km. This massive area requires a grid of roughly 2x2 product scenes (close to 4 scenes total of Landsat or Sentinel-2) to achieve complete spatial coverage.
- **Incomplete Daily Coverage**: Because of sensor orbit path timings, swath widths, and scene collection grids, we will **never** have 100% spatial coverage of the full AOI on a single acquisition day/timestamp. Some parts of the AOI will have scenes on day `d`, while other parts will have nodata.
- **Orbit/Swath Boundary Nodata**: Scenes near orbit boundaries or swath edges often contain significant regions of native nodata. Cells that overlap scene edges will have partial observations.
- **Mosaicing & Composite Strategy in Direct-Source Pipeline**:
  - For a given 1 km x 1 km grid cell, it may fall in the overlap region of multiple adjacent/swath-overlapping scenes on the same day, or it may fall on the edge of a scene where part of the cell is nodata.
  - If we replicate GEE's naive `.first()` selection per day per cell, we might ingest a scene that only partially covers the cell even if a fully-covering scene is available, or we might miss coverage in the overlap areas.
  - Therefore, the local adapters must handle mosaicing of all available scenes/granules for the target day *before* cropping to the 1 km grid cell. This ensures maximum coverage and reduces artificial `nodata` boundaries within grid cells.
- **Heterogeneous Daily Coverage in daily mosaic**:
  - The final daily mosaic will always be incomplete (heterogeneous coverage). Some 1 km grid cells will be completely invalid (all `-9999` inputs, leading to `nodata` in predictions), while others will have valid outputs.
  - The `DailyMosaicWriter` must be robust to missing grid cells or cells with degenerate outputs, stitching only valid predictions into the daily UTM 11N COG.

### Temporal window

| Parameter | Value | Source |
| --- | --- | --- |
| Cube inference period | Configurable; default `2024-02-01 → 2024-04-30` (snowmelt) | TBD per §8 Q2 |
| **Archive ingest period** | `start - 7 days → end` | Needed to fill the 8-day window for `d = start` |
| Timestep stride | 1 day (`DAYS_PER_TIMESTEP`) | `config.py` |
| Window per inference | 8 days (`NUM_TIMESTEPS`) | `config.py` |
| Prediction cadence | 1 prediction per cell per day, `d ∈ [start, end]` | sliding window |

The 7-day prefill is non-negotiable: the model needs 8 timesteps. Phase 0
archive audit MUST verify ingest coverage of `start − 7` through `end`, not
just the inference range.

### Filename convention (CONTRACT — resolved here)

The existing `LandsatEvalDataset` parses filenames at
`src/fsc/landsat_eval.py:171-176, 254-262`. Two branches:

- **Landsat-prefixed** (`LC*`, `LE09*`, `LC08*`, `PR*`): name format is
  `L0X_YYYYMMDD_LAT_LON_SC{cloud}.tif`. Month at `parts[1][4:6]`, lat at
  `parts[3]`, lon at `parts[4]`.
- **Non-Landsat** (default): month at `parts[0][5:7]` (i.e. `parts[0]` is
  shaped `XXXXX-MM-DD-...` or similar), lat at `parts[2]`, lon at `parts[3]`.

The non-Landsat branch is brittle and undocumented. **Decision:** the
`LocalSourceExporter` emits the **Landsat-style filename** for every cell:

```
dates={YYYYMMDD}_{window_start_YYYYMMDD}_{LAT}_{LON}_SC00.tif
```

Wait — that doesn't match. Re-reading `landsat_eval.py:171-176`: the
Landsat-style requires `parts[0]` to start with one of `LC|LE09|LC08|PR` and
`parts[1]` to be `YYYYMMDD`. So the exporter emits:

```
PR_{YYYYMMDD_window_end}_{LAT_DEG}_{LON_DEG}_SC00.tif
```

where `PR` is the recognized "synthetic / predicted" prefix already supported
in the parser, and `LAT_DEG`/`LON_DEG` are signed decimal degrees (`53.1234`,
`-115.6789`). **SPEC AC:** regex
`^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$` matches and
`prediction_month_from_file` returns the expected month for every exported tif.

The `PR` prefix's meaning in the existing code MUST be verified during FDD
phase (see §8 Q9) — if `PR` denotes PlanetScope rather than predictions, pick
a different recognized prefix or add a new one and patch the parser allowlist
in the same PR (touches `landsat_eval.py:172` — minimal additive change).

---

## 4. Architecture

Ports & Adapters. The existing `Dataset` is a *consumer* of a logical raster
stack. We introduce a `LocalSourceExporter` that produces the same stack from
local files, plus an `InferenceGridDriver` that orchestrates per-cell
inference.

```
                ┌──────────────────────────────────────────┐
                │       InferenceGridDriver (new)          │
                │  AOI → 1 km grid → per-cell jobs          │
                └────────────┬─────────────────────────────┘
                             │ per (cell, day)
                             ▼
                ┌──────────────────────────────────────────┐
                │  LocalSourceExporter (new)               │
                │  Implements the same contract as          │
                │  EarthEngineExporter.create_ee_image     │
                └────────────┬─────────────────────────────┘
                             │ assembles dynamic + static
                             ▼
        ┌──────────────────────────────────────────────────────────┐
        │  Source Adapters (new, one per modality)                 │
        │  S1GRDLocal • S2HarmonizedLocal • Landsat89Local •        │
        │  S3OLCILocal • MOD09GALocal • VNP09GALocal •              │
        │  ERA5LandLocal • CopernicusDEMLocal • ESAWorldCoverLocal │
        └────────────┬─────────────────────────────────────────────┘
                     │ GeoTIFF in canonical layout
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   LandsatEvalDataset (existing, unchanged)               │
        │   → MaskedOutput groups                                  │
        └────────────┬─────────────────────────────────────────────┘
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   EncoderWithHead (existing) → per-cell 10×10 FSC        │
        └────────────┬─────────────────────────────────────────────┘
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │   DailyMosaicWriter (new) → daily FSC COG over AOI       │
        └──────────────────────────────────────────────────────────┘
```

### Module layout (new)

```
src/data/local_sources/
  __init__.py
  base.py             # LocalSourceAdapter ABC, GridCell dataclass, CellWindow
  s1.py               # Sentinel-1 GRD adapter
  s2.py               # Sentinel-2 L1C → harmonized DN
  landsat.py          # Landsat 8/9 C2 T1 TOA (encapsulates L9→L8 fallback)
  s3.py               # Sentinel-3 OLCI radiance
  modis.py            # MOD09GA (HDF/Mosaic); preserves -28672 fill value
  viirs.py            # VNP09GA (HDF/Mosaic) — emits both fine (per-pixel) and coarse (per-pixel raster, loader averages)
  era5.py             # ERA5-Land daily aggregates (NetCDF/GRIB); emits raw Kelvin
  dem.py              # Copernicus DEM GLO-30 + slope + aspect
  worldcover.py       # ESA WorldCover v200 (static, 2021 map; ignores day)
  cube_cache.py       # Per-modality per-(cell, day) numpy cache (.npz)
  exporter.py         # LocalSourceExporter — assembles multiband tif from cache
  grid.py             # AOI → 1 km grid generator (modes A and B)
  layout.py           # Canonical band order constants (re-exports from eo.py)

src/inference/
  __init__.py
  driver.py           # InferenceGridDriver
  mosaic.py           # DailyMosaicWriter (per-day COG, UTM 11N)
  windows.py          # Sliding 8-day window builder

scripts/
  export_bow_valley_cube.py     # build the cube from local archive
  infer_bow_valley_daily_fsc.py # run model + mosaic per day

configs/bow_valley/
  cube.yaml          # archive paths, AOI bbox, date range, CRS, mode A/B
  inference.yaml     # checkpoint path, batch size, output dir, days

tests/test_local_sources/
  test_grid.py
  test_filename_contract.py    # exporter filenames pass LandsatEvalDataset parser
  test_<modality>_adapter.py   # one per adapter, value-domain + CRS asserts
  test_s1_parity.py            # GEE parity spike (Phase 3 step 0)
  test_s2_parity.py            # GEE parity spike (Phase 3 step 0)
  test_exporter_parity.py      # full-stack numeric parity vs GEE reference patches
  test_tracer_end_to_end.py    # one cell, one day → EncoderWithHead → assertions
  test_inference_driver.py
```

### Adapter contract

Every adapter implements:

```python
class LocalSourceAdapter(Protocol):
    bands_out: list[str]                       # exact names expected downstream
    spatial_kind: Literal["high","med","low","time","space","static"]
    native_fill: float | None                  # e.g. -28672 for MODIS; None if only -9999

    def fetch(
        self,
        cell: GridCell,                        # polygon, CRS, target transform, shape
        day: datetime.date | None,             # None for static layers
    ) -> np.ndarray:                           # shape (C, H, W); -9999 nodata
        ...
```

Rules enforced by `base.py`:
- Output reprojected to the cell's target grid (`EPSG:4326`, scale=10) using:
  - **bilinear** for continuous bands,
  - **nearest** for QA / categorical (WorldCover, cloud flags).
- Missing acquisition → return `-9999` array of declared shape
  (`create_placeholder` equivalent).
- MODIS adapter MUST preserve the native `-28672` fill value in addition to
  `-9999`. The downstream NDSI/NDVI computation at
  `src/fsc/landsat_eval.py:317, 331` asserts the fill value is *encountered*
  (sentinel for "MODIS data was actually present"). Stripping it will crash
  the loader.
- VIIRS coarse adapter emits a per-pixel raster at the cell grid. The loader
  averages spatially into `time_x`. Do not pre-average in the adapter.
- ERA5 adapter emits raw Kelvin / native units. The known temperature shift
  inconsistency lives in `Normalizer`, downstream of the adapter; adapter is
  not responsible for replicating it.
- Landsat adapter encapsulates the L9→L8 fallback internally (single
  `bands_out=["B2_landsat",..,"B7_landsat"]`), matching GEE behaviour.
- WorldCover adapter ignores `day` and returns the v200 2021 map. Hardcoded.

### LocalSourceExporter

Mirrors `create_ee_image` exactly:
- Iterates `NUM_TIMESTEPS=8` days.
- Calls each time-varying adapter per day in the **fixed order**:
  `S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5
   + MODIS cloud flag + S2 cloud flag + Landsat cloud flag`.
- Appends static stack once: `DEM, slope, aspect, WorldCover Map`.
- Writes a multiband GeoTIFF per (cell, window-end-day) under the Landsat-style
  filename defined in §3.

### Cube cache layout (correctness fix vs prior draft)

The cache is **per-modality per-(cell, day)** numpy arrays (`.npz`), not
per-(cell, window-end-day) multiband tifs. Rationale: 8-day windows for
consecutive days overlap by 7 days. Caching at the multiband-tif level would
duplicate ~8× the storage. The exporter:

1. For each `(cell, day)` in the 8-day window, queries the cache
   `cube_cache.get(modality, cell_id, day)`.
2. On miss, calls the adapter, writes the array to `.npz`, returns it.
3. After all 8 days × all modalities are gathered, assembles the multiband tif
   in the canonical band order and writes it to the exporter output dir.

Storage estimate (mode A, 500 cells × 90 inference days = 96 archive days):

- Per-cell per-day per-modality: ~100×100 × float32 × bands. S1 (3) + S2 (6+1
  QA) + Landsat (6+1 QA) + S3 (2) + MODIS (7+1 QA) + VIIRS fine (2) +
  VIIRS coarse (4) + ERA5 (5) ≈ 38 bands × 40 KB = 1.5 MB per (cell, day).
- 500 cells × 96 days × 1.5 MB ≈ **72 GB cache**, plus ~75 GB of assembled
  multiband tifs (500 × 90 × ~1.6 MB).

Cache is FIFO-evicted with a size cap (configurable, default 200 GB). Scratch
dir is configured in `cube.yaml`.

---

## 5. Inference Grid Driver

Pseudocode:

```python
grid = build_grid(aoi_bbox_utm, cell_size_m=1000, crs="EPSG:32611", mode=mode_A_or_B)
for day in daterange(start, end):                              # inference range
    cube_paths = []
    for cell in grid:                                          # parallel
        # window covers [day - 7, day], inclusive
        tif = exporter.export(cell, window_end=day)
        cube_paths.append((cell, tif))
    batches = batch_cells(cube_paths, batch_size=N)
    preds = run_encoder_with_head(model, batches)              # (B, 10, 10) FSC
    mosaic.write_day(day, preds, grid)                         # COG, EPSG:32611
```

Key design choices:
- **Cube reuse:** see §4 cache layout — per-modality per-day, not per-window.
- **Parallelism:** per-cell export is embarrassingly parallel
  (`multiprocessing.Pool`). Inference is GPU-batched across cells.
- **Mosaic:** `DailyMosaicWriter` writes one COG per day in EPSG:32611. Each
  cell's 10×10 FSC patch maps to a 100 m pixel grid (`100 m × 10 px = 1 km`
  cell). Cells with all input groups masked fall back to `nodata`. The 10×10
  FSC raster from each cell is reprojected from EPSG:4326 (the loader's grid)
  to EPSG:32611 at mosaic-write time using nearest-neighbour (FSC is a
  prediction; bilinear would blend invalid neighbours).
- **Per-cell independence:** the model runs one forward pass per 1 km cell
  with no cross-cell spatial context (`patch_size_high_res=10`, input 100×100,
  output 10×10 at `src/fsc/patch_predict.py:26`). Edge effects on the 100 m
  output pixels at cell boundaries are a known *modelling* limitation, not a
  mosaic-stitching bug. Documented in §6.

---

## 6. Verification & FMEA (sketch — formalized in FDD)

| Risk | Mitigation |
| --- | --- |
| Filename contract mismatch with `LandsatEvalDataset` parser | `test_filename_contract.py` asserts every exporter-emitted filename parses correctly via `prediction_month_from_file` and yields expected `(month, lat, lon)`. Tested before any adapter is implemented. |
| Adapter value-domain drift vs GEE | Per-adapter parity test against GEE reference patch (numeric diff thresholds per source). S1 and S2 parity spikes run **first** (Phase 3 step 0) — highest interchange risk per `DATA_ANALYSIS.md`. |
| CRS / pixel-alignment mismatch | All adapters share one resampler in `base.py` driven by `GridCell.transform`. Golden-grid test asserts exact `transform`, `shape`, `crs`. |
| Band order regression | `layout.py` re-exports the canonical lists from `eo.py`; integration test asserts byte-for-byte band-name equality against `create_ee_image` output. |
| MODIS native fill stripped | MODIS adapter test asserts `-28672` is preserved in output where source had it. NDSI/NDVI assertions at `landsat_eval.py:317, 331` would crash otherwise. |
| Landsat L9→L8 fallback regression | Landsat adapter test exercises 3 scenarios: L9 present, L9 missing + L8 present, both missing → `-9999`. |
| VIIRS coarse pre-averaging breaks `time_x` | VIIRS coarse adapter returns shape `(4, 100, 100)`, not `(4,)`. Test asserts shape and that the loader's spatial mean reproduces GEE values. |
| Mosaic seams between adjacent cells | Cells are non-overlapping by design; verify via overlap=0 assertion in `grid.py` and a 2×2 mosaic visual test. Cross-cell context limitation documented separately. |
| Memory/IO blowup on full AOI sweep | Per-modality per-(cell, day) `.npz` cache, FIFO-evicted, size cap. S1 SAFE archives processed via windowed reads (read-only the cell footprint), not full-scene loads. Concrete sizing: §4. |
| Cloud flag bands dropped silently downstream | Emit them in the GeoTIFF anyway; loader will drop them. Documented. Keeps GEE byte-layout parity. |
| ERA5 normalization bug accidentally "fixed" | Adapter emits Kelvin; bug lives in `Normalizer` and stays as-is (out of scope per §2). |
| Per-cell modelling edge effects at 100 m output boundaries | Documented limitation: every cell is an independent forward pass. No mitigation planned in this work. Flag in `KNOWLEDGE.md`. |

### Tracer-bullet integration test (concrete assertions)

Export one cell for one window-end-day from local archive, run it through
`LandsatEvalDataset` + `EncoderWithHead`, assert:

1. `space_time_high_res_x.shape == (100, 100, 8, 15)`
2. `space_time_med_res_x.shape == (5, 5, 8, 2)` (note: docstring at
   `landsat_eval.py:236` says `(3, 3, T, C_STM)` but is stale; actual target
   is 5×5 per `DATA_ANALYSIS.md` §Compatibility caveats)
3. `space_time_low_res_x.shape == (2, 2, 8, 11)` (incl. NDSI + NDVI)
4. `time_x.shape == (8, 9)`
5. `space_x.shape == (100, 100, 14)`
6. `static_x.shape == (3,)`
7. FSC prediction shape `== (10, 10)`, values ∈ `[0.0, 1.0]`
8. `valid_data_mask_*` all set wherever inputs are `-9999` or below their
   `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`.
9. Filename parses correctly: `prediction_month_from_file` returns the month
   matching `window_end.month`.

These nine assertions are the SPEC's primary ACs.

### Tribal knowledge

`KNOWLEDGE.md` (create if missing) gets entries for:
- MODIS native fill `-28672` is sentinel for "data present" — don't strip.
- ERA5 normalization sign error is known and deliberately preserved.
- S3 normalization TODO is known; identity is intentional.
- `PR` filename prefix is recognized by the eval-dataset parser; verify its
  original meaning before reusing.
- Per-cell forward pass has no cross-cell spatial context — 100 m FSC pixels
  near cell boundaries may have edge effects.

---

## 7. Phased Delivery

Each phase ends with an explicit approval gate per CLAUDE.md workflow rules.

1. **Phase 0 — Archive Audit + GEE Reference Patch Generation (no production
   code).**
   - Catalog every file we have for each modality: paths, formats (SAFE / HDF /
     NetCDF / COG), CRS, native scale, coverage of `[start-7, end]` × AOI,
     gaps.
   - Run the existing GEE exporter (`scripts/export_for_eval.py`) over a
     5–10 cell × 3-day held-out sample to produce **GEE reference patches**
     used by every parity test in Phase 2/3.
   - Verify the meaning of the `PR` filename prefix in `landsat_eval.py:172`.
   - Output: `docs/agents/planning/bow_valley/ARCHIVE_AUDIT.md` +
     `tests/fixtures/gee_reference_patches/`.
2. **Phase 1 — FDD.** Formal Design Document per planning skill, including the
   tracer test of §6 with the nine concrete assertions. Approval gate.
3. **Phase 2 — SPEC.** Acceptance criteria as test sentences. Per-adapter
   value-domain assertions, per-adapter parity thresholds (numeric diff
   tolerance) vs GEE reference patches generated in Phase 0. Approval gate.
4. **Phase 3 — Tasks (vertical slices).** Re-ordered to de-risk high-impact
   adapters early.
   - **Step 0 — Parity spikes (throwaway).** Stand up minimal S1 and S2 GRD/L1C
     download + reprojection scripts. Compare to GEE reference patches.
     Quantify drift. *Decision point:* if drift is too large to recover with
     processing, escalate before sinking effort into the full ports/adapters
     stack.
   - **Step 1 — Contract.** `base.py` + `GridCell` + `grid.py` + `layout.py` +
     `cube_cache.py`. `test_grid.py`, `test_filename_contract.py` pass.
   - **Step 2 — Placeholder exporter + tracer test.** `LocalSourceExporter` +
     placeholder adapters returning `-9999`. `test_tracer_end_to_end.py` passes
     with all-`-9999` cubes (FSC will be degenerate but pipeline plumbed).
   - **Step 3 — Adapters, in difficulty / parity-risk order:**
     1. `worldcover.py` (static, easy)
     2. `dem.py` (static, easy)
     3. `era5.py` (low parity risk, well-defined NetCDF)
     4. `modis.py` (mind the `-28672` fill)
     5. `viirs.py`
     6. `s3.py`
     7. `landsat.py` (with L9→L8 fallback test)
     8. `s2.py` (parity spike already done, now production)
     9. `s1.py` (parity spike already done, now production)
   - **Step 4 — `InferenceGridDriver` + `DailyMosaicWriter`.**
   - **Step 5 — `scripts/export_bow_valley_cube.py` and
     `scripts/infer_bow_valley_daily_fsc.py`.**

### Tooling per task

Detected from repo state (confirm in FDD):
- Linting: `ruff` (default per CLAUDE.md if `.pre-commit-config.yaml` does not
  specify otherwise).
- Type checking: `mypy` (default per CLAUDE.md).
- Tests: `pytest`.

Every Phase 3 step must pass `ruff check`, `mypy`, and the relevant test set
before approval.

---

## 8. Open Questions (need user input before FDD)

1. **Archive locations.** Where on disk does each modality live? Naming
   conventions? (blocks Phase 0)
2. **Date window.** Confirm `2024-02-01 → 2024-04-30` default. Available
   CSV-recorded dates span 2024-01-05 to 2025-12-22.
3. **Sweep mode.** (A) infer only the 500 CSV cells, or (B) full 30 k-cell
   AOI tile? (drives compute estimate; default (A))
4. **CSV `date` column semantics.** Does the date in
   `sampled_cells_bow_river_with_dates.csv` represent the
   *prediction day for that cell* (cell-specific window-end) or just sampling
   metadata to ignore? If per-row prediction days, the driver loop changes
   from "all cells, every day" to "each cell, its assigned day."
5. **Output destination.** Local disk vs object storage for daily COGs?
6. **Checkpoint.** Which finetuned `EncoderWithHead` checkpoint feeds
   inference? Path?
7. **Compute budget.** GPU count and wall-clock target. Mode (A): ~45 k
   forwards, hours on one GPU. Mode (B): ~2.7 M forwards, needs multi-GPU.
8. **Sentinel-2 product level.** Confirm L1C (matches `S2_HARMONIZED` value
   domain) vs L2A (would break normalization).
9. **`PR` filename prefix meaning.** Inspect existing eval tifs to confirm
   what `PR` originally denoted. If it's used by another data source already
   on disk, pick a different prefix (e.g. `SY` for synthetic) and add to the
   parser allowlist at `src/fsc/landsat_eval.py:172` in the same PR.
10. **Cloud-flag emission.** Keep emitting (default; preserves GEE byte
    layout, dropped downstream) or skip to save IO? Recommend keep.

---

## 9. Non-Negotiables (from `DATA_ANALYSIS.md`)

- Dynamic band order per timestep is fixed.
- Nodata is `-9999`. MODIS additionally preserves native
  `MODIS_FILL_VALUE=-28672` — both must survive into the exported tif.
- Per-cell export grid is `EPSG:4326`, `scale=10`, ~100×100 px per cell.
- WorldCover stays as a single `Map` band; the loader one-hot encodes.
- Normalization constants are tuned to GEE-exported numeric ranges — do not
  "fix" the ERA5 temperature sign or S3 identity normalization in this work.
- Categorical / QA → nearest-neighbor resampling only.
- VIIRS coarse bands are exported as per-pixel rasters on the cell grid; the
  loader (not the adapter) does the spatial mean into `time_x`.
- Landsat L9→L8 fallback is encapsulated inside the Landsat adapter.
- **Mosaicing overlapping daily scenes is mandatory**: Any scene overlaps or swath edge boundaries must be composite-mosaiced prior to cropping to avoid artificial nodata boundaries within a single 1 km grid cell.

