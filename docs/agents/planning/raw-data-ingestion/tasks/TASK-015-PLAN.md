# TASK-015 implementation plan — InferenceGridDriver + DailyMosaicWriter

## Goal (SPEC FR-21, FR-22; AC-28, AC-29, AC-31)
Add `src/inference/{windows,driver,mosaic}.py` + `tests/test_local_sources/test_inference_driver.py`.
For each day in the configured window, build the 8-day window per cell, export each cell's
cube, run `EncoderWithHead` per cell → 10×10 FSC, and mosaic the per-cell patches into one
daily COG in EPSG:32611, recording per-day AOI coverage.

## Binding constraint — DOWNSTREAM IS SACRED (additive only)
This pipeline is an **extension** of the GEE path, not a modification. Every existing module,
import path, signature, and behaviour must still work unchanged after TASK-015:
- **Zero edits** to `src/fsc/*` (`LandsatEvalDataset`, `EncoderWithHead`, `DatasetOutput`,
  `Normalizer`), `src/data/earthengine/*`, or any existing loader/model code. Verified
  net-new: nothing imports `src.inference` today; `_tif_to_array`/`__getitem__`/`DatasetOutput`
  are **shared** with the GEE `_predict_and_store_output` path → strictly **read-only** to us.
- The GEE → `_predict_and_store_output` pathway keeps running untouched; `InferenceGridDriver`
  is a **parallel, additive** entry point, not a replacement.
- **One fragile coupling point, isolated in OUR code.** Driving the loader requires the
  `__new__`-bypass-then-set-attrs trick (the tracer-test pattern, because the real `__init__`
  is folder-glob-driven and can't take an in-memory per-(cell,day) tif). We confine this to a
  single shim `src/inference/_loader_bridge.py::masked_output_for_tif(tif)` so that if the
  loader's private attrs ever shift, we fix our shim — **never** the loader. No monkeypatching
  of downstream classes anywhere.

## Resolved design decisions
1. **Mosaic CRS — direct UTM placement (user-confirmed 2026-06-09).** The per-cell grid is
   already EPSG:32611 (PLAN §5 corrected 2026-06-04; cube.yaml `cell_crs`; memory
   `bow-valley-cell-grid-utm-not-4326`). The 10×10 FSC patch is therefore already UTM 11N.
   The mosaic places each patch into the daily mosaic grid by metric bounds — **no
   cross-CRS warp**. SPEC FR-22 / AC-29 wording "reproject from EPSG:4326 with NN" is stale
   (predates the CRS correction); NN is still the only resampling permitted *if* a grid-snap
   is ever needed, but with aligned 100 m cells placement is exact. Note added to TASK-015.
2. **Model is injected, not built here.** The driver takes a ready `EncoderWithHead` (and
   `device`). Checkpoint construction is TASK-016's entry-point script. Tests pass a tiny
   untrained encoder (the tracer-test pattern) so they need no checkpoint and no GPU.
3. **Per-cell FSC patch = 10×10 at 100 m** inside a 1 km cell (`100 m × 10 = 1 km`). The
   daily mosaic pixel size is 100 m. Cell bounds come from `GridCell.polygon.bounds` (UTM).
4. **Reuse the existing loader inference path READ-ONLY via a shim.** `LandsatEvalDataset`
   `split="inference"` + `__getitem__` produces the `MaskedOutput`; `EncoderWithHead.forward(...
   patch_size_high_res=10, patch_size_med_res=1, patch_size_low_res=1)` returns `(B,100,1)`;
   squeeze→reshape `(10,10)`. This is exactly `test_tracer_end_to_end.py` and
   `_predict_and_store_output`. We call these unchanged through `_loader_bridge.py` — we add
   no methods to and edit no lines of `src/fsc/`.

## Files (ALL net-new; no existing file edited)

### `src/inference/_loader_bridge.py` (NEW — isolates the sole downstream coupling)
- `masked_output_for_tif(tif: Path) -> MaskedOutput`: builds the `__new__`-bypassed
  `LandsatEvalDataset` inference instance (tracer pattern), sets only the attrs `__getitem__`
  reads, returns `ds[0][0]`. The **one** place that knows the loader's private surface.
  Calls the loader strictly as-is. If the loader changes, only this file changes.

### `src/inference/windows.py`
- `inference_days(window_start, window_end) -> list[date]` — every day inclusive (mirror
  `grid._window_days`, but public for the driver).
- `eight_day_window(window_end) -> list[date]` — `[window_end-7 … window_end]` ascending
  (mirror `exporter._window_days`; reuse `NUM_TIMESTEPS`, `DAYS_PER_TIMESTEP`).

### `src/inference/mosaic.py` — `DailyMosaicWriter`
- `__init__(*, grid: list[GridCell], out_dir: Path, fsc_px_per_cell=10)`.
- Computes the AOI mosaic grid once: union of all cell UTM bounds → mosaic transform at
  100 m px (`cell_size_m / fsc_px_per_cell`), shape covering every cell. EPSG:32611.
- `write_day(day, fsc_by_cell: dict[cell_id, np.ndarray|None]) -> Path`:
  - Allocate `(mosaic_h, mosaic_w)` float32 filled `nodata` (-9999).
  - For each cell with a non-None 10×10 FSC, compute its integer pixel offset from the cell's
    UTM top-left vs the mosaic top-left (exact — cells are 1 km on a 100 m grid), place the
    block. **No double-write**: cells are non-overlapping (`grid.py` guarantees), so each
    target block is disjoint — assert no target pixel is overwritten (AC-29 seam guard).
  - Write a COG (rasterio, `driver=GTiff` + `build_overviews` + `copy` to COG, or
    `driver="COG"` if available) in EPSG:32611, nodata=-9999.
  - Tag `aoi_coverage_fraction` = valid mosaic px / total in-AOI px (AC-28).
- All-masked cell → its FSC is `None` → stays nodata (AC-28).

### `src/inference/driver.py` — `InferenceGridDriver`
- `__init__(*, exporter: LocalSourceExporter, model: EncoderWithHead, grid: list[GridCell],
  window_start, window_end, device="cpu", batch_size=8)`.
- `run() -> list[Path]`: for each `day in inference_days(...)`:
  - For each cell: `tif = exporter.export(cell, window_end=day)`; build a one-tif
    `LandsatEvalDataset` inference instance (the tracer `inference_dataset` fixture pattern)
    → `MaskedOutput`; collect.
  - Batch cells (`batch_size`), run `model(*batched, patch_size_high_res=10, ...)`,
    reshape each row → `(10,10)`; map `cell_id -> fsc` (None if degenerate/all-masked, per
    the existing mask convention).
  - `mosaic.write_day(day, fsc_by_cell)`.
  - **Q4/AC-31:** the loop iterates `inference_days(window_start, window_end) × grid` and
    never reads any CSV `date`. Test asserts two cells with different legacy CSV `date` both
    predicted on the same configured day.
- Parallel per-cell export (`multiprocessing.Pool`) is the production path but is an
  optimization; ship a serial default with a `pool` hook so tests run inline. (Heavy
  parallel/GPU batching detail is fine to keep minimal — TASK-016 wires the real run.)

## Tests (`test_inference_driver.py`) — TDD, synthetic, no GPU/checkpoint
1. **windows**: `eight_day_window(d)` == 8 ascending days ending d; `inference_days` count.
2. **AC-31 (driver loop)**: build a 2-cell grid where the two cells carry *different* legacy
   CSV `date`s (irrelevant — driver takes `GridCell`s + explicit window); assert both are
   predicted for the same configured day and the CSV date never enters the loop. Use a
   fake/tiny `EncoderWithHead` + monkeypatched exporter that returns a placeholder cube, or a
   stub model returning fixed `(B,100,1)` to keep it fast.
3. **AC-29 (seams)**: 2×2 adjacent cells → `DailyMosaicWriter.write_day` places four 10×10
   blocks with no overlapping/double-written pixel; assert the four blocks land at the four
   expected disjoint offsets and total valid px == 4×100.
4. **AC-28 (COG + nodata + coverage)**: output opens as EPSG:32611, all-masked cell →
   nodata block, `aoi_coverage_fraction` tag present and equals valid/total.
5. NN-only: assert mosaic introduces no interpolated (non-source) FSC values — placed values
   are exactly the input patch values.

## Verification
```
uv run pytest tests/test_local_sources/test_inference_driver.py -v
uv run ruff check src/inference/
uv run mypy src/inference/driver.py src/inference/mosaic.py src/inference/windows.py
```
Full-suite delta vs TEST_BASELINE.md — NEW-failures list empty. NOT pytest -x.

## Out of scope (TASK-016)
Entry-point scripts (`scripts/developer_scripts/bow_valley_inference_local/infer_bow_valley_daily_fsc.py`), checkpoint/model build,
full-stack parity gate, real GPU batched run, multiprocessing tuning.
