# Bow Valley Direct-Source — Task Index

Decomposed from `FDD_BOW_VALLEY_DATA.md` §4 (8 RGR steps), with the adapter step
expanded into 9 per-modality tasks per `PLAN_BOW_VALLEY_DATA.md` §7 Phase 3 Step 3.
Sourced from PLAN, FDD, SPEC, `DATA_ANALYSIS.md`, `docs/agents/KNOWLEDGE.md`.

**Ordering is strict dependency order.** Each task is a vertical slice (interface →
implementation → test) and follows Red → Green → Refactor. Do not start a task before
its predecessors are approved (per CLAUDE.md: stop and get sign-off at each gate).

> **The test suite is already red on a clean checkout (6 pre-existing failures).** See
> [`test-baseline.md`](test-baseline.md). Every task validates against the **delta** —
> a task passes when it introduces **no new** failures vs that baseline and its own new
> tests are green. Never use `pytest -x` at the suite level (it halts on a pre-existing
> failure before reaching new tests); never try to fix the baseline failures as part of
> Bow Valley work.

| Task     | Title                                                      | FDD step | Key SPEC ACs                      |
| -------- | ---------------------------------------------------------- | -------- | --------------------------------- |
| TASK-001 | Phase 0 audit + generated cube CSV + GEE reference patches | §4.1     | AC-10, AC-11, AC-11b, AC-30       |
| TASK-002 | AOI clip stage (Phase 0.5) — **approval gate**             | §4.2     | AC-1…AC-8                         |
| TASK-003 | Contract: base.py, grid.py, layout.py, cube_cache.py       | §4.3     | AC-9, AC-10, AC-11, AC-11b        |
| TASK-004 | Placeholder exporter + tracer-bullet test                  | §4.4     | AC-13, AC-23, AC-24, AC-25, AC-26 |
| TASK-005 | S1/S2 parity spikes — **go/no-go gate**                    | §4.5     | AC-14, AC-15 (spike form)         |
| TASK-006 | ESA WorldCover adapter                                     | §4.6 #1  | AC-12, AC-22                      |
| TASK-007 | Copernicus DEM adapter                                     | §4.6 #2  | AC-12, AC-21                      |
| TASK-008 | ERA5-Land adapter                                          | §4.6 #3  | AC-12, AC-13, AC-20               |
| TASK-009 | MODIS MOD09GA adapter (preserve -28672)                    | §4.6 #4  | AC-12, AC-13, AC-18               |
| TASK-010 | VIIRS VNP09GA adapter (fine + coarse per-pixel)            | §4.6 #5  | AC-12, AC-13, AC-19               |
| TASK-011 | Sentinel-3 OLCI adapter (tie-point geolocation)            | §4.6 #6  | AC-12, AC-13, AC-17               |
| TASK-012 | Landsat 8/9 adapter (L9→L8, cross-zone, coalesce)          | §4.6 #7  | AC-12, AC-13, AC-15b, AC-16       |
| TASK-013 | Sentinel-2 adapter (−1000 DN, coalesce)                    | §4.6 #8  | AC-12, AC-13, AC-15, AC-15b       |
| TASK-014 | Sentinel-1 GRD adapter (edge mask, windowed reads)         | §4.6 #9  | AC-12, AC-13, AC-14               |
| TASK-015 | InferenceGridDriver + DailyMosaicWriter                    | §4.7     | AC-28, AC-29, AC-31               |
| TASK-016 | Entry-point scripts + directory contract + parity gate     | §4.8     | AC-27, AC-32                      |

**SPEC AC coverage:** AC-1…AC-32 are all mapped above (AC-12 recurs across every
adapter task as the golden-grid contract; AC-13 recurs for every time-varying adapter).

## Cross-cutting notes carried into the tasks

- **CSV semantics (Q4 RESOLVED):** the cell/date input is the **generated cross-product
  CSV** (`configs/bow_valley/cube_cells.csv`), not the legacy training CSV. The driver
  ignores the CSV `date`; it sweeps the configured window × all in-AOI cells. (TASK-001,
  TASK-003, TASK-015.) See memory `bow-valley-inference-csv-decision`.
- **Filename discrepancy to resolve in TASK-004:** `export_from_csv_utm` emits
  `PR_{date}_{cx}_{cy}.tif` (UTM coords, 3 fields) for GEE fixtures, but the
  `LocalSourceExporter` contract is `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (5 fields,
  degrees). Both parse via the `PR` branch (`landsat_eval.py:171-176`).
- **Two non-negotiable scene-source rules** (§9): cross-tile mosaic-before-crop AND
  same-(tile,date) valid-pixel coalesce (NOT averaging). Contract in `base.py` (TASK-003);
  first production use Landsat (TASK-012), then S2 (TASK-013).
- **Write boundaries:** clip stage writes only `clipped_bow_valley_selection_raw`; all
  Stage 2 writes go under `bow_valley_processing/` subdirs. Enforced by AC-32 (TASK-016).

## Open questions still gating a production run (not blocking task implementation)

- Q3 sweep mode A vs B · Q5 output destination · Q6 checkpoint path · Q7 GPU budget.
