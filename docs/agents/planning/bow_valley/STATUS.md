# Bow Valley Inference — Status

Snapshot of what is done, in-flight, and deferred across the planning phases.
Last updated: **2026-06-18**. See [`README.md`](README.md) for the document index.

## ✅ Done

- **Design of record complete** — PLAN → FDD → SPEC → review audit (phase 10).
- **AOI clip stage** — non-destructive clip of every raw source with the two-stage
  intersect gate (TASK-002 / `20-clipping-plan.md`).
- **All 9 source adapters** — WorldCover, DEM, ERA5, MODIS, VIIRS, S3, Landsat, S2, S1
  (TASK-006…014), each parity-validated against GEE reference patches.
- **Cube cache** — wiring, version-stamp invalidation, day-frontier eviction
  (phase 30), proven at Mode-B scale.
- **Inference driver + daily mosaic** — TASK-015/016; direct-UTM pixel-offset mosaic.
- **Full Mode-B sweep** — 21 daily FSC COGs (2025-04-06…04-26), full AOI coverage,
  validated. See [`50-operations/50-full-run-after-action-report.md`](050-operations/050-full-run-after-action-report.md).

## 🔬 Analyzed, fix not yet applied

- **Diagonal acquisition seams in the FSC output** (AAR §9). Root-caused to three
  co-primary footprint-seam drivers — **Landsat WRS-2 path** + **Sentinel-2 R070/R113
  orbit** value-steps and **Sentinel-1 swath coverage edges** (S3 weaker). Mitigation
  tiers proposed (seam-mask offset removal → cross-path normalization + feather →
  overlapping inference grids); none implemented.
- **Heavy-day RAM ceiling** (AAR §4.2). The run shipped on a 12-worker stopgap; the
  malloc-arena root cause is understood but the real fix is deferred (below).

## ⏳ Deferred follow-ups (not started — need explicit go-ahead)

From the AAR §7, highest-leverage first:

1. **`MALLOC_ARENA_MAX=2` + `malloc_trim(0)` per cube** in `parallel_export` — the real
   RAM fix; should restore the light-day baseline on heavy days and let workers return
   to 16+. (A/B harness drafted during the run.)
2. **Set `GDAL_CACHEMAX` in code** (`_init_worker`), not just the launch env.
3. **Diagonal-seam mitigation** — cross-path/orbit normalization + feathered blend for
   Landsat/S2, average+feather for S1; eventually overlapping grids + averaging (also
   kills the 1 km grid outline).
4. **Skip-if-exists in the inference driver** — make resume a one-flag operation.
5. **Reduce S2 JP2-decode cost** — the 42 % CPU hot spot and a memory driver.
6. **`max_tasks_per_child`** on the pool; **inference-stage progress logging**;
   **auto-resume wrapper**.

## ⚠️ Standing baseline note

The test suite is **already red on a clean checkout** (6 pre-existing failures). Judge
new work by the **delta**, never `pytest -x` at the suite level. See
[`20-data-ingestion/tasks/test-baseline.md`](020-data-ingestion/tasks/test-baseline.md).
