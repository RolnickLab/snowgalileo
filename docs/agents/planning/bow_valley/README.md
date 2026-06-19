# Bow Valley Direct-Source Inference — Planning Index

**Start here.** This directory holds every planning, design, task, and operations
document for the Bow Valley direct-source data cube and daily fractional-snow-cover
(FSC) inference pipeline — the local-archive pipeline that supplements Google Earth
Engine ingestion. Documents are grouped into numbered phase folders that follow the
work in the order it happened: analysis → design → ingestion → cube cache → inference →
operations, plus a cross-cutting QA viewer.

For *what is done vs. in-flight vs. deferred*, see [`STATUS.md`](STATUS.md).

> Files keep a `*Formerly `OLD_NAME.md`.*` line under their title so older references
> (e.g. a docstring citing `CLIPPING_PLAN §2.0`) remain greppable after the 2026-06-18
> reorganization.

## Timeline at a glance

```
000 Analysis ─▶ 010 Design ─▶ 020 Data ingestion ─▶ 030 Cube cache ─▶ inference ─▶ 050 Operations
   (audit the      (PLAN →       (clip, parity,         (wiring,         tasks         (full Mode-B
    archive &        FDD →         per-modality           invalidation,   015/016        run AAR,
    GEE dataset)     SPEC →        adapters,              day-eviction)                  RAM deep-dives)
                     review)       tasks 001–016)
                                                                          060 Viewer (QA tool, cross-cutting)
```

## Documents in reading order

### 000 — Analysis (understand the inputs)

| Doc                                                                                        | What it covers                                                                         |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| [000-analysis/000-gee-dataset-analysis.md](000-analysis/000-gee-dataset-analysis.md)       | How the GEE-export dataset is built and how ingestion scales to a larger region.       |
| [000-analysis/001-data-ingestion-analysis.md](000-analysis/001-data-ingestion-analysis.md) | The tensor contract and source-by-source value domains (the golden grid).              |
| [000-analysis/002-archive-audit.md](000-analysis/002-archive-audit.md)                     | Phase-0 catalog of the raw archive, cube CSV, coverage, S1 sparsity (TASK-001 output). |

### 010 — Design (the design of record)

| Doc                                                              | What it covers                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------------- |
| [010-design/010-plan.md](010-design/010-plan.md)                 | The plan — goal, architecture, FMEA, phasing.                 |
| [010-design/011-fdd.md](010-design/011-fdd.md)                   | **Formal Design Document — the design of record.**            |
| [010-design/012-spec.md](010-design/012-spec.md)                 | SPEC — acceptance criteria (AC-1…AC-32) derived from the FDD. |
| [010-design/013-review-audit.md](010-design/013-review-audit.md) | Hyper-critical technical/geospatial review of the plans.      |

### 020 — Data ingestion (clip, parity, adapters, tasks)

| Doc                                                                                          | What it covers                                                                          |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [020-data-ingestion/020-clipping-plan.md](020-data-ingestion/020-clipping-plan.md)           | Non-destructive AOI clip of every raw source (the two-stage intersect gate).            |
| [020-data-ingestion/021-parity-spike-notes.md](020-data-ingestion/021-parity-spike-notes.md) | S1/S2 parity de-risking spikes against GEE reference patches (TASK-005).                |
| [020-data-ingestion/022-s1-pergranule-snap.md](020-data-ingestion/022-s1-pergranule-snap.md) | Sentinel-1 per-granule ESA SNAP processing (process-then-clip).                         |
| [020-data-ingestion/tasks/](020-data-ingestion/tasks/)                                       | **The 16 implementation tasks** — see its [README](020-data-ingestion/tasks/README.md). |

The tasks are grouped by phase under `tasks/`:

- `foundation/` — TASK-001…005 (audit, clip, contract/grid, tracer, parity gate)
- `adapters-static/` — TASK-006…010 (WorldCover, DEM, ERA5, MODIS, VIIRS)
- `adapters-scene/` — TASK-011…014 (+012b/013b/013c) (S3, Landsat, S2, S1)
- `inference/` — TASK-015, 016 (driver + mosaic; entry points + parity gate)

### 030 — Cube cache (per-(modality, cell, day) `.npz` cache)

| Doc                                                                      | What it covers                                             |
| ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| [030-cube-cache/030-wiring.md](030-cube-cache/030-wiring.md)             | Wiring the CubeCache into the exporter.                    |
| [030-cube-cache/031-invalidation.md](030-cube-cache/031-invalidation.md) | Version-stamp invalidation + interactive overwrite policy. |
| [030-cube-cache/032-day-eviction.md](030-cube-cache/032-day-eviction.md) | Day-frontier eviction for Mode-B scale.                    |

### 050 — Operations (running the full sweep)

| Doc                                                                                                      | What it covers                                                                                        |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [050-operations/050-full-run-after-action-report.md](050-operations/050-full-run-after-action-report.md) | **Full Mode-B sweep AAR** — timeline, per-day table, problems/fixes, the diagonal-seam analysis (§9). |
| [050-operations/051-ram-investigation.md](050-operations/051-ram-investigation.md)                       | RAM peak root cause #1 — GDAL_CACHEMAX default.                                                       |
| [050-operations/052-ram-investigation-2.md](050-operations/052-ram-investigation-2.md)                   | RAM root cause #2 — glibc malloc arena retention (the dominant driver).                               |

### 060 — Viewer (cross-cutting QA tool)

| Doc                                                                | What it covers                            |
| ------------------------------------------------------------------ | ----------------------------------------- |
| [060-viewer/060-viewer-plan.md](060-viewer/060-viewer-plan.md)     | Clipped-archive visual validation viewer. |
| [060-viewer/061-cube-fsc-tabs.md](060-viewer/061-cube-fsc-tabs.md) | v2 — cube + daily-FSC tabs.               |
| [060-viewer/062-contract.md](060-viewer/062-contract.md)           | Per-modality renderer contract.           |
| [060-viewer/spike_reads.py](060-viewer/spike_reads.py)             | Read-path spike script.                   |
