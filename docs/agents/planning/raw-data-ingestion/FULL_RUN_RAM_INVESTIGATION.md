# Full Mode-B run — RAM peak investigation (root cause: GDAL_CACHEMAX default)

**Date:** 2026-06-17 (during the full Mode-B sweep, day 8 of 21).
**Trigger:** heavy-export-day RAM peaks trended up across the run — 49.5 GB (day 5) →
54 GB (day 8 start) → **57 GB used / 7.2 GB available** (day 8 peak) on a 62 GB host,
with swap creeping 1.4 → 2.4 GB. Every day still reset cleanly to ~6 GB at the
export→inference boundary (no OOM-killer ever fired), but the thinning margin warranted a
root-cause check before mitigating.

## Root cause

**GDAL's per-process block cache is at its default of 5 % of system RAM, never overridden.**

```
GDAL_CACHEMAX (effective) = 3,367,286,988 bytes ≈ 3.37 GB  per process
```

- `GDAL_CACHEMAX` is **not set** anywhere in `src/`, `scripts/`, or `configs/` — so GDAL
  uses its built-in default (5 % of physical RAM). 5 % × 62 GB ≈ 3.1–3.4 GB.
- The export pool runs **16 worker processes**, each with its own GDAL cache:
  **16 × 3.37 GB ≈ 54 GB** of GDAL block cache alone, before Python/array overhead.
- Measured per-worker RSS (4.5–5.2 GB) = ~3.37 GB GDAL cache + ~1–2 GB Python/arrays.
  16 × ~3.5 GB ≈ 54–57 GB total — **matches the observed peak exactly.**

### Why heavy days peak higher (it is NOT a leak)

GDAL fills its block cache **lazily** as raster blocks are read, up to the cap. A light day
(few granules per cell) never fills the 3.37 GB cap; a heavy day (more distinct
JP2/NetCDF granules touched across the worker's ~1,374 cells) reads more blocks, so each
worker's cache fills closer to its full 3.37 GB ceiling → the total climbs toward 54 GB.
The peak is **bounded** (by the cap × workers) and **resets** every day when the pool is
torn down for the inference phase — confirmed 8× — so it is a *too-high ceiling*, not
unbounded growth.

## Why it was invisible until now

- Mode A (~344 cells) and the 48-cell smoke test: few workers / short runs never filled
  the GDAL caches.
- Early full-run days: lighter windows kept per-worker cache below the cap.
- Only a heavy day at full 16-worker width fills enough caches simultaneously to approach
  the 62 GB host limit.

## The fix (permanent)

Set `GDAL_CACHEMAX` to a sane **per-process** value so `workers × cache` has comfortable
headroom. With 16 workers on 62 GB, a 512 MB per-process cache → 16 × 0.5 = 8 GB of GDAL
cache (vs 54 GB), leaving the host overwhelmingly free. JP2/NetCDF decode does not need a
3 GB block cache per process — the windowed reads are small (one cell's neighbourhood).

Options, lowest-effort first:
1. **Env var at launch** (zero code): `GDAL_CACHEMAX=512` (MiB) in the run environment, or
   exported in the driver script before the pool spawns. Each worker inherits it.
2. **Set in `_init_worker`** (`parallel_export.py`): `rasterio.Env(GDAL_CACHEMAX=512)` or
   `osgeo.gdal.SetCacheMax(512 * 1024 * 1024)` once per worker process — explicit, travels
   with the code, not dependent on the launch environment.

Recommended: **(2)** — make it explicit in `_init_worker` so the cap is guaranteed
regardless of how the pool is launched. A follow-up task; not applied mid-run.

## Immediate mitigation (this run)

Restarting the live run with `export_workers: 14` (already in
`inference_full_run.yaml`) drops the peak by 2 × ~3.5 GB ≈ 7 GB → ~48 GB used, restoring a
comfortable margin without touching code. `--cache-policy reuse` makes the restart
cache-fast (the SSD `cube_cache` is warm; only the in-flight day re-exports). This is the
stopgap; the real fix is the `GDAL_CACHEMAX` cap above, which would let 16 workers run
safely (16 × 0.5 GB = 8 GB) and is the better long-term setting.

## Follow-up task

- [ ] Set `GDAL_CACHEMAX` (≈512 MiB) in `_init_worker` (`parallel_export.py`), with a
  comment pointing here. Then `export_workers` can return to 16 (or higher) safely, since
  worker RAM becomes `workers × (0.5 GB cache + ~1.5 GB arrays)` ≈ workers × 2 GB.
- [ ] Optionally also cap it for the inference phase / single-process paths for consistency.
