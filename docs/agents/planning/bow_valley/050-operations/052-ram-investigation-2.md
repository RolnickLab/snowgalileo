# Full Mode-B run — RAM investigation #2: heavy-day worker bloat (malloc arena retention)

*Formerly `FULL_RUN_RAM_INVESTIGATION_2.md`.*

**Date:** 2026-06-17 (full Mode-B sweep, day 13 of 21).
**Supersedes the mechanism in:** `FULL_RUN_RAM_INVESTIGATION.md` (which correctly found the
GDAL_CACHEMAX default issue, but that fix alone did **not** fully bound heavy-day peaks).

## What happened

After fixing GDAL_CACHEMAX (3.37 GB → 512 MB/proc) and dropping to 14 workers, heavy
(Sentinel-2) days still climbed to **57.8 GB used / 3.8 GB available, swap fully
exhausted (8/8 GB)** — the worst margin of the run, with the peak still ahead. A live
snapshot was taken (`scratch/ram_snapshot_183132/`) before restarting at 12 workers.

## The snapshot — what it showed

Per-worker `/proc/PID/status` on the heavy day (14-worker pool, mid-export):

| Worker      | RssAnon | VmSwap  | RssAnon+Swap |
| ----------- | ------- | ------- | ------------ |
| 2754693     | 9.71 GB | 5.75 GB | **~15.4 GB** |
| 2754694     | 9.70 GB | 5.75 GB | ~15.4 GB     |
| 2754695…699 | 9.70 GB | 5.75 GB | ~15.4 GB     |

- **Every worker is ~15 GB anonymous (RssAnon + swapped), near-identical across workers.**
- `smaps_rollup` of the biggest: `Anonymous: 9.7 GB`, `Private_Dirty: 3.65 GB`,
  `Shared_Dirty: 6.08 GB`, `Swap: 5.74 GB`. The memory is **anonymous heap**, not
  file-backed, not GDAL cache (confirmed `GDAL_CACHEMAX=512` in the worker env).
- 12–14 workers × ~15 GB ≫ 64 GB → swap fills, available collapses.

## Root cause: glibc malloc arena retention of peak transient allocations (NOT a leak)

Two measurements pin it down:

1. **No per-cell growth.** On the fresh 12-worker pool (light day), a worker's RssAnon was
   **2,950,092 kB → 2,950,160 kB over 625 cells = 0 MB growth.** Flat. So it is **not** a
   leak that accumulates with cells processed.
2. **Heavy vs light is ~3.3×.** Light day (no S2): ~2.9 GB RssAnon/worker. Heavy day (S2 in
   the 8-day window): ~9.7 GB (+5.7 GB swapped) /worker. The delta (~6.8 GB) is
   **S2-day-specific**.

Mechanism: on a heavy S2 day, assembling a cell's cube transiently allocates large float64
arrays — every S2 band (~13 bands) × multiple granules in the 8-day window, each decoded
from JP2 into a windowed array, coalesced/mosaicked/reprojected. These are **freed** after
each cube (the code is clean — all `with` blocks, no retained instance state), **but glibc
malloc does not return the freed chunks to the OS**: it keeps them in per-thread arenas for
reuse. With the default `MALLOC_ARENA_MAX = 8 × nproc` (= 128 possible arenas) and lazy
trimming, each worker's RSS **plateaus at the high-water mark of its peak transient
allocation** and stays there. Hence: flat per-cell (no growth), but high on heavy days
(big peak transient) — exactly the snapshot.

The GDAL fix was real and necessary (it removed 3.37 GB/proc of block cache), but it does
not touch this: the remaining ~9.7 GB/worker on heavy days is Python/numpy heap retained
by malloc, not GDAL.

## Candidate fixes (for a follow-up task — NOT applied mid-run)

Lowest-effort first; all target the per-worker high-water retention:

1. **`MALLOC_ARENA_MAX=2` (env, zero code).** Caps glibc arenas, drastically cutting
   fragmentation/retention surface across the many transient allocations. The single
   highest-leverage knob for multiprocess numpy workloads; commonly cuts RSS 30–50 %.
2. **`malloc_trim(0)` after each cube** (or every N cells) in the worker — explicitly
   return freed heap to the OS. `ctypes.CDLL("libc.so.6").malloc_trim(0)` after
   `exporter.export()` in `_export_one`. Bounds the high-water mark to roughly one cell's
   transient instead of the run's worst.
3. **Recycle workers periodically** — `ProcessPoolExecutor(max_tasks_per_child=N)` (Py 3.11+)
   so each worker is replaced after N cells, discarding its inflated heap. Simple, robust,
   small re-fork cost.
4. **Combine 1 + 2** for belt-and-suspenders.

Recommended: **start with `MALLOC_ARENA_MAX=2` + `malloc_trim(0)` per cube in
`_export_one`** — both are tiny, env/one-liner changes, and together should let workers
return toward the ~2.9 GB light-day baseline even on heavy days, restoring 16-worker
headroom.

## Immediate mitigation (this run)

Restarted at **12 workers** (`INFER_EXPORT_WORKERS=12`), resuming from day 15 (2025-04-20;
14 valid COGs kept, days 04-06…04-19). 12 × ~9.7 GB heavy-day peak ≈ 116 GB *if unbounded*,
but with swap + the per-day reset it holds; light-day baseline is ~12 × 2.9 ≈ 35 GB. This
is a stopgap — the malloc fixes above are the real solution and would also let the worker
count go back up.

## Correction to investigation #1

`FULL_RUN_RAM_INVESTIGATION.md` attributed the heavy-day peak entirely to GDAL block cache.
That was **one** contributor (~3.37 GB/proc, real, fixed). The **larger** heavy-day
contributor is this malloc-retained Python/numpy heap (~9.7 GB/proc on S2 days), which the
GDAL fix does not address. Both are bounded-but-large, neither is a leak; the heavy/light
split is still driven by S2 presence in the 8-day window.
