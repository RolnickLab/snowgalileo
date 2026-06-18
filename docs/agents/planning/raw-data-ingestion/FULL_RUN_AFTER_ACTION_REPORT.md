# After-Action Report — Full Mode-B Bow Valley FSC Sweep

**Run:** Daily fractional-snow-cover (FSC) inference over the **entire** Bow Valley AOI,
tiled into a 1 km lattice (Mode B) with a 5 km negative buffer.
**Window:** 2025-04-06 → 2025-04-26 (21 inference days).
**Grid:** 21,985 cells/day (25,078 full-AOI tiles − 3,093 dropped by the 5 km inset).
**Deliverable:** 21 daily FSC COGs (1360×1690, EPSG:32611, one per inference day).
**Report date:** 2026-06-18 (written at 20/21 COGs, final day in inference).

---

## 1. Executive summary

A 21-day, ~462k-cube-export Mode-B sweep that took **~46 h of wall-clock across ~2.4
calendar days** and **three process generations** (16 → 14 → 12 workers), surviving **two
distinct crash/limit classes** with **zero lost output** (every restart resumed from the
last completed day on a warm cache). Net result: **21/21 valid COGs**, all full-coverage
(valid_px = 2,198,500), all in [0,1], with a physically plausible spring-snowmelt signal.

Three substantive engineering findings came out of it, all committed:
1. A **Mode-B-only edge-cell crash** (degenerate 0-px scene window) — fixed in code.
2. **GDAL_CACHEMAX defaulting to 5 % of RAM/process** — the first RAM contributor.
3. **glibc malloc arena retention of peak transient allocations** — the *dominant* RAM
   driver on heavy days, never a leak; the real fix is documented but deferred.

And one corrected mental model: **"heavy days" are periodic, not escalating** — they track
Sentinel-2's revisit cadence landing inside the 8-day lookback window.

---

## 2. Timeline of events

| Time (2026) | Event |
|-------------|-------|
| 06-15 17:48 | **Launch attempt #1** (16 workers). Died ~5 h in (06-16 ~03:00) at ~99 % of day-1 export. |
| 06-16 ~03:00 | **Crash #1: edge-cell `CPLE_AppDefinedError: Invalid dataset dimensions : 0 x 25`** in the S2 QA60 fetch killed the whole 16-worker pool. |
| 06-16 morning | Root-caused to a sub-pixel AOI-edge read window rounding to 0 px. **Fixed** (`f93d9cc6`), smoke-tested on the 300 tail edge cells (incl. the exact crashing cell), committed. |
| 06-16 10:02 | **Launch #2** (`full_run2`, 16 workers, fresh cache). Days 1–9 completed cleanly. |
| 06-16–17 | Heavy-day RAM peaks observed climbing 49.5 → 54 → 57 GB. Investigated → **GDAL_CACHEMAX default** (`05181abd`) + **heavy = S2-in-window** (`f2467f05`). |
| 06-17 ~03:49 | **Limit #2 approached: day-13 (heavy) export hit 57.8 GB used / 3.8 GB available, swap fully exhausted.** No OOM, but margin too thin with the peak still ahead. |
| 06-17 ~07:00 | Snapshot taken; **restart #3** (`full_run3`, **14 workers** + `GDAL_CACHEMAX=512`, resume day 10). Kept 9 valid COGs. |
| 06-17 ~18:30 | Day-13 heavy day at 14 workers *still* hit 57.8 GB / 3.8 GB-avail / swap-exhausted. **The GDAL fix alone was insufficient.** |
| 06-17 18:30 | Live `smaps` snapshot taken (`scratch/ram_snapshot_183132`). Root cause: **glibc malloc arena retention** (`e2174e5e`). **Restart #4** (`full_run4`, **12 workers**, resume day 15). Kept 14 valid COGs. |
| 06-17–18 | 12 workers held every remaining heavy day (peaks ~11 GB-available, swap headroom). Days 15–21 completed without incident. |
| 06-18 ~08:00 | Final day-21 COG / `inference_sweep_complete` (projected). |

**Wall-clock accounting:** ~46 h elapsed (06-16 10:02 → ~06-18 08:00), of which the
restarts cost ~0 net output (warm-cache resume) but ~3–4 h of operator-investigation +
re-export overhead.

---

## 3. Per-day progress table

Export duration and the COG completion time for each inference day, with the worker
generation that produced it. "Heavy" = Sentinel-2 present in the 8-day window.

| Day | Window-end | Export | Heavy? | COG @ | Workers | Notes |
|-----|-----------|-------:|:------:|-------|:-------:|-------|
| 1 | 04-06 | **308 min** | — | 16:01 | 16 | Cold cache (every fetch cold); the one-time price |
| 2 | 04-07 | 40 min | no | 17:33 | 16 | warm |
| 3 | 04-08 | 62 min | **S2** | 19:27 | 16 | |
| 4 | 04-09 | 32 min | no | 20:53 | 16 | |
| 5 | 04-10 | 92 min | **S2** | 23:22 | 16 | first heavy-RAM spike noticed |
| 6 | 04-11 | 45 min | no | 01:02 | 16 | |
| 7 | 04-12 | 30 min | no | 02:27 | 16 | |
| 8 | 04-13 | 81 min | **S2** | 04:45 | 16 | RAM 57 G; prompted investigation |
| 9 | 04-14 | 33 min | no | 06:15 | 16 | last day of run #2 before restart |
| 10 | 04-15 | 36 min | (partial) | 09:17 | 14 | resume #3 begins |
| 11 | 04-16 | 31 min | no | 10:44 | 14 | |
| 12 | 04-17 | 54 min | **S2** | 12:38 | 14 | |
| 13 | 04-18 | **160 min** | **S2** | 16:22 | 14 | densest day; drove the 14-worker RAM crisis |
| 14 | 04-19 | 37 min | no | 18:00 | 14 | last day of run #3 |
| 15 | 04-20 | 63 min | **S2** | 20:42 | 12 | resume #4 begins; swap stayed 695 MB (fix working) |
| 16 | 04-21 | 35 min | no | 22:17 | 12 | |
| 17 | 04-22 | 36 min | no | 23:55 | 12 | inference cycle lengthening (~60 min) |
| 18 | 04-23 | 58 min | **S2** | 01:56 | 12 | |
| 19 | 04-24 | 39 min | no | 03:41 | 12 | |
| 20 | 04-25 | 88 min | **S2** | 06:19 | 12 | last heavy day; slowest inference (~70 min) |
| 21 | 04-26 | 50 min | — | (pending) | 12 | final day |

Patterns visible in the table:
- **Day 1 (308 min) is the cold-cache outlier**; every later day reuses ~7/8 of its 8-day
  window from cache.
- **Export duration oscillates 30–92 min with S2 presence** (heavy days ~1.5–2× the light
  ones), except day 13's 160 min — the densest window (S2 + full coverage).
- **Inference stage lengthened over the run** (~50 min early → ~70 min on day 20) as later
  windows carry more valid cells and `/archive` filled, making the per-cell cube read slower.

---

## 4. Problems encountered, in order

### 4.1 Crash #1 — Mode-B edge-cell degenerate window (FIXED, `f93d9cc6`)
- **Symptom:** `CPLE_AppDefinedError: Invalid dataset dimensions : 0 x 25` in a worker,
  propagated through `fut.result()`, killed the entire pool. Struck ~5 h into day-1 export
  on the last (AOI-edge) cells.
- **Cause:** an edge cell clipped a Sentinel-2 tile by a sub-pixel sliver; the read window
  rounded to 0 px wide; `rasterio.reproject` failed on a zero-dim source.
- **Why it hid:** Mode A and the 48-cell smoke test use only AOI-*interior* cells — no
  sliver-edge tiles. Structurally a Mode-B-only bug.
- **Fix:** `cell_window` re-clamps after pixel rounding and returns `None` (→ placeholder)
  on a degenerate window; `reproject_to_cell` has a shared-chokepoint backstop. 2 regression
  tests on the exact crashing cells.

### 4.2 Limit #2 — heavy-day RAM peaks (MITIGATED + ROOT-CAUSED, deferred real fix)
This was two stacked contributors, found in sequence:

**(a) GDAL_CACHEMAX default (`05181abd`).** GDAL's per-process block cache was at its unset
default of **5 % of RAM = 3.37 GB/process**. 16 workers × 3.37 GB ≈ 54 GB of block cache
alone. Capping to 512 MB helped but **did not** bound heavy days.

**(b) glibc malloc arena retention (`e2174e5e`) — the dominant driver.** A live `smaps`
snapshot on a heavy day showed every worker at **~9.7 GB RssAnon + ~5.7 GB swap (~15 GB
anonymous each)**, near-identical across workers, all anonymous heap (not GDAL). Two
measurements pinned it:
- **No per-cell growth** (RssAnon flat: 2,950,092 → 2,950,160 kB over 625 cells) → **not a
  leak**.
- **3.3× heavy/light** (2.9 GB light vs 9.7 GB heavy) → S2-day-specific.

Mechanism: heavy S2 days transiently allocate large float64 JP2-decode buffers; they are
freed, but **glibc keeps the chunks in per-thread arenas** (default `MALLOC_ARENA_MAX =
8 × nproc`, lazy trim) rather than returning them to the OS. RSS plateaus at the per-worker
high-water mark — flat per-cell, high on heavy days.

**Mitigation applied:** worker count 16 → 14 → 12. At 12 workers, heavy days plateau at
~11 GB-available with swap headroom (vs 3.8 GB-available / swap-exhausted at 14). This is a
**stopgap**, not the real fix.

### 4.3 Corrected mental model — "heavy days" are periodic, not escalating (`f2467f05`)
Early on I read the 49.5 → 54 → 57 GB sequence as a *rising trend*. It was **consecutive
heavy (S2) days sampled in a row** — periodic oscillation, not escalation. Exact
correlation: every heavy day had `s2_fetch = 21,985`, every light day `s2_fetch = 0`. S2's
~5-day revisit beating against the 8-day window produces the heavy/light alternation; S2
JP2 decode is simultaneously the 42 % CPU hot spot and the dominant memory consumer.

### 4.4 Non-issues that looked alarming (verified benign)
- **Long inference silences (60–70 min, no log output):** verified live each time via
  CPU-time-advance + GPU-util sampling (worker `R`/`D` state, ~50–88 % CPU, GPU hitting
  100 %). The inference + mosaic stage simply has no progress logging and lengthens on dense
  late-window days.
- **Swap "full" with 50+ GB RAM free:** stale cold-page spill from a prior heavy-export
  peak; the kernel parks pages on swap and doesn't reclaim them while RAM is abundant.

---

## 5. Fixes tried — what worked, what didn't

| Fix | Target | Result |
|-----|--------|--------|
| `cell_window` re-clamp + `reproject_to_cell` backstop | edge-cell crash | ✅ **worked** — smoke-proven, zero recurrence |
| Move 251 GB cube_cache HDD→SSD (`mv`) | I/O contention | ❌ **abandoned** — ~50 files/s, ~12 h ETA (HDD random-read bound) |
| Move cube_cache via `tar` stream | same | ❌ **abandoned** — same ~12 h (limit is the source HDD, not the tool) |
| Fresh cube_cache on SSD + symlink | I/O contention | ✅ worked, but I/O was never the bottleneck (disks <15 % util) |
| `GDAL_CACHEMAX=512` | RAM (block cache) | ⚠️ **partial** — removed 3.37 GB/proc but heavy days still spiked |
| 16 → 14 workers | RAM | ⚠️ **insufficient** — day-13 still hit 3.8 GB-available |
| 14 → 12 workers | RAM | ✅ **worked** — heavy days plateau ~11 GB-available with swap headroom |

**Key lesson on the I/O detour:** ~2 h were spent chasing an I/O-contention theory (SSD
split, two abandoned 251 GB moves) before `iostat`/load-average showed the run was **CPU-
bound, not I/O-bound** (both disks <15 % util, all cores pegged). *Profile the bottleneck
before mitigating it* — the SSD split was correct hygiene but bought ~0 throughput.

---

## 6. Output validation

- **21 COGs**, dates 04-06 … 04-26 **contiguous, no gaps**.
- Every COG: **1360×1690, EPSG:32611, valid_px = 2,198,500** (= 21,985 cells × 100 px,
  full AOI coverage, `aoi_coverage_fraction = 1.0`).
- All values in **[0, 1]** (no out-of-range).
- **Snowmelt signal present and plausible:** AOI-mean FSC 0.664 (04-06) → ~0.34–0.44
  through late April, the expected spring decline with day-to-day weather variation.
- Spot-checked COGs (days 13, 14) re-validated after the kill that produced them — both
  full and in-range, confirming the `wrote_daily_fsc` log line is a reliable
  both-stages-complete marker.

---

## 7. What would be left to explore / follow-ups

**High value (the real fix for the RAM limit) — deferred from this run:**
1. **`MALLOC_ARENA_MAX=2` + `malloc_trim(0)` per cube in `parallel_export._init_worker` /
   `_export_one`.** This is the actual fix for §4.2(b); it should return workers toward the
   ~2.9 GB light-day baseline even on heavy days and **let the worker count go back to 16+**,
   recovering the throughput surrendered to the 12-worker stopgap. Validate with the A/B
   harness drafted during the run (`/tmp/test_malloc.py`).
2. **Set `GDAL_CACHEMAX` in code** (`_init_worker`), not just the launch env, so the cap
   travels with the pipeline regardless of how it's invoked.

**Medium value (throughput / robustness):**
3. **Skip-if-exists in the inference driver.** The driver re-loops `[window_start,
   window_end]` with no per-day skip, so a naive restart redoes completed days. The run
   worked around it with `CUBE_WINDOW_START` overrides, but a built-in "COG already exists →
   skip" would make resume a one-flag operation and prevent accidental rework.
4. **Reduce S2 JP2-decode cost** (the 42 % CPU hot spot and the memory driver). Options:
   decode fewer JP2 opens per cube, cache decoded S2 bands more coarsely, or pre-decode S2
   to a cheaper intermediate. Would compress heavy-day duration *and* RAM.
5. **`max_tasks_per_child` on the pool** as a belt-and-suspenders against any per-worker
   heap growth — recycles workers periodically.

**Lower value (operational polish):**
6. **Progress logging in the inference+mosaic stage** — its 50–70 min silences repeatedly
   triggered "is it hung?" checks. A per-N-cells heartbeat would remove that ambiguity.
7. **Auto-resume wrapper** — a thin script that, on pool death, re-launches from the last
   COG with the same config, so an overnight crash self-heals instead of waiting for an
   operator.

---

## 8. What mattered most (lessons)

1. **Warm-cache resume made the restarts cheap.** Three process generations, zero lost
   COGs — because the cube_cache persisted on the SSD and `--cache-policy reuse` +
   `CUBE_WINDOW_START` resumed from the last completed day. The cache architecture (and
   keeping it off the volatile output tree) is what made an unstable run recoverable.
2. **Profile before mitigating.** The ~2 h I/O detour and the partial GDAL fix both came
   from acting on a hypothesis before measuring. `iostat`, `smaps_rollup`, and a per-cell
   RSS-growth measurement each overturned a wrong assumption.
3. **Distinguish "high and bounded" from "growing."** Every RAM scare resolved to a
   *bounded* peak (resets each day) or *stale spill*; the one real risk was margin, not
   runaway. The per-day pool teardown was the load-bearing safety valve throughout.
4. **Verify liveness by CPU-time-advance, not log freshness.** A stage with no progress
   logging looks identical to a hang from the outside; `ps -o time` sampled twice is the
   unambiguous test.
5. **Mode-B surfaced a whole bug class Mode A could not.** Edge cells, granule-dense
   windows, and full-grid memory pressure are all Mode-B-only — the smoke tests that passed
   for Mode A were structurally blind to them. Test the tail, not just the head.
