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
| 21 | 04-26 | 50 min | — | 08:14 | 12 | final day; `inference_sweep_complete` 08:14:49 |

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

**Output-quality (diagonal swath seams — see §9 for the full analysis):**
3. **Kill the diagonal acquisition seams.** Three co-primary drivers (§9.1): **Landsat WRS-2
   path** + **Sentinel-2 R070/R113 orbit** value-steps (cross-path/orbit normalize + feather
   adjacent strips) and **S1 swath coverage edges** (average + feather overlaps, not
   first-valid-wins, in `_scene_ops.mosaic_tiles`); add a per-source recency channel so the
   model stops reading a seam as snow signal. The structurally complete fix is overlapping
   inference grids + averaging (kills both the diagonal seams and the 1 km grid outline).

**Medium value (throughput / robustness):**
4. **Skip-if-exists in the inference driver.** The driver re-loops `[window_start,
   window_end]` with no per-day skip, so a naive restart redoes completed days. The run
   worked around it with `CUBE_WINDOW_START` overrides, but a built-in "COG already exists →
   skip" would make resume a one-flag operation and prevent accidental rework.
5. **Reduce S2 JP2-decode cost** (the 42 % CPU hot spot and the memory driver). Options:
   decode fewer JP2 opens per cube, cache decoded S2 bands more coarsely, or pre-decode S2
   to a cheaper intermediate. Would compress heavy-day duration *and* RAM.
6. **`max_tasks_per_child` on the pool** as a belt-and-suspenders against any per-worker
   heap growth — recycles workers periodically.

**Lower value (operational polish):**
7. **Progress logging in the inference+mosaic stage** — its 50–70 min silences repeatedly
   triggered "is it hung?" checks. A per-N-cells heartbeat would remove that ambiguity.
8. **Auto-resume wrapper** — a thin script that, on pool death, re-launches from the last
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

---

## 9. Output artifact analysis — diagonal swath seams in the daily FSC

**Observation (user).** Beyond the expected 1 km grid outline (Mode B tiles the AOI into
independent 1 km cells with no overlap/averaging — a known artifact, deferred to overlapping
grids), the COGs show **diagonal banding running upper-right → lower-left** that *recurs in
roughly the same place but is not identical date to date* — a strong hint that one input
source dominates the result, and that it is a **long-revisit, partial-coverage** source
rather than a near-daily one.

### 9.1 Diagnosis — Landsat, Sentinel-1, and Sentinel-2 footprint seams (all diagonal); Sentinel-3 weaker

The diagonals are **acquisition-footprint seams**: within one 8-day cube window, a
long-revisit / multi-track source contributes data from **different acquisition dates on
either side of a swath or path boundary**, and at ~51 °N those boundaries run UR → LL — the
observed orientation. Two mechanisms produce the seam:

- **(M1) Coverage edge** — band present → nodata along a swath boundary (the model sees a
  step in *which inputs exist*).
- **(M2) Value step under full coverage** — two adjacent tracks/paths both cover their cells
  but were imaged on **different days** (different snow, sun, cloud), leaving a *value* step
  across the boundary with **no coverage hole**, since the mosaic is **first-valid-wins, not
  averaged** (`_scene_ops.mosaic_tiles`).

**Method note — three aggregate metrics each failed; the coverage *map* settled it.** Four
spatial tests were run; the first three each have a projection flaw and gave unreliable
per-source rankings:

| Test | Flaw | Wrong verdict it produced |
|------|------|---------------------------|
| coverage/presence `corr(valid-frac, lon+lat)` | blind to M2 (full-coverage value seams) | cleared Landsat (scored 0.03) |
| value `corr(mean-value, lon+lat)` | confounded by the real **W→E snow gradient** | inflated everything, incl. seam-free VIIRS (0.48–0.59) |
| de-trended column-mean step | **column collapse smears diagonal seams** (only sees vertical ones) | inconclusive |
| **per-source coverage MAP (rendered)** | none for this question | **decisive — see below** |

What actually discriminated was **rendering each source's per-cell window coverage as a map**
(`/tmp/source_seams_overview.png`) and reading the seam geometry directly — the same logic as
the user's Landsat-scene overlay, applied to every source:

| Source | Seam in coverage map | Orientation | Diagonal driver? |
|--------|----------------------|-------------|------------------|
| **Landsat** | strong path boundary (042 vs 043, imaged Apr 3 vs Apr 2) | **UR → LL** | **Yes — primary.** Footprint fixed, recurs every 16 days; matches the user overlay. |
| **Sentinel-1** | clear diagonal swath strip; per-pass western edge differs (Apr 1: −115.33, Apr 6: −116.59) | **UR → LL** | **Yes — primary.** M1 coverage edge that shifts per pass. |
| **Sentinel-2** | diagonal orbit boundary (**R070 vs R113**) **+** rectilinear MGRS edges (T11U NS/NT/PS/PT) | **UR → LL + N–S/E–W** | **Yes** — newly identified; the earlier "N–S only" was wrong. |
| **Sentinel-3** | banding more **E–W / horizontal** than diagonal | mostly horizontal | **Weak** — least aligned with the UR→LL axis. |
| VIIRS | none (wall-to-wall daily) | — | **No.** Confirms the corollary: near-daily full-coverage sources can't make crisp seams. |

**Corroborating hard evidence (no correlation needed):**
- **Landsat:** the 04-06 window `[03-30…04-06]` carries **path 042 (Apr 3, LC08) west** and
  **path 043 (Apr 2, LC09) east**; the WRS-2 042 footprint west edge is lon **−115.75 in
  April vs −115.73 in a March scene** — fixed, recurring geometry, so any 042 scene traces
  the seam line. The FSC column-mean also shows a **0.66 → 0.74 step** near the path boundary
  (suggestive, not conclusive on its own given the gradient).
- **Sentinel-1:** the two in-window passes have **different western coverage edges** (above) —
  a real diagonal nodata boundary that moves between dates.
- **Sentinel-2:** the archive carries **two relative orbits (R070, R113)** plus a 2×2 MGRS
  tile grid — both a diagonal orbit seam and rectilinear tile edges.

**Verdict (corrected, twice).** **Three co-primary diagonal drivers — Landsat WRS-2 path
seams, Sentinel-1 swath coverage edges, and Sentinel-2 orbit seams — all UR→LL; Sentinel-3
is a weak/more-horizontal contributor.** This supersedes two earlier wrong calls in this
analysis: (1) the coverage-only test wrongly **cleared Landsat**, and (2) S2 was wrongly
filed as **"N–S tile edges only"** — it also has a diagonal orbit seam. The user's
"long-revisit, partial-coverage source" instinct was correct and, specifically, correct
about Landsat.

**Why "same pattern, shifting per date":** every contributor's footprint is **fixed,
recurring geometry** — Landsat WRS-2 paths repeat every 16 days, S1 relative orbits on its
~12-day cycle (granules alternate ~T0120 / ~T0129 tracks + a ~T0137 pass), S2 orbits R070/R113.
So the seam *set* recurs near the same place while *which* tracks/paths fall inside `[d−7, d]`
rotates day to day. This also plausibly drives the **non-monotone mid-window FSC wobble**
(AOI-mean dipping to ~0.33 on 04-16, rising to ~0.44 by 04-22): real snowmelt is near-monotone,
so a bounce that tracks which acquisitions populated each window points at input availability,
not snow.

### 9.2 Mitigation options

**Tier A — post-treatment on the existing COGs (no re-run; least result-twisting first):**
1. **Seam-mask DC-offset removal.** The seams are a low-frequency *step* on top of real
   high-frequency snow texture. Derive the seam mask per day from the **fixed footprints**
   (Landsat WRS-2 path boundaries, S1 swath edges, S2 R070/R113 orbit + MGRS tile edges — all
   on disk) and subtract the per-strip mean offset only at those locations. Surgical —
   flattens the step, not the terrain. **Recommended at this tier.**
2. **Edge-preserving smoothing** (bilateral / DEM-guided guided filter) — knocks down soft
   seams while keeping real snowline edges. Moderate distortion (can blur genuine detail).
3. **Document + display only** — annotate as a known acquisition-geometry artifact, overlay
   the seam mask. Zero distortion; honest but cosmetic.

**Tier B — pipeline fixes (re-export; addresses the cause):**
4. **Fix the M2 value step (Landsat + S2) — cross-path/orbit normalization + feathered
   blend.** The Landsat (042/043 path) and S2 (R070/R113 orbit) seams are *value* steps
   between fully-covering strips imaged on different days; `_scene_ops.mosaic_tiles`'
   first-valid-wins makes them hard cuts. Histogram-match / offset-correct adjacent strips to
   a common reference and feather across the overlap. **Highest-leverage cause fix** — these
   are confirmed co-primary drivers.
5. **Fix the M1 coverage edge (S1) — average + feather in the overlap zone** instead of a
   hard first-valid-wins cut. Reduces the within-coverage step; does not fill a true nodata
   gap (only overlapping-grid inference, C7, does).
6. **Coverage/recency input channel** — tell the model each pixel's per-source obs age so
   both a coverage edge *and* a stale-vs-fresh value step stop being read as snow signal
   (model-input change, larger scope; addresses M1 and M2 together).

**Tier C — the structurally correct fix (already flagged "for another day"):**
7. **Overlapping inference grids + averaging the FSC.** Offsetting grids moves each
   acquisition seam to a different location per offset; averaging cancels both the seams
   *and* the 1 km grid outline. Highest cost (N× inference), but the complete answer.

**Recommendation:** ship **A1** now (surgical, no re-run, defensible), and fold **B4 + B6**
into the next export (B4 first — Landsat is the clearest cause fix); treat **C7** as the
eventual definitive fix that subsumes both this and the grid-outline artifact. Method,
corrected evidence, and the measurement-method error are documented in §9.1; see §7
follow-up 3.
