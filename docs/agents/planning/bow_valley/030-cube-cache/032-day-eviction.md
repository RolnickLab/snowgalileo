# PLAN — Cube cache day-frontier eviction (Mode B scale)

*Formerly `PLAN-CUBE-CACHE-DAY-EVICTION.md`.*

## Problem

The cube cache's "race avoided" guarantee rests on `working_set < cap`. Measured:

| Mode | Cells | Working set (cells × 28 days × 9 modalities) | vs 200k cap |
|------|-------|----------------------------------------------|-------------|
| A    | 344   | ~86.7k                                        | under ✓     |
| B    | 25,078 | ~6.3M                                        | ~31× over ✗ |

Mode B (full-AOI tiling) is **73× more cells**. The cap is blown by ~31×, so the current
per-`put` FIFO eviction **will** fire continuously across 8 concurrent workers — the exact
cross-process eviction race the cache module deferred (PLAN-CUBE-CACHE-WIRING §Concurrency,
point 3). "Raise the cap above the working set" is no longer viable at 6.3M entries.

## Key insight — the sweep is day-ordered, so dead entries are provable

`InferenceGridDriver.run()` (driver.py:103) iterates `inference_days(...)` **strictly
ascending**, fully exporting + predicting day `D` before day `D+1`. Parallelism is *within*
a day (cells across workers), never across days.

Each cube for day `D` reads cube-cache entries for days `[D − 7 … D]` only
(`_window_days`, `NUM_TIMESTEPS=8`, `DAYS_PER_TIMESTEP=1`; exporter.py:278). Therefore once
the driver advances to `current_day`, **every entry with `day < current_day − 7` is dead —
it will never be read again.** This is the safe eviction frontier; it requires no recency
guessing and is correct regardless of worker count.

`WINDOW = (NUM_TIMESTEPS - 1) * DAYS_PER_TIMESTEP = 7` days of backlook.

## Decisions (settled with operator)

1. **Prune rule: day-frontier only.** When `len(cache) > cap`, drop only entries with
   `day < current_day − WINDOW`. The live window (last 8 days) is **always kept**. If
   pruning the dead frontier still leaves the cache over cap, **log a warning and proceed**
   — never evict a live entry. (Rejected: FIFO-evict live entries when still over cap —
   reintroduces the race on the hot window.)
2. **Prune site: parent, between days.** The driver prunes once at each day boundary, in
   the parent process, before spawning the next day's workers. **Workers never evict.** The
   cross-process race stays structurally impossible (same discipline as the overwrite clear:
   only the parent ever mutates the cache set).
3. **Lazy, not eager.** Pruning fires only when over the cap — not per-`put`. Below the cap
   (e.g. all of Mode A) nothing is ever evicted, behaviour-identical to today.
4. **Cap is configurable, default 200k.** Already present: `CubeSettings.cache_max_entries`
   (default 200_000), threaded through exporter/parallel/driver. No new setting needed; this
   plan only changes *when/how* eviction fires, not where the cap comes from.

## Why this satisfies "don't evict live data, leave margin for many workers"

The margin is **structural, not a tuned number**. Because prune only ever removes entries
strictly older than the live window, no concurrent worker can have a live entry pulled out
from under it — independent of how many workers a powerful cluster runs. There is no window
of vulnerability to widen or shrink.

## Contracts

### `CubeCache` (cube_cache.py)
- **Disable per-`put` eviction.** `put` no longer calls `_evict_to_cap`. (Workers only
  get/put; the cap is enforced by the parent's day-frontier prune.) Keep `_evict_to_cap`
  *unused-by-put* but available, OR remove it — see "Open" below.
- **New `prune_before_day(current_day: datetime.date, *, window_days: int) -> int`:**
  - No-op (returns 0) when `len(self) <= self.max_entries` (lazy trigger).
  - Else: compute `frontier = current_day - timedelta(days=window_days)`; unlink every
    entry whose parsed day `< frontier`; drop those keys from `_order`; remove now-empty
    shard dirs (reuse `_clear`'s dir-cleanup helper, factored out).
  - Returns the number of entries removed. If still `> max_entries` after, log
    `cube_cache_over_cap_after_prune` (warning) and return the count anyway — never touches
    live entries.
  - **Day parse**: filename is `{YYYYMMDD}_{modality}.npz`; parse the `YYYYMMDD` head with
    `datetime.datetime.strptime(stem.split("_", 1)[0], "%Y%m%d").date()`. A name that
    doesn't parse is left untouched (defensive — never delete an unrecognised file).

### Driver (`InferenceGridDriver`)
- In `run()`, **before** `_predict_day(day)`, call the parent-side prune **iff** the
  exporter owns a cache:
  ```python
  cache = self.exporter._cache
  if cache is not None:
      cache.prune_before_day(day, window_days=WINDOW)
  ```
  `WINDOW` imported from the same config constants the exporter uses
  (`(NUM_TIMESTEPS - 1) * DAYS_PER_TIMESTEP`). Runs in the parent, single-threaded, before
  the day's worker pool spawns — preserving the no-worker-eviction guarantee.
- The serial fallback path (`export_workers <= 1`) also benefits: same parent, same call.

### Settings — unchanged
`cache_max_entries` already configurable (default 200_000). For a Mode B run the operator
may still raise it, but the day-frontier prune means they no longer *have to* size it above
the full sweep working set — they size it to the disk they want the live+recent window to
occupy.

## Concurrency argument (the load-bearing claim)

At the moment `prune_before_day(D)` runs, the next day to be exported is `D`, reading
`[D−7 … D]`. Pruned entries are `day < D−7`. No current-or-future day reads those. The pool
for day `D` has **not yet spawned** (prune is before `_pre_export_day`). So at prune time
there is exactly one process (parent) touching the cache, and it removes only entries no
future reader wants. Worker count never enters the argument. ∎

## Tests

1. **Lazy below cap**: cache with entries `< cap`, `prune_before_day` removes nothing
   (returns 0), all entries survive — Mode-A behaviour unchanged.
2. **Frontier prune over cap**: seed `> cap` entries spanning many days; `prune_before_day(D)`
   removes exactly those with `day < D − WINDOW`, keeps `[D−7 … D]` and anything ≥ D.
3. **Live window never evicted**: an entry at `day == D − WINDOW` (boundary, still live)
   survives; `day == D − WINDOW − 1` is removed.
4. **Over cap after prune → warn, keep live**: construct a case where even after dropping
   the dead frontier the live window alone exceeds cap; assert no live entry removed and the
   warning is logged.
5. **Unparseable filename untouched**: a stray `garbage.npz` in the cache is never deleted.
6. **`put` no longer evicts**: filling past cap via `put` alone (no prune) keeps every
   entry — eviction is now exclusively the parent's job.
7. **Empty shard dirs removed after prune** (mirror the `_clear` dir-cleanup test).
8. **Driver calls prune once per day, before export**: spy/mock asserts `prune_before_day`
   is invoked with each ascending day prior to `_pre_export_day`, only when a cache exists.

## Out of scope
- Cross-process locked LRU / eviction-owner process (the day-frontier prune makes it
  unnecessary at Mode B scale; documented as the deliberate non-choice).
- Pruning *within* a day (mid-export). Unnecessary: a single day adds at most
  `cells × 9` entries; the between-day cadence bounds growth to one day's worth over cap.

## Open (decide before coding)
- **Keep or remove `_evict_to_cap`?** It becomes dead once `put` stops calling it.
  Recommendation: **remove it** (and its FIFO `_order`-popping) to avoid two eviction
  mechanisms; `_order` is then only used for `__len__` / scan-recovery. Confirm — this
  slightly widens the diff into `_order`'s role.
- **Commit shape**: one commit (CubeCache prune + driver hook + tests) is small and
  cohesive — propose single commit. Confirm.
