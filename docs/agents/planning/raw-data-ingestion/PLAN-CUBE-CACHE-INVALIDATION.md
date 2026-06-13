# PLAN — Cube cache invalidation: version stamp + interactive overwrite

## Goal

Make cube-cache staleness an explicit decision, never a silent default. After an adapter
or clip change, a stale `.npz` entry would serve wrong band data into a cube (and the
model). Two mechanisms:

1. **Version stamp (automatic safety net).** A `CACHE_VERSION` constant; the cache writes
   it into the dir and refuses to reuse a dir stamped with a different version — that case
   is force-cleared regardless of operator choice (a known-incompatible cache can never be
   reused by mistake).
2. **Interactive overwrite (operator's call).** For the ambiguous "did my clips change?"
   case the stamp can't catch, the CLI prompts: reuse the existing cache or overwrite it.
   Drives an `overwrite_cache` flag down into the modules. No reliance on remembering to
   bump a constant.

## Contracts

### `CubeCache` (cube_cache.py)
- Module constant `CACHE_VERSION: int = 1`.
- Stamp file `cube_cache/.cache_version` containing the integer.
- `__init__(root, max_entries, *, overwrite=False)`:
  - **Version mismatch** (stamp present and != `CACHE_VERSION`): clear the dir, rewrite the
    stamp, log `cube_cache_version_invalidated`. Unconditional — overrides `overwrite`.
  - **`overwrite=True`**: clear the dir, (re)write the stamp, log `cube_cache_overwritten`.
  - **Otherwise**: if no stamp (fresh/empty dir), write it; else reuse. Then `_scan_existing`
    as today.
- New private `_clear()`: remove every `*.npz` (and empty shard dirs), reset `_order`,
  preserve/rewrite the stamp. The stamp file is **never** counted as a cache entry
  (`_scan_existing` already globs `*.npz`, so the `.cache_version` file is naturally
  excluded — verify and assert).
- `is_empty` / `entry_count` already covered by `__len__`.

### Exporter (`LocalSourceExporter.__init__`)
- New `overwrite_cache: bool = False`, forwarded into `CubeCache(..., overwrite=...)`.
  Default `False` → behaviour-identical to today.

### Parallel export (`export_cells_parallel`, `_init_worker`)
- New `overwrite_cache: bool = False`, threaded into the per-worker exporter.
- **Concurrency rule (critical):** only the FIRST cache construction may clear. If 8
  workers each built with `overwrite=True`, worker 2 would wipe worker 1's fresh entries
  mid-run. **Resolution:** the CLI clears the cache ONCE, up front, in the parent process
  (before the pool spawns), then passes `overwrite_cache=False` to the workers. The workers
  never clear — they only read/write/version-check an already-clean dir. So `overwrite_cache`
  on `export_cells_parallel` is effectively "clear once before spawning", implemented by
  constructing a throwaway `CubeCache(root, overwrite=True)` in the parent, then spawning
  workers with `overwrite=False`.

### CLI (`infer_bow_valley_daily_fsc.py`, `export_bow_valley_cube.py`)
- New option `--cache-policy {prompt|reuse|overwrite}`, default `prompt`.
- Resolution at startup, in the parent process:
  - `reuse` → `overwrite_cache=False`.
  - `overwrite` → clear once up front (`overwrite_cache=True` path), workers reuse.
  - `prompt`:
    - cache dir empty/absent → no question, proceed (`reuse`, nothing to lose).
    - cache non-empty **and** stdin is a TTY → ask "Existing cube cache has N entries.
      [r]euse / [o]verwrite?"; map answer to reuse/overwrite.
    - cache non-empty **and NOT a TTY** → **error out** with a clear message instructing
      to pass `--cache-policy reuse|overwrite` (no hang, no silent staleness). This is the
      chosen no-TTY behaviour.
- A standalone **`clean-cache`** command (in `process_raw_dataset.py`, alongside the clip
  phases) wipes `CubeSettings.cube_cache_dir` on demand, reporting entries removed.

## Why this shape
- Stamp catches *known* incompatibilities deterministically (developer bumps it in the same
  diff that changes an adapter — the diff makes it obvious, but forgetting it is backstopped
  by the prompt).
- Prompt catches *unknown* ones (a re-clip leaves no code change, so the stamp won't fire —
  the operator decides).
- No-TTY error prevents the failure mode that motivated this: a background sweep silently
  reusing a cache built from pre-fix clips.
- Clearing once in the parent avoids the multi-worker clear race entirely.

## Tests
1. Version mismatch force-clears: seed a dir with entries + an old stamp; constructing
   `CubeCache` clears it and writes the current stamp.
2. Matching stamp reuses: same version → entries survive, count unchanged.
3. `overwrite=True` clears + rewrites stamp; `overwrite=False` (fresh dir) writes stamp,
   keeps nothing to clear.
4. Stamp file is not counted as an entry and survives `_clear`.
5. Exporter forwards `overwrite_cache`; parallel `_init_worker` builds workers with
   `overwrite=False` (no worker clears).
6. CLI policy resolution: `reuse`/`overwrite` map correctly; `prompt` + non-empty + no-TTY
   raises with the actionable message; `prompt` + empty dir proceeds silently.
7. `clean-cache` command empties a populated cache dir.

## Out of scope
- Auto-hashing adapter/clip source (rejected: brittle to cosmetic edits).
- Cross-process locked LRU (unchanged; cap still sized above working set).

## ⚠️ OPEN QUESTIONS — RESOLVE BEFORE IMPLEMENTING

Status: **plan drafted, NOT yet implemented.** Decisions above are settled (version
source = interactive prompt + manual constant backstop; no-TTY = error requiring explicit
`--cache-policy`). The following two are still open and must be answered before coding:

1. **Where does the `clean-cache` command live?**
   - Proposed (author's recommendation): `process_raw_dataset.py`, alongside the
     clip/process phases — the natural "data prep" home.
   - Alternative: `export_bow_valley_cube.py`.
   - → DECIDE: which file.

2. **Commit shape — one unit or incremental?**
   - The change spans ~4 components (`CubeCache`, exporter, `parallel_export`, the two
     CLIs). Smaller than the 4-step cache wiring but still multi-component.
   - Proposed (author's recommendation): **two commits** —
     (a) `CubeCache` version stamp + `overwrite` arg + `clean-cache` command;
     (b) thread `overwrite_cache` through exporter / `parallel_export` / CLIs +
     `--cache-policy`.
   - → DECIDE: accept two-commit split, or different granularity.

### ⚠️ CRITICAL IMPLEMENTATION NOTE (do not lose)
Clearing must happen **once in the parent process, before the worker pool spawns** — never
inside the workers. If all 8 workers construct `CubeCache(overwrite=True)`, worker 2 would
wipe worker 1's freshly-written entries mid-run. The CLI clears up front (parent), then
spawns workers with `overwrite_cache=False`. See the `export_cells_parallel` /
`_init_worker` contract above. This is the single most important correctness constraint of
this plan.
