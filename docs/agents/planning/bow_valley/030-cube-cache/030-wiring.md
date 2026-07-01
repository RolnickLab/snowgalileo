# PLAN — Wire the per-(modality, cell, day) CubeCache into the exporter

*Formerly `PLAN-CUBE-CACHE-WIRING.md`.*

## Problem

The 21-day Mode A inference run is **export-bound, not GPU-bound**. Observed during the
live run: GPU oscillates 0–100% with only ~1.6–4.9 GB / 12 GB VRAM, a single main
process at ~12% CPU, `rchar` ≈ 2.7 TB (page-cache reads), `read_bytes` ≈ 23 MB (not
disk-bound). The model (`ai4snow_tiny`, 192/12/3) finishes a 16-cell batch in
milliseconds, then waits. Wall time is dominated by **cube assembly**, repeated every day.

### Root cause: redundant per-day re-assembly

`LocalSourceExporter._assemble` (exporter.py:285) rebuilds the 308-band stack per
`(cell, window_end)`:

```python
for day in self._window_days(window_end):      # 8 days, sliding window
    for adapter in self._dynamic:
        blocks.append(adapter.fetch(cell, day)) # reads source rasters
for adapter in self._static:
    blocks.append(adapter.fetch(cell, None))    # window-invariant
```

Consecutive inference days' 8-day windows **overlap by 7 days**, so ~7/8 of the dynamic
`fetch(cell, day)` calls recompute an *identical* array (each `fetch` is a pure function
of `(adapter, cell, day)` — verified: every `fetch` signature is `(cell, day)`, no
`window_end` dependence; ERA5's precip `day+1` shift is still a pure function of `day`).
Across 21 inference days × 344 cells × 8 timesteps × ~9 dynamic adapters this is the
bottleneck.

### The fix already exists — but is unwired

`src/data/local_sources/cube_cache.py` ships a complete `CubeCache`: per-cell-sharded
`.npz` (`cube_cache/{cell_id}/{day:%Y%m%d}_{modality}.npz`), FIFO eviction with a
configurable cap, crash-safe atomic writes, mtime-recovered order across restarts.
`cube.yaml` even sets `cache_max_entries: 200000`. But:

- `grep cube_cache|CubeCache` in `exporter.py` / `parallel_export.py` → **zero hits**.
- `cube_cache/` on disk → **empty (0 files)**.

It is dead code. Wiring it removes the ~7/8 redundant re-fetch.

## Scope

In: connect `CubeCache` to `_assemble`'s dynamic-block fetch as a read-through /
write-back memo, keyed `(modality, cell_id, day)`. Thread the cache dir + cap through
the exporter and the parallel-export workers. Out: any change to adapter `fetch` logic,
band order, the static block's handling, or the GPU/inference path.

## Contract (the cache seam)

```
block = cache.get(modality=tag, cell_id=cell.cell_id, day=day)
if block is None:
    block = adapter.fetch(cell, day)
    cache.put(modality=tag, cell_id=cell.cell_id, day=day, array=block)
```

- **Pure-function premise** (correctness): `adapter.fetch(cell, day)` depends only on
  `(adapter, cell, day)`. Verified above. A cached block is therefore valid in *any*
  window that includes `day`.
- **`modality` tag** must be stable per dynamic adapter slot and unique within a cube.
  Adapters carry no `modality` field — only `bands_out` / `spatial_kind`, and after
  `_split_group` a slot may be a real adapter or a placeholder slice. **Decision: derive
  the tag from the slot's band signature** — `f"{spatial_kind}_{bands_out[0]}_{len}"`
  (e.g. `high_VV_3`, `time_skin_temperature_5`, `low_sur_refl_b01_7`). This is stable,
  unique per contiguous slice, and survives a placeholder/real swap of the *same* slice
  (both produce the same bands → same cache entry → still correct). Reject `type().__name__`
  (placeholders collide) and a bare index (fragile to band-order edits).
- **Static block: do NOT cache via this path.** It is `fetch(cell, None)` — window- and
  day-invariant, already cheap (one fetch per cell per cube, not ×8). Caching it adds a
  `day=None` key wart for little gain. Leave static assembly unchanged.
- **Shape/dtype invariant**: a cache hit must return exactly what `fetch` would —
  `(len(bands_out), *cell.shape)` float. Guard on read: if a hit's leading dim ≠ the
  slot's `len(bands_out)`, treat as a miss + overwrite (defends against a stale entry
  from a band-count change). Cheap and prevents a corrupt cube.

## Concurrency (parallel_export)

8 workers each build their own `LocalSourceExporter` (`_init_worker`) and write the
**same** filesystem cache dir. Implications:

1. **Cache dir must reach the worker.** `_init_worker` / `export_cells_parallel` only
   pass `(out_dir, archive_root, placeholder, verify_s1_cache)`. Add `cube_cache_dir`
   - `cache_max_entries` to that signature and to `LocalSourceExporter.__init__`.
2. **Atomic writes already make concurrent `put` safe** (temp-sibling + `replace`).
   Concurrent `get` of a half-written file can't see it (replace is atomic). Good.
3. **FIFO eviction is per-process and racy across workers** — each worker's `_order`
   only knows files it scanned at init; two workers could both evict, or `unlink` a file
   another just wrote. **Decision for v1: disable in-run eviction in the parallel path**
   by sizing the cap above the working set. Mode A working set =
   ~344 cells × (window span days) × ~9 modalities. For a 21-day inference sweep the
   distinct days touched = 21 + 7 backlook = 28 → 344×28×9 ≈ 86.7k < the 200k cap, so the
   cap is never hit and eviction never races. Document this; a cross-process locked LRU is
   out of scope (premature — the cap proxy suffices at this scale, per the cache module's
   own note). If a future run's working set exceeds the cap, raise `cache_max_entries`.
4. **Disk budget**: ~86.7k entries × per-(modality,cell,day) array (~10×10×bands×4B,
   ≤ a few KB each, most ~0.4–4 KB) ≈ low single-digit GB. Lives under
   `processing_root/cube_cache/` (cleanable). Acceptable; note it in the plan output.

## Changes (incremental, one reviewable unit each)

**Step 1 — Exporter owns an optional cache.**
`LocalSourceExporter.__init__`: add `cube_cache_dir: Path | None = None`,
`cache_max_entries: int = DEFAULT_MAX_ENTRIES`. If `cube_cache_dir` is not None and not
`placeholder`, construct `self._cache = CubeCache(cube_cache_dir, cache_max_entries)`;
else `self._cache = None` (placeholder mode + tests stay cache-free, behaviour-identical).
Add a `_modality_tag(adapter)` helper (band-signature scheme above).

**Step 2 — Read-through/write-back in `_assemble`.**
Wrap only the **dynamic** inner fetch with the get/put contract above, including the
shape-guard-on-hit. Static loop untouched. No other line of `_assemble` changes.

**Step 3 — Thread the cache through parallel export.**
`export_cells_parallel`, `_init_worker`, `_export_one` gain `cube_cache_dir` +
`cache_max_entries`; pass them into the per-worker `LocalSourceExporter`. Default
`cube_cache_dir=None` keeps every existing caller (and the stub-exporter tests)
behaviour-identical.

**Step 4 — Wire the driver + CLIs.**
`InferenceGridDriver._pre_export_day` passes `cube_cache_dir=CubeSettings.cube_cache_dir`
and `cache_max_entries=settings.cache_max_entries` into `export_cells_parallel`. The
serial fallback (`_run_batch` → `self.exporter.export`) benefits automatically once the
injected exporter is built with a cache. `export_bow_valley_cube.py` (standalone) and
`infer_bow_valley_daily_fsc.py` construct the exporter with the cache dir from settings.

## Tests

1. **Cache hit returns fetch-equal block**: build a cube twice for two overlapping
   window-ends; assert the shared-day dynamic blocks are bit-identical and the second
   assembly issues zero `fetch` for the cached `(modality, cell, day)` (spy/mock count).
2. **Cube bit-identity with vs without cache**: full `export(cell, day)` produces a
   byte-identical tif whether `cube_cache_dir` is set or None (the cache must be a pure
   memo — this is the load-bearing correctness test).
3. **Modality tag uniqueness**: every dynamic slot in `full_band_order` maps to a
   distinct `_modality_tag`; a placeholder vs real swap of the same slice yields the
   same tag.
4. **Shape-guard on stale hit**: a cache entry with the wrong leading dim is treated as
   a miss and overwritten (no corrupt cube).
5. **Parallel path uses the cache**: `export_cells_parallel` with a `cube_cache_dir`
   populates it and a second call reads it (entry count stable, mtimes refreshed).
6. **Default-off is behaviour-identical**: `cube_cache_dir=None` path equals today's
   output and writes nothing under `cube_cache/`.

## Expected impact

First inference day pays full export (cache cold). Each subsequent day reuses 7/8 of its
dynamic blocks from cache → dynamic-fetch work drops ~8×→~1× for days 2…21. Net sweep
export time roughly `(1 full day) + (20 × ~1/8 day)` instead of `21 × 1 day` — order-of
a few× faster end-to-end, GPU unchanged (it was never the limit). Exact speedup measured
post-implementation against a 2-day control.

## Does NOT touch the running job

The live 21-day run keeps its current (uncached) exporter. This benefits the **next**
run or a restart. No mid-flight change.

## Risks / call-outs

- **Correctness is everything**: the cube feeds the model. Test #2 (bit-identity with/
  without cache) is the gate — if it fails, the cache is not a pure memo and must not ship.
- **Cross-process eviction race** is deliberately avoided (cap > working set), not solved.
  Flagged as a known limitation; safe for Mode A at this scale.
- **Stale cache across code changes**: if an adapter's `fetch` logic changes (e.g. the
  ERA5/MODIS clip fixes earlier today), old `.npz` entries become wrong. **Mitigation:
  the cache dir is cleanable; document "clear `cube_cache/` after any adapter/clip change."**
  Consider a cache-version stamp file as a follow-up (out of scope for v1).
