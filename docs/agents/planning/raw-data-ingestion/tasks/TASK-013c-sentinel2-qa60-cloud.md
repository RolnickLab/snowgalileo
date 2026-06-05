# TASK-013c: Reconstruct the Sentinel-2 `QA60` cloud flag for the cloud slot

## 1. Goal
Emit a `QA60` band that reproduces GEE's `COPERNICUS/S2_HARMONIZED` `QA60` value domain on
the cell grid, filling the S2 cloud slot that TASK-013 deliberately left as a `-9999`
placeholder.

## 2. Context & Why (deferred from TASK-013)
- **N0511 SAFEs ship no `QA60.jp2`.** Baseline ≥ N0400 replaced the legacy
  `IMG_DATA/.../QA60.jp2` with the new `QI_DATA/MSK_CLASSI_B00.jp2` cloud mask (a 60 m,
  3-band {opaque, cirrus, snow} uint8 mask). GEE *backfills* a synthesized `QA60` for these
  products from its own cloud algorithm.
- A **naive repack** of `MSK_CLASSI` into the QA60 bit layout
  (`opaque·1024 + cirrus·2048`, 60 m nearest) does **not** match the reference patch:
  measured {0, 3072} reconstructed vs {0, 1024} in the GEE patch, ~17 % pixel match. So
  GEE's `QA60` is not a simple `MSK_CLASSI` transform; the exact mapping/algorithm needs
  reverse-engineering.
- **Low priority:** `QA60` is "exported but **not used** by the main dataset tensor"
  (DATA_ANALYSIS §Sentinel-2). The reflectance bands (the model input) are already
  bit-exact (TASK-013). This task is correctness-completeness, not a model blocker.

## 3. Investigation directions
1. Compare GEE `QA60` vs `MSK_CLASSI` bands pixel-by-pixel over several covered patches to
   learn the true mapping (bit alignment, 60 m grid origin, opaque-vs-cirrus precedence).
2. Check whether GEE derives `QA60` from `MSK_CLASSI` at all, or from a separate cloud
   probability product — if the latter, faithful local reconstruction may be infeasible and
   the honest outcome is a documented `-9999` placeholder + a recorded limitation.
3. Resample 60 m → 10 m cell with **nearest** (coarse-source rule); `QA60` is categorical.

## 4. Acceptance Criteria
- [ ] `S2CloudAdapter` emits `QA60` on the cell grid (categorical, NN, same coalesce/mosaic
      path as `S2Adapter`), OR a documented decision that faithful reconstruction is
      infeasible and the placeholder stays (with the evidence recorded).
- [ ] If reconstructed: `QA60` matches the GEE reference patch within a stated tolerance on
      the covered timesteps.
- [ ] Wired into the exporter cloud slot; ruff + mypy clean; no new suite failures.

## 5. Out of scope
Reflectance bands (TASK-013, done). Missing-date coverage (TASK-013b).
