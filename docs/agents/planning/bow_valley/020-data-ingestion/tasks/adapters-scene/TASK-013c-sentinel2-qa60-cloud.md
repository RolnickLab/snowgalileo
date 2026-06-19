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
- [x] `S2CloudAdapter` emits `QA60` on the cell grid (categorical, NN, same coalesce/mosaic
      path as `S2Adapter`). → **BUILT** (`s2.py::S2CloudAdapter`, `_qa60_from_msk_classi`).
- [x] If reconstructed: `QA60` matches the GEE reference patch within a stated tolerance on
      the covered timesteps. → **Bit-layout verified against a direct GEE pull**
      (`opaque<<10 | cirrus<<11`, opaque precedence, snow excluded; domain `{0,1024,2048}`);
      real-archive parity asserted after the lossless re-clip (see §6).
- [x] Wired into the exporter cloud slot; ruff + mypy clean; no new suite failures. →
      `S2CloudAdapter` added to `exporter.py` `reals`; `_split_group` auto-fills the QA60
      slot. ruff/mypy clean; suite-delta empty.

## 5. Out of scope
Reflectance bands (TASK-013, done). Missing-date coverage (TASK-013b).

## 6. Outcome — QA60 reconstructed (corrected 2026-06-08)

**Result:** GEE's `QA60` **is** a deterministic repack of ESA `MSK_CLASSI`, not a separate
cloud algorithm — built as `S2CloudAdapter`.

**Mechanism (verified by a direct GEE pull, project `bow-valley-inference`):** the image
`COPERNICUS/S2_HARMONIZED/20250408T184941_20250408T185946_T11UNT` exposes `QA60`,
`MSK_CLASSI_OPAQUE/CIRRUS/SNOW_ICE`. Over PR_20250414: QA60=1024, OPAQUE=1, CIRRUS=0,
SNOW=0 → `QA60 = MSK_CLASSI_OPAQUE<<10 | MSK_CLASSI_CIRRUS<<11`, **opaque precedence, snow
excluded**, domain `{0,1024,2048}` (post-2024-02-28 reconstruction).

**Correction of the earlier same-day "infeasible" conclusion (it was WRONG).** That branch
read the *lossy-corrupted clipped* MSK_CLASSI, in which the patch's opaque had flipped
1→0, making GEE's QA60=1024 look unexplainable. Root cause was the **clip lossy-JP2 bug**
(`_clip_geotiff_to` wrote lossy JP2; reflectance ±2 DN + categorical class-flips) — fixed
in `clip/clippers.py` with a lossless-JP2 guard, then **S2 re-clipped**. The **raw** SAFE
MSK_CLASSI opaque matches GEE (=1). The SNAP-route idea (S1 analogy) does not apply: QA60
is an ESA L1C band, not a SNAP product.

**Tests:** `_qa60_from_msk_classi` bit-layout + opaque-precedence + snow-exclusion;
`S2CloudAdapter` bands/kind/missing-day/end-to-end reconstruction; real-archive QA60 parity
vs GEE patches (post-reclip). Clip guard: `test_clip_dataset.py::test_sentinel2_clip_is_lossless`.
