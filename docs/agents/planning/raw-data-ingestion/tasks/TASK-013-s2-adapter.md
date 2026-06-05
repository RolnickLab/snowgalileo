# TASK-013: Implement the Sentinel-2 adapter (−1000 DN harmonization, coalesce)

## 1. Goal
Promote the S2 parity spike to a production adapter that emits `[B2,B3,B4,B8,B11,B12]`
on the cell grid, subtracting 1000 DN for baseline ≥ 04.00 (N0511) granules to match
`S2_HARMONIZED`, with same-(tile,date) coalesce before cross-tile mosaic-before-crop.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #8 — parity spike done in TASK-005, now production).
- **SPEC:** FR-9, FR-9b, FR-11, AC-12, AC-13, AC-15, AC-15b; Verification Plan step 6.
- **PLAN:** §4 adapter rules (S2 harmonization; coalesce before mosaic), §6 FMEA,
  §9 non-negotiables, §8 Q8 (all archive granules L1C N0511 → −1000 DN required for
  every granule).
- **Upstream tasks:** TASK-005 (S2 parity spike + tolerance), TASK-002 (clipped S2,
  EPSG:32611), TASK-003 (coalesce/mosaic contract), TASK-001 (reference patches).
- **Coverage validated (TASK-012b lesson, 2026-06-05):** like Landsat, the S2 archive has
  a download gap — **9 of 20 reference-patch S2 timesteps fall on dates absent from the
  clipped archive** (they land in the archive's 7-day cadence gaps). BUT **every patch has
  ≥1 covered date**, so bit-exact parity is validatable now; the 9 missing dates →
  **TASK-013b** (no blocker). A `test_every_patch_has_a_covered_s2_date` test enforces this
  and xfails with the explicit backlog. CRS note: unlike Landsat's WRS-2 mixed-zone trap,
  **S2 MGRS tiles are single-zone** (all archive tiles `T11U**` = EPSG:32611); the adapter
  still reads per-band CRS defensively.
- **Source semantics (DATA_ANALYSIS.md §Sentinel-2 + §Verified Catalog):**
  - Clipped SAFE JP2, `EPSG:32611`, tiles `T11UNS/NT/PS/PT`; `uint16`.
  - Bands `B2,B3,B4,B8,B11,B12` (10 m + 20 m SWIR resampled onto the grid).
  - **Resample = NEAREST** (CORRECTED 2026-06-05 from "bilinear"): the 20 m SWIR bands
    (B11/B12) are coarser than the 10 m cell, so GEE upsamples as constant blocks → nearest
    is bit-exact (B4 signed-median 0 on 3 covered patches; bilinear smeared SWIR ~20 DN).
    The 10 m bands (B2/B3/B4/B8) are on the cell grid already. Same coarse-source rule as
    MODIS/Landsat. (Supersedes §4 "bilinear for reflectance".)
  - **Harmonization:** direct Copernicus L1C does NOT have the GEE +1000 DN harmonization.
    Read the processing baseline from `<PROCESSING_BASELINE>` in the granule's
    `MTD_MSIL1C.xml` (verified present as `05.11`), falling back to the
    `N0511`-from-path token if the tag is missing; if `≥ 04.00` (N0511) subtract 1000
    DN. **All 116 archive granules are N0511 → required for every granule** (Q8).
    (REVIEW_AUDIT.md verdict #6.)
  - ÷10000 downstream; valid `>= -1`.
  - `QA60` cloud flag emitted separately in the S2 cloud slot.
  - `spatial_kind="high"`.
- **Same-tile/date coalesce (DATA_ANALYSIS §Same-tile/date; verified S2 ≥7 groups, e.g.
  `20250420 T11UNT` = R113 vs R070):** gather **all** same-(tile,date) products, coalesce
  per pixel (first valid wins, latest-processing-time order), `-9999` only where all
  nodata; runs per-tile before the cross-tile mosaic. Valid-pixel union, not an average.
- **Relevant skills:** `geospatial` (JP2 read, SWIR resample, coalesce/mosaic), `tdd`.

## 3. Subtasks
- [x] 1. Write `test_s2_adapter.py` (Red): `bands_out=[B2,B3,B4,B8,B11,B12]`; N0511 granule
      → −1000 DN; coalesce complementary-mask + latest-proc winner; **coverage-validation
      test** (every patch ≥1 covered date, xfails with TASK-013b backlog); **bit-exact B4
      parity** on covered patches (signed median 0).
- [x] 2. Implement `s2.py`: read JP2 bands, baseline-check + −1000 DN, same-(tile,date)
      coalesce, cross-tile mosaic-before-crop, **NEAREST** reproject, stack `(6, H, W)`.
      Coalesce/mosaic lifted to shared `_scene_ops.py` (also used by Landsat).
- [~] 3. `QA60` cloud-flag path — **deferred to TASK-013c** (N0511 ships no `QA60.jp2`; a
      naive `MSK_CLASSI` repack does not match GEE's backfilled QA60). Stays placeholder.
- [x] 4. Wire into exporter (S2Adapter in `reals`). 5. Green + Refactor (`_scene_ops` lift).

## 4. Requirements & Constraints
- **Technical:** **NEAREST for reflectance** (20 m SWIR > 10 m cell → bit-exact; corrected
  from "bilinear"); baseline parsed from `MTD_MSIL1C.xml` `<PROCESSING_BASELINE>`
  (fallback: `N0511`-from-name); deterministic product order by processing time.
- **Business:** −1000 DN harmonization is required for every (N0511) granule. Coalesce
  is a valid-pixel union, never an average.
- **Out of scope:** L2A surface reflectance (not interchangeable), S1 (TASK-014).

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-15): `bands_out` correct; N0511 → −1000 DN; **bit-exact** B4 parity
      (signed median 0) on the covered patches (`S2_HARMONIZED` domain reproduced).
- [x] AC-2 (SPEC AC-15b): complementary-mask coalesce → zero false `-9999`; surviving value
      = latest-proc winner. (Real R113-vs-R070 case lives in the archive; the unit test uses
      synthetic complementary masks for determinism.)
- [x] AC-3 (SPEC AC-12): band order correct; output on the cell grid.
- [x] AC-4 (SPEC AC-13): missing `(S2, day)` → all-`-9999`.
- [x] AC-5: ruff + mypy clean; 10 new tests green (1 xfail = TASK-013b backlog); full-suite
      delta = 0 new failures.
- [+] **Coverage AC (added per user):** every reference patch has ≥1 covered S2 date
      (hard-asserted); the 9 missing dates are captured in **TASK-013b**.

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_s2_adapter.py -v
uv run ruff check src/data/local_sources/s2.py
uv run mypy src/data/local_sources/s2.py
```
Expected: adapter test green (−1000 DN + coalesce); parity within tolerance; ruff/mypy
exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/s2.py tests/test_local_sources/test_s2_adapter.py
   git commit -m "feat(bow-valley): Sentinel-2 adapter (−1000 DN harmonization, coalesce) — closes TASK-013"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-014.
