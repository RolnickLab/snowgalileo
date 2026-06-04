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
- **Source semantics (DATA_ANALYSIS.md §Sentinel-2 + §Verified Catalog):**
  - Clipped SAFE JP2, `EPSG:32611`, tiles `T11UNS/NT/PS/PT`; `uint16`.
  - Bands `B2,B3,B4,B8,B11,B12` (10 m + 20 m SWIR resampled onto the grid).
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
- [ ] 1. Write `test_s2_adapter.py` (Red): golden-grid triple; `bands_out =
      [B2,B3,B4,B8,B11,B12]`; N0511 granule → −1000 DN applied; reflectance domain matches
      `S2_HARMONIZED` (÷10000 downstream); parity vs reference within TASK-005 tolerance;
      **coalesce (AC-15b)** with the real `20250420 T11UNT` R113-vs-R070 case →
      coalesced valid-pixel count ≥ max of either alone, zero false `-9999`.
- [ ] 2. Implement `s2.py`: read JP2 bands, baseline-check + −1000 DN, resample SWIR onto
      grid, same-(tile,date) coalesce, cross-tile mosaic-before-crop, reproject, stack
      `(6, H, W)`.
- [ ] 3. Implement the `QA60` cloud-flag path (NN) for the cloud slot.
- [ ] 4. Wire into exporter (replace spike). 5. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Bilinear for reflectance, NN for `QA60`; baseline parsed from
  `MTD_MSIL1C.xml` `<PROCESSING_BASELINE>` (fallback: `N0511`-from-path);
  deterministic product order by processing time.
- **Business:** −1000 DN harmonization is required for every (N0511) granule. Coalesce
  is a valid-pixel union, never an average.
- **Out of scope:** L2A surface reflectance (not interchangeable), S1 (TASK-014).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-15): `bands_out` correct; N0511 → −1000 DN; domain matches
      `S2_HARMONIZED`; parity within tolerance.
- [ ] AC-2 (SPEC AC-15b): real R113-vs-R070 coalesce → valid-pixel count ≥ max of either
      alone; zero false `-9999`; surviving value = deterministic-order winner.
- [ ] AC-3 (SPEC AC-12): golden-grid triple; band order correct.
- [ ] AC-4 (SPEC AC-13): missing `(S2, day)` → all-`-9999`.
- [ ] AC-5: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

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
