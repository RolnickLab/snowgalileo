# TASK-012: Implement the Landsat 8/9 adapter (L9→L8 fallback, cross-zone reproject, coalesce)

## 1. Goal
Replace the Landsat placeholder with a real adapter that emits renamed bands
`B2_landsat..B7_landsat` on the cell grid, encapsulating the L9→L8 fallback,
reprojecting cross-zone EPSG:32612→4326, and coalescing same-(tile,date) products.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #7).
- **SPEC:** FR-9, FR-9b, FR-12, AC-12, AC-13, AC-15b, AC-16; Verification Plan step 6.
- **PLAN:** §4 adapter rules (L9→L8 fallback encapsulated; same-tile/date coalesce
  before cross-tile mosaic-before-crop), §6 FMEA (L9→L8 regression; false `-9999`),
  §9 non-negotiables (mosaic + coalesce).
- **Upstream tasks:** TASK-002 (clipped Landsat in native EPSG:32612), TASK-003
  (`base.py` declares coalesce/mosaic contract), TASK-001 (reference patches).
- **Source semantics (DATA_ANALYSIS.md §Landsat 8/9 + §Verified Catalog):**
  - Clipped GeoTIFF bands, **native EPSG:32612** (cross-zone reproject to the 4326 cell
    grid happens HERE, in the adapter).
  - Original `B2,B3,B4,B5,B6,B7` → renamed `B2_landsat..B7_landsat` (avoid S2 collision).
  - **L9→L8 fallback:** try L9 first for date/region, fall back to L8 if L9 absent;
    both absent → all-`-9999` (renamed band names).
  - Convert DN→TOA reflectance via scene `_MTL` coefficients + sun-angle (NOT L2 SR);
    ÷10000 downstream; valid `>= 0.0000001` (zero is invalid/no-data).
  - `QA_PIXEL` cloud flag emitted separately in the Landsat cloud slot.
  - `spatial_kind="high"`.
- **Same-tile/date coalesce (DATA_ANALYSIS §Same-tile/date multi-product; verified
  Landsat 9 `20250425` path/row `044024` twice):** gather **all** products sharing
  the same (tile, date) — not `.first()` — coalesce per pixel: first valid
  (non-nodata, in-threshold) value wins, fall through where nodata, `-9999` only where
  all are nodata; deterministic order = latest processing time first. Runs per-tile
  **before** the cross-tile mosaic. Valid-pixel union, **not** an average.
- **Relevant skills:** `geospatial` (cross-zone reproject, mosaic, coalesce), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_landsat_adapter.py` (Red): three fallback cases (L9 present;
      L9 missing+L8 present; both missing → all-`-9999`); `bands_out =
      B2_landsat..B7_landsat`; cross-zone 32612→4326 reprojection asserted against the
      cell grid; golden-grid triple; **coalesce (AC-15b)**: two synthetic same-(tile,date)
      products with complementary nodata → zero `-9999` where either is valid, surviving
      value = latest-processing-time winner.
- [ ] 2. Implement `landsat.py`: DN→TOA via MTL, L9→L8 fallback, same-(tile,date)
      coalesce, cross-tile mosaic-before-crop, cross-zone reproject, stack `(6, H, W)`.
- [ ] 3. Implement the `QA_PIXEL` cloud-flag path (NN, L9→L8 fallback) for the cloud slot.
- [ ] 4. Wire into exporter. 5. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** MTL coefficient parsing (`_MTL.json`); bilinear for reflectance, NN for
  `QA_PIXEL`; deterministic product ordering by processing time.
- **Business:** L9→L8 fallback and coalesce are non-negotiable (§9). TOA, not L2 SR.
  Coalesce is a valid-pixel union, never an average (preserves GEE value domain).
- **Out of scope:** S2 (TASK-013), S1 (TASK-014). Coalesce **algorithm** is shared via
  `base.py`; this task is its first production use + the Landsat-specific test.

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-16): three fallback cases pass; `bands_out` renamed; cross-zone
      32612→4326 asserted.
- [ ] AC-2 (SPEC AC-15b): complementary-mask coalesce → zero `-9999` where either input
      valid; surviving value = deterministic-order winner; coalesced valid-pixel count ≥
      max of either product alone.
- [ ] AC-3 (SPEC AC-12): golden-grid triple; band order correct.
- [ ] AC-4 (SPEC AC-13): both-missing → all-`-9999`.
- [ ] AC-5: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_landsat_adapter.py -v
uv run ruff check src/data/local_sources/landsat.py
uv run mypy src/data/local_sources/landsat.py
```
Expected: adapter test green (3 fallback cases + coalesce); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/landsat.py tests/test_local_sources/test_landsat_adapter.py
   git commit -m "feat(bow-valley): Landsat 8/9 adapter (L9→L8 fallback, cross-zone, coalesce) — closes TASK-012"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-013.
