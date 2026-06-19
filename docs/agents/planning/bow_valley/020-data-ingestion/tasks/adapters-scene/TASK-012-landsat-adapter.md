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
- **Upstream tasks:** TASK-002 (clipped Landsat, native UTM — see CRS note below),
  TASK-003 (`base.py` declares coalesce/mosaic contract), TASK-001 (reference patches).
- **Source semantics (DATA_ANALYSIS.md §Landsat 8/9 + §Verified Catalog):**
  - Clipped GeoTIFF bands in their **native UTM zone, which is MIXED per scene**
    (CORRECTED 2026-06-05, TASK-012b): paths **043/044 = EPSG:32611** (UTM 11N, SAME
    zone as the cell grid → no zone change), **042024 = 32612** (true cross-zone),
    **042025 = 32611**. USGS assigns zone by scene-center longitude; the archive is NOT
    uniformly 32612. The adapter must **read each band's CRS** and pass it to
    `base.reproject_to_cell` (zone-agnostic — same-zone and cross-zone both work); it
    must **never hardcode 32612**. Target is the **EPSG:32611** (UTM 11N) cell grid
    (PLAN §3 Grid+CRS table / `docs/agents/KNOWLEDGE.md`). See memory
    `landsat-mixed-utm-zone` + REVIEW_AUDIT #2.
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
- [x] 1. Write `test_landsat_adapter.py` (Red): three fallback cases (L9 present;
      L9 missing+L8 present; both missing → all-`-9999`); `bands_out =
      B2_landsat..B7_landsat`; per-band native-CRS reprojection asserted against the
      EPSG:32611 cell grid (synthetic source in BOTH 32611 same-zone and 32612
      cross-zone); coalesce (AC-15b) complementary-mask + latest-proc winner; **real-patch
      bit-exact parity** vs the 3 TASK-012b reference patches (B4_landsat).
- [x] 2. Implement `landsat.py`: DN→TOA via MTL, L9→L8 fallback, same-(tile,date)
      coalesce, cross-tile mosaic-before-crop, per-band native-CRS reproject, stack
      `(6, H, W)`. **Resample = NEAREST** (30 m > 10 m cell; GEE upsamples as constant
      blocks → bit-exact; bilinear smears — see §4 correction).
- [x] 3. Implement the `QA_PIXEL` cloud-flag path (NN, L9→L8 fallback) for the cloud slot.
- [x] 4. Wire into exporter. 5. Green + Refactor (extracted `_mosaic_tiles`).

## 4. Requirements & Constraints
- **Technical:** MTL coefficient parsing (`_MTL.json`); **NEAREST for reflectance AND
  `QA_PIXEL`** (CORRECTED 2026-06-05 from "bilinear for reflectance" — the archive
  disproved it: GEE upsamples the 30 m source to the 10 m cell as constant blocks, so
  nearest is bit-exact on all 3 reference patches while bilinear smears 0.003–0.012 over
  snow/cloud edges; same coarse-source rule as MODIS). Deterministic product ordering by
  processing time.
- **Business:** L9→L8 fallback and coalesce are non-negotiable (§9). TOA, not L2 SR.
  Coalesce is a valid-pixel union, never an average (preserves GEE value domain).
- **Out of scope:** S2 (TASK-013), S1 (TASK-014). Coalesce **algorithm** is shared via
  `base.py`; this task is its first production use + the Landsat-specific test.

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-16): three fallback cases pass; `bands_out` renamed; per-band
      native-CRS reproject to the EPSG:32611 cell grid asserted for BOTH a same-zone
      (32611) and a cross-zone (32612) source (no hardcoded zone).
- [x] AC-2 (SPEC AC-15b): complementary-mask coalesce → zero `-9999` where either input
      valid; surviving value = deterministic-order winner (latest proc time).
- [x] AC-3 (SPEC AC-12): **bit-exact parity** (median 0) on all 3 TASK-012b reference
      patches (PR_20250406 t3 / PR_20250414 t2 / PR_20250510 t1, B4_landsat); band order
      correct. Required switching reflectance to NEAREST (§4).
- [x] AC-4 (SPEC AC-13): both-missing → all-`-9999` (optical + cloud adapters).
- [x] AC-5: ruff + mypy clean; 14 new tests green; full suite delta = 0 NEW failures
      (6 total = known-red baseline).

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
