# TASK-014: Implement the Sentinel-1 GRD adapter (windowed reads, edge mask)

## 1. Goal
Promote the S1 parity spike to a production adapter that emits `[VV, VH, angle]` on the
cell grid from GRD SAFE archives, applying the project edge mask (pixels `< -30.0`) and
using windowed reads of the cell footprint — never full-scene loads.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #9 — parity spike done in TASK-005, now production).
- **SPEC:** FR-9, AC-12, AC-13, AC-14; Verification Plan step 6.
- **PLAN:** §4 module note, §6 FMEA (memory/IO blowup → windowed reads), §9, §3 temporal
  window (S1 on only ~16 dates → dominant sparsity risk; many windows fully `-9999`).
- **Upstream tasks:** TASK-005 (S1 parity spike + tolerance), TASK-002 (clipped S1),
  TASK-003, TASK-001 (reference patches).
- **Source semantics (DATA_ANALYSIS.md §Sentinel-1 + §Verified Catalog):**
  - `.zip` SAFE, swath/sensor geometry, `uint16`, scene shape ~`(16708, 26079)`.
  - **Range geometry, GCPs only — verified.** `gdalinfo` on archive measurement
    TIFFs shows `GCP Projection = GEOGCRS["WGS 84"]`, 210 GCPs, **no `PROJCRS`**,
    raw pixel grid (`UL (0,0) → LR (26079,16708)`). A prior review wrongly claimed
    these GRD products are already UTM-projected; they are not. Terrain-correction
    to a map grid (this task) is therefore required, and the clip stage's GCP-based
    slice (`CLIPPING_PLAN.md §2.5`) is correct. (REVIEW_AUDIT.md verdict #5.)
  - Bands `[VV, VH, angle]`; IW mode; `angle` in degrees.
  - Preprocessing to match GEE `S1_GRD`: orbit metadata, thermal/border noise,
    radiometric calibration, terrain correction to the map grid; convert to dB.
  - Edge mask: invalidate pixels `< -30.0` in the SAR bands.
  - Expected domain ≈ `[-50, 1]` dB for VV/VH; valid thresholds `VV/VH >= -50`,
    `angle >= 0`. Missing acquisition → `-9999`.
  - `spatial_kind="high"`.
- **S1 sparsity (FDD §3 dominant risk):** present on ~16 dates → many windows have zero
  S1 timesteps → full `-9999` for the S1 group. The model masks them; this is normal.
- **Relevant skills:** `geospatial` (SAR calibration, terrain correction, windowed
  reads), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_s1_adapter.py` (Red): golden-grid triple; `bands_out = [VV, VH,
      angle]`; pixels `< -30.0` masked; domain ≈ `[-50, 1]`; parity vs reference within
      TASK-005 tolerance; missing day → all-`-9999`; windowed read (assert the full scene
      is not loaded — e.g. peak memory / windowed-read call assertion).
- [ ] 2. Implement `s1.py`: windowed GRD read of the cell footprint, calibrate +
      terrain-correct to the map grid, dB conversion, edge mask, reproject, stack
      `(3, H, W)`.
- [ ] 3. Wire into exporter (replace spike). 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Windowed reads (`rasterio` windows over SAFE measurement TIFFs); SAR
  calibration/terrain correction documented (SNAP/`pyroSAR`/`sarsen` — state which);
  tolerance from TASK-005.
- **Business:** Edge mask `< -30.0` is non-negotiable. Match GEE dB domain. Full-scene
  loads are forbidden (memory).
- **Out of scope:** Other sources; the inference driver/mosaic (TASK-015).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-14): `bands_out = [VV, VH, angle]`; `< -30.0` masked; domain ≈
      `[-50, 1]`; parity within tolerance.
- [ ] AC-2 (SPEC AC-12): golden-grid triple; band order correct.
- [ ] AC-3 (SPEC AC-13): missing `(S1, day)` → all-`-9999` (the common case).
- [ ] AC-4: windowed read verified (no full-scene load).
- [ ] AC-5: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_s1_adapter.py -v
uv run ruff check src/data/local_sources/s1.py
uv run mypy src/data/local_sources/s1.py
```
Expected: adapter test green (edge mask + windowed read + parity); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/s1.py tests/test_local_sources/test_s1_adapter.py
   git commit -m "feat(bow-valley): Sentinel-1 GRD adapter (edge mask, windowed reads) — closes TASK-014"
   ```
4. Check off subtasks/ACs. 5. Notify the user (all 9 adapters now production); request
   approval before TASK-015.
