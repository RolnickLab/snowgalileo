# TASK-007: Implement the Copernicus DEM adapter (elevation + slope + aspect)

## 1. Goal
Replace the DEM placeholder with a real adapter that mosaics Copernicus DEM GLO-30
tiles and emits `[DEM, slope, aspect]`, computing slope/aspect with
**latitude-correct metric pixel spacing** (matching GEE `ee.Terrain`), then
resampling to the 10 m cell grid.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #2 — static).
- **SPEC:** FR-15, AC-12, AC-21; Verification Plan step 6.
- **PLAN:** §4 adapter rule ("compute slope/aspect with latitude-correct metric pixel
  spacing matching `ee.Terrain`, THEN resample to the 10 m cell grid; do NOT detour
  through EPSG:32611"), §6 FMEA ("DEM slope/aspect scale sensitivity").
- **Upstream tasks:** TASK-002 (clipped DEM), TASK-003, TASK-001 (GEE reference patches
  for the slope/aspect parity assertion).
- **Source semantics (DATA_ANALYSIS.md §Copernicus DEM):**
  - Clipped GeoTIFF, `EPSG:4326`, single `DEM` band (`float32`), elevation in metres.
  - **Parity target = GEE.** `src/data/earthengine/copernicus_dem.py:14-16` calls
    `ee.Terrain.slope`/`aspect` on the DEM's **native grid** using latitude-aware
    true ground pixel dimensions, then export resamples to 4326/scale=10. The
    adapter must replicate that: compute slope/aspect (Horn, degrees) with the
    **correct metres-per-pixel in x/y at the cell's latitude**, then resample
    DEM+slope+aspect to the 10 m cell grid. Match GEE within tolerance.
  - **DO NOT compute terrain in EPSG:32611.** A prior external review proposed a
    "compute in UTM 32611, reproject slope/aspect to 4326" detour — that diverges
    from the GEE reference patches (which were never computed in a UTM frame) and
    fails AC-21 parity. The bug to avoid is running the kernel on a degree grid
    with unit (`1°≈1 m`) pixel spacing (→ gradients ×111,000, all slopes → 90°);
    that is a *pixel-spacing* error fixed by correct metric spacing, not a CRS
    change. (See REVIEW_AUDIT.md validation verdict #1.)
  - Valid thresholds: `DEM >= 0.0000001`, `slope >= 0`, `aspect >= 0`; invalid → `-9999`.
  - Identity normalization downstream.
- **Relevant skills:** `geospatial` (terrain derivatives, reprojection order), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_dem_adapter.py` (Red): golden-grid triple; `bands_out ==
      [DEM, slope, aspect]`; slope/aspect match GEE reference within tolerance;
      degenerate guard (slopes NOT all ≈90°); `day` ignored.
- [ ] 2. Implement `dem.py`: mosaic tiles → Horn slope/aspect in degrees with
      latitude-correct metric pixel spacing (matching `ee.Terrain`) → resample
      DEM+slope+aspect (bilinear) to the 10 m cell grid → stack `(3, H, W)`;
      `spatial_kind="space"`, `native_fill=None`. Do NOT compute in EPSG:32611.
- [ ] 3. Wire into exporter, replace placeholder. 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Bilinear for elevation; slope/aspect via Horn (or `richdem`/`gdaldem`
  equivalent) on the reprojected grid; tolerance constant logged.
- **Business:** Static (ignores `day`). Reprojection-before-derivative order is
  non-negotiable.
- **Out of scope:** WorldCover (TASK-006), ERA5 (TASK-008).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-12): golden-grid `(transform, shape, crs)`; `bands_out` in order.
- [ ] AC-2 (SPEC AC-21): slope/aspect (latitude-correct metric spacing, matching
      `ee.Terrain`) within tolerance of GEE-derived values — NOT a 32611-computed
      reference; degenerate guard (slopes not all ≈90°); emits `[DEM, slope, aspect]`.
- [ ] AC-3: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_dem_adapter.py -v
uv run ruff check src/data/local_sources/dem.py
uv run mypy src/data/local_sources/dem.py
```
Expected: adapter test green (slope/aspect within tolerance); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/dem.py tests/test_local_sources/test_dem_adapter.py
   git commit -m "feat(bow-valley): Copernicus DEM adapter (elevation+slope+aspect) — closes TASK-007"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-008.
