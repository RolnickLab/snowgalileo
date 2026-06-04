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
    true ground pixel dimensions, then `create_ee_image`'s export resamples the
    `[DEM,slope,aspect]` image to the **cell grid** (`scale=10` + the export crs).
    The adapter must replicate that **two-step** order: (1) compute slope/aspect
    (Horn, degrees) **in the DEM's native frame** with the **correct
    metres-per-pixel in x/y at the cell's latitude**, then (2) resample
    DEM+slope+aspect to the cell grid. Match GEE within tolerance.
  - **⚠️ CELL-GRID CRS CORRECTED 2026-06-04 — the resample TARGET is now
    `EPSG:32611` (UTM 11N), 100×100, NOT `EPSG:4326`.** The cell grid is UTM 11N
    (see PLAN §3 Grid+CRS table / `docs/agents/KNOWLEDGE.md`; the GEE *inference*
    patches from `export_from_csv_utm` are UTM, confirmed against
    `data/eval_tifs/LC09_*`). So step (2) resamples to the `GridCell`'s UTM
    transform via `base.reproject_to_cell`, not 4326. This does **not** change
    step (1): terrain is still computed in the DEM's **native** lat-aware frame.
  - **DO NOT compute terrain in EPSG:32611 either.** This rule stands, but its
    *original justification was stale*: it said "the reference patches were never
    computed in a UTM frame," which conflated the **terrain-computation frame**
    (native DEM grid — unchanged by the correction) with the **cell/export grid
    CRS** (now UTM). The correct reason to keep terrain in the native frame is that
    `ee.Terrain` does — computing slope/aspect *in a UTM grid* would diverge from
    `ee.Terrain`'s native-frame result **regardless** of the final cell CRS. The
    bug to avoid is still running the Horn kernel on a degree grid with unit
    (`1°≈1 m`) pixel spacing (→ gradients ×111,000, all slopes → 90°) — a
    *pixel-spacing* error fixed by correct metric spacing, not a CRS change.
    (See REVIEW_AUDIT.md validation verdict #1.)
  - Valid thresholds: `DEM >= 0.0000001`, `slope >= 0`, `aspect >= 0`; invalid → `-9999`.
  - Identity normalization downstream.
- **Relevant skills:** `geospatial` (terrain derivatives, reprojection order), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_dem_adapter.py` (Red): golden-grid triple; `bands_out ==
      [DEM, slope, aspect]`; slope/aspect match GEE reference within tolerance;
      degenerate guard (slopes NOT all ≈90°); `day` ignored.
- [ ] 2. Implement `dem.py`: mosaic tiles → Horn slope/aspect in degrees **in the
      DEM's native frame** with latitude-correct metric pixel spacing (matching
      `ee.Terrain`) → resample DEM+slope+aspect (bilinear) to the cell's
      **EPSG:32611 100×100** grid via `base.reproject_to_cell` → stack `(3, H, W)`;
      `spatial_kind="space"`, `native_fill=None`. Do NOT compute terrain in any
      projected grid (native frame only); the UTM target applies to the resample
      step only.
- [ ] 3. Wire into exporter, replace placeholder. 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Bilinear for elevation; slope/aspect via Horn (or `richdem`/`gdaldem`
  equivalent) computed on the **native DEM grid** (NOT the reprojected grid),
  then resampled to the UTM cell grid; tolerance constant logged.
- **Business:** Static (ignores `day`). **Derivative-before-reprojection order is
  non-negotiable** (compute terrain in the native lat-aware frame, *then* resample
  to the cell grid — matching `ee.Terrain` + export).
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
