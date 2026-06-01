# TASK-011: Implement the Sentinel-3 OLCI adapter (tie-point geolocation)

## 1. Goal
Replace the S3 placeholder with a real adapter that emits `[Oa17_radiance,
Oa21_radiance]` on the cell grid, georeferenced via the OLCI tie-point coordinate
grids, with identity normalization preserved.

## 2. Context & References
- **FDD step:** §4.6 (adapter order #6).
- **SPEC:** FR-7, AC-12, AC-13, AC-17; Verification Plan step 6.
- **PLAN:** §4 adapter rule ("S3 OLCI geolocation via tie-point grids"), §3 archive
  formats (SEN3 NetCDF: `Oa17_radiance.nc`, `Oa21_radiance.nc`, `geo_coordinates.nc`).
- **Upstream tasks:** TASK-002 (clipped S3), TASK-003, TASK-004.
- **Source semantics (DATA_ANALYSIS.md §Sentinel-3 OLCI):**
  - SEN3 NetCDF; radiance bands in separate `.nc` files; **tie-point/coordinate grids
    in `geo_coordinates.nc`** must drive georeferencing — naive geolocation causes
    significant misalignment vs GEE's orthorectified OLCI.
  - Emit only `Oa17_radiance, Oa21_radiance`.
  - Identity normalization downstream (`shift=[0,0]`, `div=[1,1]`) — preserve radiance
    units/scaling as exported by GEE; any source-side scale flows into the model.
  - `spatial_kind="med"` (loader downsamples to 5×5). Valid `>= -1`. Missing → `-9999`.
  - ~300 m, swath geometry; usually full AOI when present (1270 km swath).
- **Relevant skills:** `geospatial` (tie-point warping, swath reprojection), `tdd`.

## 3. Subtasks
- [ ] 1. Write `test_s3_adapter.py` (Red): golden-grid triple; `bands_out =
      [Oa17_radiance, Oa21_radiance]`; tie-point-warped output aligns with the cell grid
      (assert against GEE reference patch within tolerance); identity scaling preserved;
      missing day → `-9999`.
- [ ] 2. Implement `s3.py`: read radiance + `geo_coordinates.nc`, warp via tie points to
      the cell grid (bilinear), stack `(2, H, W)`; `spatial_kind="med"`.
- [ ] 3. Wire into exporter. 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** `xarray`/`h5netcdf` for SEN3 NetCDF; tie-point interpolation/warp
  (e.g. `pyresample` swath def or GDAL geoloc arrays).
- **Business:** Preserve radiance scale (identity normalization downstream — out of
  scope to "fix" the S3 normalization TODO).
- **Out of scope:** S3 normalization fix, VIIRS (TASK-010), Landsat (TASK-012).

## 5. Acceptance Criteria
- [ ] AC-1 (SPEC AC-12): golden-grid triple; band order correct.
- [ ] AC-2 (SPEC AC-17): tie-point georeferencing aligns to the cell grid; identity
      normalization preserved.
- [ ] AC-3 (SPEC AC-13): missing `(S3, day)` → all-`-9999`.
- [ ] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

## 6. Testing & Validation
```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_s3_adapter.py -v
uv run ruff check src/data/local_sources/s3.py
uv run mypy src/data/local_sources/s3.py
```
Expected: adapter test green (tie-point alignment within tolerance); ruff/mypy exit 0.

**Regression check (suite is already red):** run the delta check in `TEST_BASELINE.md` — the "NEW failures" list must be empty. Do NOT use `pytest -x` at the suite level.

## 7. Completion Protocol
1. Verify ACs. 2. Run Section 6 commands.
3. Commit:
   ```bash
   git add src/data/local_sources/s3.py tests/test_local_sources/test_s3_adapter.py
   git commit -m "feat(bow-valley): Sentinel-3 OLCI adapter (tie-point geolocation) — closes TASK-011"
   ```
4. Check off subtasks/ACs. 5. Notify the user; request approval before TASK-012.
