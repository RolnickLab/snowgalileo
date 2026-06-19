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
- **Clipped S3 radiance was empty until the TASK-002 scale_factor fix (2026-06-03).**
  The S3 clip masks the swath to the AOI from `geo_coordinates.nc`, whose lat/lon are
  CF-scaled int32 (`scale_factor ≈ 1e-6`). The original clip compared the raw integers
  to degree bounds → empty mask → every `Oa*_radiance.nc` clipped to `(0,0)`, while the
  manifest still reported ~33 M valid pixels (from full-copied non-science datasets).
  Fixed; S3 re-clipped (radiance now `(N, M)` per overpass — validated 125/125 products
  non-empty, real swath windows rows 197–739 × cols 412–725, `Oa17` range ~480–48000
  uint16, `scale_factor ≈ 0.00493`). When building this adapter, the same
  `geo_coordinates` grids drive tie-point georeferencing — **decode their CF scaling**
  there too, or geolocation is silently wrong. See CLIPPING_PLAN §2.6 +
  `docs/agents/KNOWLEDGE.md`.
- **Do NOT trust the clip manifest `valid_pixel_count` as a science-pixel metric.** It
  counts valid pixels over *every* 2D dataset matching the geo grid plus full-copied
  non-science datasets (`removed_pixels`, `instrument_data`), so it is inflated by
  ~100× — a clipped S3 product shows ~50–59 M in the manifest while the actual
  `Oa17_radiance` swath holds only ~0.5 M valid pixels. This inflation is exactly what
  masked the `(0,0)` bug (the count stayed high while science bands were empty). When
  this adapter validates clipped inputs, count valid pixels on the **radiance array
  itself** (`arr != _FillValue`), never the manifest column.
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

> **As-built notes (2026-06-04).** (a) **Loose parity by design** — investigation (per
> GEE catalog) confirmed the radiance `scale_factor` 0.00493004 is *identical* to GEE's,
> and `geo_coordinates.nc` (full per-pixel) is the best geolocation input. The residual
> (~18% median, corr ~0.67) is OLCI's **un-orthorectified geolocation**; GEE
> terrain-orthorectifies in SNAP (catalog-confirmed) — the same SNAP-parity wall as S1.
> The test asserts georeferencing alignment with a loose bound (median |Δ| ≤ 60 + corr
> ≥ 0.4), NOT bit-exactness; SNAP ortho is a documented follow-up. (b) Used the **full
> per-pixel `geo_coordinates.nc`**, not the ×64-subsampled tie-point grid the goal text
> names. (c) SEN3 NetCDF is read via **h5py** (h5netcdf/xarray fail on these files'
> HDF5 dimension-scale refs). See PARITY_SPIKE_NOTES §10.

> **SNAP-ortho follow-up — REJECTED (2026-06-09).** The §50 "SNAP ortho is a documented
> follow-up" line is now **closed, not open.** After TASK-014 proved SNAP closes the S1
> parity wall, the OLCI-ortho hypothesis was re-tested with SNAP's actual optical ortho
> path (`Reproject orthorectify=true` + SRTM 1Sec; `src/data/local_sources/parity/s3.py`
> + `scripts/developer_scripts/bow_valley_inference_local/spikes/s3_olci_ortho_graph.xml`,
> both kept as evidence). It went the **wrong direction** vs
> the production `griddata` warp on the same patch/day/cell (10403 co-valid px): Oa17 corr
> 0.666→0.658, Oa21 0.783→0.774. The residual is therefore **not** terrain distortion but
> sampling geometry (patch ~3 OLCI px wide; ~300 m px on a ~1 km cell), and the `med` 5×5
> downsample erases any sub-pixel difference regardless. SNAP also can't read the
> **clipped** `.nc` (the h5py landmine), so ortho would force the raw product for zero
> gain. **The swath-warp adapter stays as shipped; the open S3 lever is normalization, not
> geolocation.** Full numbers + method: PARITY_SPIKE_NOTES §10.1.

## 3. Subtasks
- [x] 1. Write `test_s3_adapter.py` (Red): golden-grid triple; `bands_out =
      [Oa17_radiance, Oa21_radiance]`; tie-point-warped output aligns with the cell grid
      (assert against GEE reference patch within tolerance); identity scaling preserved;
      missing day → `-9999`.
- [x] 2. Implement `s3.py`: read radiance + `geo_coordinates.nc`, warp via tie points to
      the cell grid (bilinear), stack `(2, H, W)`; `spatial_kind="med"`.
- [x] 3. Wire into exporter. 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** `xarray`/`h5netcdf` for SEN3 NetCDF; tie-point interpolation/warp
  (e.g. `pyresample` swath def or GDAL geoloc arrays).
- **Business:** Preserve radiance scale (identity normalization downstream — out of
  scope to "fix" the S3 normalization TODO).
- **Out of scope:** S3 normalization fix, VIIRS (TASK-010), Landsat (TASK-012).

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-12): golden-grid triple; band order correct.
- [x] AC-2 (SPEC AC-17): tie-point georeferencing aligns to the cell grid; identity
      normalization preserved.
- [x] AC-3 (SPEC AC-13): missing `(S3, day)` → all-`-9999`.
- [x] AC-4: ruff + mypy clean; targeted new tests green; full suite introduces NO new failures vs `TEST_BASELINE.md` (delta check, NOT `pytest -x`).

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
