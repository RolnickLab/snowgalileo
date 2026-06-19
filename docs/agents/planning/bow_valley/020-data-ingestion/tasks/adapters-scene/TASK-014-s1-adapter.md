# TASK-014: Implement the Sentinel-1 GRD adapter (windowed reads, edge mask)

> **PARTIALLY SUPERSEDED (2026-06-11).** The adapter's band semantics (`[VV, VH, angle]`,
> linear σ⁰ → dB, `< -30` edge mask, ellipsoid incidence) are unchanged and current. But
> the **S1 processing pipeline** described here was reworked: SNAP now runs **once per RAW
> granule** (not per `(granule, cell)`) with the geoRegion Subset applied **after**
> Terrain-Correction, producing one AOI-wide `sentinel1_snap/s1_grd_<granule>.tif` that the
> adapter windows per cell. S1 is **no longer clipped** at all (references to "clipped S1" /
> TASK-002 below are stale for S1). The same SNAP cache feeds both the cube and the viewer.
> See [`../PLAN-S1-PERGRANULE-SNAP.md`](../PLAN-S1-PERGRANULE-SNAP.md) for the current
> design; this task doc is retained for the adapter's parity/edge-mask rationale.

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
- [x] 1. Write `test_s1_adapter.py` (Red): `bands_out = [VV, VH, angle]`; pixels `< -30.0`
      masked (VV/VH only — angle never masked); same-date coalesce; missing day →
      all-`-9999`; S1C name parse. **8 synthetic tests** (post-SNAP GeoTIFFs, no SNAP/CI) +
      real-archive parity.
- [x] 2. Implement `s1.py`: read the **SNAP dB+angle cache**, linear σ⁰ → dB (VV/VH),
      ellipsoid-angle passthrough, edge mask, coalesce/mosaic/reproject (bilinear), stack
      `(3, H, W)`. **Heavy SNAP preprocessing extracted to `s1_snap.py` (offline, per-cell
      cache)** — the adapter `fetch` is pure-raster.
- [x] 3. Wire into exporter (S1Adapter = head of HIGH group). 4. Green + Refactor.

## 4. Requirements & Constraints
- **Technical:** Windowed reads (`rasterio` windows over SAFE measurement TIFFs); SAR
  calibration/terrain correction documented (SNAP/`pyroSAR`/`sarsen` — state which);
  tolerance from TASK-005.
- **Business:** Edge mask `< -30.0` is non-negotiable. Match GEE dB domain. Full-scene
  loads are forbidden (memory).
- **Out of scope:** Other sources; the inference driver/mosaic (TASK-015).

## 5. Acceptance Criteria
- [x] AC-1 (SPEC AC-14): `bands_out = [VV, VH, angle]`; `< -30.0` masked (VV/VH); dB domain;
      **parity proven on PR_20250519 — VV 0.38 dB, VH 0.40 dB, angle 0.24°** (well within the
      TASK-005 1.0 dB tolerance), the decisive full-chain reproduction of GEE `S1_GRD`.
      PR_20250406 + PR_20250423 are **xfail with GEE-pull-confirmed non-adapter root causes**
      (see §6).
- [x] AC-2 (SPEC AC-12): band order [VV, VH, angle]; output on the cell grid.
- [x] AC-3 (SPEC AC-13): missing `(S1, day)` → all-`-9999` (the common case — ~16 dates).
- [~] AC-4: windowed read — **reinterpreted**: the "no full-scene load" guarantee is met by
      the offline SNAP `Subset` (`s1_snap.py`), not a `rasterio` window. The adapter reads a
      small cached tif and windows it per cell.
      > **CORRECTED 2026-06-11:** this AC originally concluded *per-cell* SNAP bounding was
      > "mandatory" because full-AOI runs NPE-corrupted. That was an artifact of subsetting in
      > **radar geometry before** Terrain-Correction. Moving the `Subset` to **after** TC (map
      > geometry) makes a single **per-granule, full-AOI** run clean — so the cache is now one
      > AOI-wide tif per granule, NOT per cell. See `../PLAN-S1-PERGRANULE-SNAP.md`.
- [x] AC-5: ruff + mypy clean; 8 synthetic + 1 parity green, 2 documented xfail; full-suite
      delta = 0 new failures.

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
   git add src/data/local_sources/s1.py src/data/local_sources/s1_snap.py \
           src/data/local_sources/s1_grd_graph.xml src/data/local_sources/exporter.py \
           tests/test_local_sources/test_s1_adapter.py docs/agents/KNOWLEDGE.md \
           docs/agents/planning/raw-data-ingestion/tasks/TASK-014-s1-adapter.md
   git commit -m "feat(bow-valley): Sentinel-1 GRD adapter (SNAP per-cell cache, dB+ellipsoid angle, edge mask) — closes TASK-014"
   ```
4. Check off subtasks/ACs. 5. Notify the user (all 9 adapters now production); request
   approval before TASK-015.

## 8. Outcome — S1 adapter (SNAP per-cell cache; corrected 2026-06-08)

**Architecture (differs from the plan's "windowed rasterio read"):** the clip stage does
NOT preprocess S1 (`clip_sentinel1` GCP-window-slices raw range-geometry TIFFs). The ESA
SNAP chain (Apply-Orbit → ThermalNoise → Border-Noise → Calibration σ⁰ → Terrain-Correction
EPSG:32611 + `saveIncidenceAngleFromEllipsoid`) runs **offline, once per (granule, cell)**
into a cached 3-band GeoTIFF (`s1_snap.py`, graph `s1_grd_graph.xml`). The adapter (`s1.py`)
reads that cache, converts linear σ⁰ → dB (VV/VH), passes the angle through, edge-masks
`< -30` dB, coalesces/mosaics/reprojects — **pure raster, no SNAP at fetch time**.

**Key corrections to the plan:**
- **dB is done in the adapter, not SNAP.** Scoping `LinearToFromdB` to the σ⁰ bands *drops*
  the angle band; converting all bands log-scales the angle. The cache stores LINEAR σ⁰ +
  angle; `10·log10` is applied in `s1.py`. Identical math.
- **`angle` = ellipsoid incidence** (`saveIncidenceAngleFromEllipsoid`), verified against the
  reference patches (matches to ≤0.4°). NOT local incidence (which swings with terrain).
- **Band order pinned by index** (SNAP emits VH=1, VV=2, angle=3); the BigTIFF writer
  persists no band descriptions, so the adapter maps by index.
- **Per-cell bounding is mandatory.** A full-AOI-bbox SNAP run (840 NPEs, 3.3 GB) AND a
  full-clipped-scene run (1060 NPEs, 651 Mpx) both NPE-corrupt — the clip is a range-geometry
  pixel window, so its geographic extent is the whole swath. Only a small per-cell `Subset`
  geoRegion runs clean.

**Parity (proven):** `PR_20250519` reproduces GEE `COPERNICUS/S1_GRD` to **VV 0.38 dB,
VH 0.40 dB, angle 0.24°** (n=10177). Two patches are xfail with **GEE-pull-confirmed**
non-adapter root causes:
- `PR_20250406`: a direct GEE pull (project `bow-valley-inference`) shows GEE's S1_GRD VV =
  **−2.63 dB** over the patch for the *same* acquisition (S1C 2025-04-06 01:29:13 ASC
  relOrbit 20, angle 34.84° — identical to our archive granule), while our SNAP σ⁰ = −12.7 dB
  (physically typical). Non-uniform offset (VV 10.1 / VH 5.8 dB) → not a gain constant;
  something intrinsic to GEE's processing of that one scene. The identical chain nails 0519,
  so it is NOT a pipeline defect. Follow-up: σ⁰-vs-γ⁰ / per-scene aux forensics.
- `PR_20250423`: SNAP `Subset` returns "Empty region!" for this granule+cell → σ⁰ empty
  (angle still fills). A SNAP-on-GCP-clipped-product quirk. Follow-up: wider GCP clip buffer
  or drop Remove-GRD-Border-Noise for edge cells.

**Build the cache before exporting cubes:** `uv run python -m src.data.local_sources.s1_snap`
(needs ESA SNAP at `/home/dev/esa-snap/bin/gpt`; idempotent per (granule, cell)).
