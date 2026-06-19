# S1 / S2 Parity Spike Notes (TASK-005)

*Formerly `PARITY_SPIKE_NOTES.md`.*

> **Purpose** — de-risk the two highest value-domain-drift sources (S1 GRD, S2
> L1C) against the Phase-0 GEE reference patches *before* building the full
> adapter stack (TASK-012/013/014). This file records the spike design, the
> toolchain decision, the chosen tolerances, the measured per-band drift, and
> the **go/no-go verdict** (SPEC AC-3 / AC-14 / AC-15, spike form).
>
> The parity logic now lives in `src/data/local_sources/parity/` with thin CLI
> wrappers under `scripts/developer_scripts/bow_valley_inference_local/spikes/`
> (the original one-shot `scripts/spikes/` scripts were promoted there once a test
> took a dependency on `run_s2_spike`). They are **de-risk tools** — they exist to
> measure drift, not to be the production path. The real adapters
> (TASK-012/013/014) re-implement the recovered recipe.

## 0. Status

- [x] S1 spike run, drift recorded — **within tolerance (GO)**, §4
- [x] S2 spike run, drift recorded — **within tolerance (GO)**, §4
- [x] Go/no-go verdict stated (§5) — **GO** for both

(Checked off as each completes; numbers filled in §4.)

## 1. The parity target

`tests/fixtures/gee_reference_patches/PR_*.tif` — 6 GEE-exported 308-band cubes,
**EPSG:32611**, ~103×101 px (GEE `export_from_csv_utm` slightly over-exports; the
loader's `subset_image` crops the ≥100 surplus to 100×100). Per-timestep band
layout (from `layout.full_band_order()`):

| Source | bands | 1-based index at timestep *t* |
|--------|-------|-------------------------------|
| **S1** | `VV, VH, angle`             | `38·t + 1 … 38·t + 3`  |
| **S2** | `B2, B3, B4, B8, B11, B12`  | `38·t + 4 … 38·t + 9`  |

The spike reads the reference patch's S1/S2 band slice for the matching
timestep and diffs the spike output against it on the **same UTM cell grid**
(reproject both to the patch's `(crs, transform, 100×100)` via
`base.reproject_to_cell`, then per-band diff over valid pixels).

## 2. What `COPERNICUS/S1_GRD` actually is (the recipe we must recreate)

GEE's `COPERNICUS/S1_GRD` is **Level-1 GRD after the Sentinel-1 Toolbox (SNAP)
chain**, in order:

1. Apply orbit file (precise/restituted ephemeris).
2. GRD border-noise removal.
3. Thermal-noise removal.
4. Radiometric calibration → σ⁰ (sigma-nought).
5. **Range-Doppler terrain correction** (orthorectify to a DEM; SRTM in GEE).
6. Convert to dB (`10·log10`).

Bands `[VV, VH, angle]`; IW mode; expected domain ≈ `[-50, 1]` dB for VV/VH;
edge mask `< -30 dB`.

### Toolchain decision (2026-06-04): **ESA SNAP `gpt` — the engine GEE itself uses**

**First attempt (`sarsen`) — abandoned for S1C.** `sarsen` 0.9.5 (+ `xarray-sentinel`)
does calibration + range-Doppler terrain correction, but `xarray-sentinel` 0.9.5
(the latest stable) **cannot read Sentinel-1C SAFEs** — the Bow Valley archive is
all `S1C_*`. Two distinct S1C bugs: (1) the `s1[ab]` filename regex rejects `s1c`
(one-char shim possible), and (2) the GCP-annotation reader returns a zero-size
array despite 210 valid GCP points in the XML, crashing
`get_footprint_linestring`. The second is in the geolocation core terrain
correction depends on; progressively patching the library's internal XML readers
is library-porting, not de-risking. **Dead end for S1C.** (`sarsen`/`xarray-sentinel`
remain in the `spikes` dep group but are NOT the working route.)
See `docs/.../memory` note `xarray-sentinel-s1c-regex-bug`.

**Working route — ESA SNAP (user installed 2026-06-04).** GEE's
`COPERNICUS/S1_GRD` *is* the output of the SNAP Sentinel-1 Toolbox, so running
SNAP via headless `gpt` reproduces the reference recipe with the **same engine**,
and SNAP fully supports S1C. The graph
(`scripts/developer_scripts/bow_valley_inference_local/spikes/s1_grd_snap_graph.xml`)
chains **all six steps** — Apply-Orbit →
**ThermalNoiseRemoval → Remove-GRD-Border-Noise** → Calibration(σ⁰) →
Terrain-Correction(**SRTM 1Sec**, EPSG:32611, 10 m) → LinearToFromdB — so the
border/thermal-noise gap that `sarsen` could not cover **is closed**. The verdict
is therefore unconditional (no "noise steps unmeasured" caveat).

Run notes: `gpt` is at `/home/dev/esa-snap/bin/gpt` (NOT `/usr/bin/snap`, which is
Ubuntu snapd). The graph **subsets to the AOI in radar geometry right after
Apply-Orbit-File** — terrain-correcting the full ~250 km IW swath at 10 m overflows
SNAP's classic-GeoTIFF 4 GB writer limit and wastes compute; the AOI subset is
both the fix and a large speedup. SNAP auto-downloads the SRTM 1Sec + EGM96 tiles
on first run. **SNAP writes the bands VH-then-VV** (not the `VV,VH` graph order) —
the spike assigns them by matching against the reference medians, not by index.

## 3. What `COPERNICUS/S2_HARMONIZED` is (S2 recipe)

For processing baseline **≥ N0400** (all archive granules are **N0511**, verified
from the SAFE names), `S2_HARMONIZED` is L1C DN with a **−1000 DN offset** applied
(the `RADIO_ADD_OFFSET`), reflectance = `DN / 10000` downstream. No atmospheric
correction (L1C TOA). Bands `[B2,B3,B4,B8,B11,B12]`; B11/B12 are 20 m (resampled
to the 10 m cell grid). Fully recreatable with `rasterio` JP2 reads — no external
toolchain.

## 4. Tolerances & measured drift

**Chosen tolerances** (explicit constants, also in the spike scripts + tests):

| Source | Metric | Tolerance | Rationale |
|--------|--------|-----------|-----------|
| **S1** VV/VH | median abs diff | **≤ 1.0 dB** | SNAP-chain dB values; sub-dB agreement is the bar for a missing-noise-step spike with the −30 dB mask. |
| **S1** angle | median abs diff | **≤ 1.0°** | incidence angle is geometry-only, should agree tightly. |
| **S2** B2…B12 | median abs diff | **≤ 50 DN** (post −1000) | harmonized DN domain ~0–10000; 50 DN = 0.005 reflectance, well under model normalization sensitivity. |

**Measured drift** (filled by the spike run — the `run_*_parity.py` wrappers under
`scripts/developer_scripts/bow_valley_inference_local/spikes/` emit `structlog`):

_S1 (per band, vs reference, valid pixels, −30 dB masked). Cell
``PR_20250406…5653083.8`` t0, date 2025-03-30, granule ``S1C…88AD``; SNAP `gpt`
full S1_GRD chain → EPSG:32611 → reprojected to patch grid:_

| band | median \|Δ\| | p95 \|Δ\| | within tol (≤1.0 dB median)? |
|------|--------------|-----------|------------------------------|
| VV   | **0.54 dB**  | 2.32 dB   | ✅ |
| VH   | **0.48 dB**  | 2.11 dB   | ✅ |
| angle| not emitted  | —         | n/a (see note) |

**Interpretation.** VV/VH agree with GEE's `COPERNICUS/S1_GRD` to **sub-dB**
(median ~0.5 dB) using the full SNAP chain — including thermal/border-noise
removal, so this is unconditional, not the conditional-GO fallback. p95 ~2.3 dB
is the speckle/edge tail (SAR is inherently noisy pixel-to-pixel); the gate uses
the median. **`angle`** (incidence) is a deterministic geometry band, not a
value-domain drift risk; this graph did not request the TC
`projectedLocalIncidenceAngle` output, so it is not in the spike. The reference
angle is ~43.6° (near-constant over the 1 km patch); the real adapter (TASK-014)
recovers it by enabling that TC output. **S1 = GO.**

_S2 (per band, post −1000 DN). Cell ``PR_20250406…5653083.8`` t4, date 2025-04-03,
tile ``T11UNS``; valid (non-0, non-−9999) reference pixels:_

| band | median \|Δ\| (DN) | p95 \|Δ\| (DN) | within tol (≤50 median)? |
|------|-------------------|----------------|--------------------------|
| B2   | **0.00**          | 0.00           | ✅ exact |
| B3   | **0.00**          | 0.00           | ✅ exact |
| B4   | **0.00**          | 0.00           | ✅ exact |
| B8   | **0.00**          | 0.00           | ✅ exact |
| B11  | 19.38             | 86.19          | ✅ |
| B12  | 19.44             | 88.38          | ✅ |

**Interpretation.** The four 10 m bands (B2/B3/B4/B8) are **bit-exact** vs GEE: the
−1000 DN offset perfectly reproduces the harmonized domain and no resampling is
needed onto the 10 m cell grid. B11/B12 are native 20 m upsampled to 10 m; their
~19 DN median residual is **resampling-kernel difference** (spike bilinear vs GEE
reprojection), not a domain error — 19 DN ≈ 0.002 reflectance. p95 (~87 DN) exceeds
50 only at sharp-edge pixels where kernels diverge most; the gate uses the median
(correct central-tendency metric for domain drift). **S2 = GO.**

## 5. Go / no-go verdict

**GO for both S1 and S2.** Both sources' value domains are recoverable within
the stated tolerances against the GEE reference patches:

- **S2_HARMONIZED** — −1000 DN offset is exact for the 10 m bands (0.00 DN);
  20 m B11/B12 carry only sub-tolerance resampling noise (~19 DN median). → TASK-013.
- **S1_GRD** — the full SNAP chain (the engine GEE uses) reproduces VV/VH to
  ~0.5 dB median, including the noise-removal steps. → TASK-014.

**Proceed to TASK-006.** No escalation needed.

**Recipe hand-off to the production adapters:**
- **TASK-013 (S2):** read L1C JP2, subtract 1000 DN (N0400+), reproject to the
  cell grid. Trivial; bit-exact for native-res bands.
- **TASK-014 (S1):** the adapter must run the SNAP `gpt` chain (or an equivalent
  that includes orbit + thermal + border-noise + calibration + RD-terrain-
  correction + dB). **It cannot use `xarray-sentinel`/`sarsen`** for S1C ingest
  (see §2). Subset to the cell AOI before TC (4 GB writer limit + speed). Enable
  the TC incidence-angle output for the `angle` band. SNAP emits bands VH-then-VV;
  assign by name, not index.

---

## 6. DEM terrain parity (TASK-007, 2026-06-04)

Validated the Copernicus DEM adapter recipe against the **DEM/slope/aspect bands
(305/306/307)** of all six Phase-0 GEE reference patches.

**Recipe (matches `ee.Terrain` + `create_ee_image` export):**
1. Mosaic the clipped GLO-30 tiles in their **native EPSG:4326** frame (+0.05°
   margin so the Horn kernel has edge neighbours).
2. Compute slope/aspect with a 3×3 **Horn** kernel using **latitude-correct metric
   pixel spacing**: `dy = yres·M_PER_DEG`, `dx = xres·M_PER_DEG·cos(lat)`, where
   `M_PER_DEG = 2πR/360`, `R = 6378137 m`. (GLO-30 longitude spacing is thinned
   poleward — the degree grid is anisotropic; raw-degree spacing would inflate
   gradients ×111 000 → all slopes ≈90°.) Aspect = `(450 − atan2(dz_dy,−dz_dx)) mod 360`.
3. Resample DEM + slope + aspect to the cell's EPSG:32611 grid with **NEAREST**.

**Resampling decision — NEAREST, not bilinear (deliberate base-convention deviation).**
GEE computes terrain at the native ~30 m scale then upsamples to the 10 m export
grid; nearest replicates that pixel reuse. Measured per-patch medians (interior,
5 px border dropped):

| patch | tiles | DEM med (m) | slope med (°) | slope p95 (°) | aspect med (°) |
|---|---|---|---|---|---|
| 0406 | 2 | 0.000 | 0.941 | 6.09 | 2.20 |
| 0414 | 1 | 0.217 | 0.569 | 6.06 | 5.98 |
| 0423 | 1 | 0.000 | 0.828 | 4.78 | 1.97 |
| 0502 | 1 | 0.000 | 0.435 | 2.85 | 7.43 |
| 0510 | 2 | 0.491 | 1.205 | 6.12 | 10.01 |
| 0519 | 2 | 0.176 | 0.928 | 5.08 | 10.44 |

Bilinear roughly **doubles** the slope error (median ~1.7°, p95 ~6.5° on patch 0406)
and adds DEM smoothing bias — confirming nearest is the parity-correct final step.
Aspect medians run higher (≤10.4°) because aspect is circular and unstable on
near-flat pixels; the test diffs it on the unit circle.

**Test tolerances (`tests/test_local_sources/test_dem_adapter.py`):** DEM median
≤1.0 m, slope median ≤1.5°, aspect median (circular) ≤12°, plus a degenerate
guard (slopes not near-uniformly ≈90°). **DEM = GO; adapter shipped (TASK-007).**

---

## 7. ERA5-Land parity (TASK-008, 2026-06-04)

Validated the ERA5 adapter against the t2m/precip bands of the Phase-0 GEE
reference patch `PR_20250406` across the 8-day window.

**Recipe (matches GEE `ECMWF/ERA5_LAND/DAILY_AGGR`):**
1. Read the **already-daily** archive — one slice/day, no hourly re-aggregation.
   Instantaneous vars from the monthly `YYYYMM_ERA5LAND/` folder (`skt`, `t2m`,
   `u10`, `v10`); precip from `YYYYMM_ERA5LAND_totalprecip.nc` (`tp`, `accum`).
2. **Precip `i+1` day-shift:** precip for inference day `d` = the `tp` slice stamped
   `00:00` of `d+1` (the accumulation closing day `d`). Resolved by `valid_time`
   lookup **across month files** (April-30 reads the May precip file). Instantaneous
   vars are unshifted (slice labelled `d`). Verified: shifted matches GEE (0.0),
   same-day read gives 0.0029 — wrong day.
3. Raw units (Kelvin/m/s/m). The temperature Kelvin→Celsius shift is the downstream
   `Normalizer`'s job, NOT the adapter.

**Resample = NEAREST (supersedes the spec's "bilinear" text).** The 0.1° (~11 km)
grid is far coarser than the 1 km cell — GEE upsamples it as a constant block per
ERA5 cell. Measured medians across 8 timesteps vs the reference patch:

| resample | t2m med (K) | precip med (m) |
|---|---|---|
| **nearest** | **0.0001** | **0.0001** |
| bilinear | 0.2565 | 0.0001 |

Nearest is essentially exact (the patch's valid t2m is a single constant 268.32017 K,
reproduced bit-for-bit). Bilinear smears across ERA5-cell boundaries for no benefit.
Routed through `base.reproject_to_cell(categorical=True)` (its nearest path).

**Test tolerances (`tests/test_local_sources/test_era5_adapter.py`):** t2m median
≤0.01 K, precip median ≤0.001 m, plus a deterministic synthetic-NetCDF day-shift
test (incl. the cross-month boundary) and a raw-Kelvin guard. **ERA5 shipped (TASK-008).**

---

## 8. MODIS MOD09GA parity (TASK-009, 2026-06-04)

Validated the MODIS adapter against the `sur_refl_b01` band of the Phase-0 GEE
reference patch `PR_20250406` across all 8 timesteps (DOY 089–096).

**Recipe (matches GEE `MODIS/061/MOD09GA`):**
1. Read the clip stage's per-band sinusoidal GeoTIFFs directly (no HDF4 driver):
   `MODIS_Grid_500m_2D__sur_refl_bNN_1.tif` (science) + `MODIS_Grid_1km_2D__state_1km_1.tif`
   (cloud). Each grid keeps its own resolution/transform — never hardcode 1200/2400.
2. Mosaic per-tile GeoTIFFs by acquisition date `A{YYYY}{DOY}` (this AOI is single-tile
   `h10v03`; the mosaic path holds for cells crossing a seam).
3. Reproject sinusoidal (`+R=6371007.181`) → EPSG:32611 cell grid with **NEAREST**.
4. **Preserve `-28672`** (`restore_fill=-28672` on the science bands) — the loader
   sentinel. Do NOT apply the scale factor. `state_1km` is categorical (NN, bit-flag).

**Resample = NEAREST (supersedes the spec's "nodata-aware bilinear").** The 500 m grid
is far coarser than the 10 m cell — GEE upsamples it as a constant block per MODIS
pixel. Per-timestep median |Δ| vs the reference patch (sur_refl_b01):

| resample | per-ts median \|Δ\| (DN) | mean |
|---|---|---|
| **nearest** | 0,0,0,0,0,0,0,0 | **0** |
| bilinear | 350,34,225,172,201,941,441,215 | 322 |
| cubic | 347,28,189,124,134,791,373,224 | 276 |
| average | 0×8 | 0 |

**Nearest is bit-exact.** It also makes the `-28672` edge-bleed risk (the reason the
spec mandated nodata-aware bilinear) *impossible* — nearest never interpolates across
the fill boundary, so no garbage negative can appear. Routed through
`base.reproject_to_cell(categorical=True, src_nodata=-28672, restore_fill=-28672)`.

**Test (`tests/test_local_sources/test_modis_adapter.py`):** 8 bit-exact parity
timesteps (median == 0), `-28672` preservation, a no-bleed guard, missing-day →
`-9999`, and the `state_1km` cloud path. **MODIS shipped (TASK-009).**

This is the **third** coarse source (after DEM §6 and ERA5 §7) where GEE's
`create_ee_image` export resamples by nearest, not bilinear. Treat nearest as the
default for any source coarser than the 10 m cell; reserve bilinear/aware-bilinear for
sources at or finer than 10 m (S2/Landsat/S1).

---

## 9. VIIRS VNP09GA parity (TASK-010, 2026-06-04)

Validated the two VIIRS adapters against the I1 (fine) and M5 (coarse) bands of the
Phase-0 reference patch `PR_20250406` across all 8 timesteps.

**Recipe (matches GEE `NASA/VIIRS/002/VNP09GA`):**
1. Read the clip stage's per-band sinusoidal GeoTIFFs (no HDF5 driver):
   `VIIRS_Grid_500m_2D__SurfReflect_I{1,3}_1.tif` (fine),
   `VIIRS_Grid_1km_2D__SurfReflect_M{5,7,10,11}_1.tif` (coarse).
2. **Scale x0.0001 (reflectance).** Unlike MODIS (raw DN), GEE exports VNP09GA as
   reflectance — confirmed by the normalizer `(x+0.795)/0.805` and a DN/ref ratio of
   exactly 10000. Scale valid pixels; restore `-28672` **after** scaling (the fill is
   never scaled), preserving the loader sentinel.
3. Reproject sinusoidal → EPSG:32611 cell grid with **NEAREST** (coarse source; bit-exact).
4. **Coarse stays a per-pixel `(4, H, W)` raster** — the loader spatial-means it into
   `time_x`; the adapter never pre-averages (pre-averaging biases the mean over the
   diagonal-band clip's nodata and breaks `time_x`).

Per-timestep scaled median |Δ| vs the reference patch: **0.000000 for all 8 timesteps,
both I1 and M5** (bit-exact; residual ~1e-8 in the assembled cube is float32 rounding).

**MODIS vs VIIRS scale contrast (important):** MOD09GA ref values are raw DN (~7464);
VNP09GA ref values are reflectance (~0.55). GEE keeps MODIS as DN but scales VIIRS.
Each adapter matches its own source's GEE domain — do not unify them.

**Test (`tests/test_local_sources/test_viirs_adapter.py`):** 16 bit-exact parity
timesteps (fine + coarse), reflectance-domain guard, per-pixel-raster guard (coarse is
`(4,H,W)` not pre-averaged), `-28672` preservation, missing-day → `-9999`.
**VIIRS shipped (TASK-010).** Fourth coarse source confirming the GEE-nearest rule.

---

## 10. Sentinel-3 OLCI parity (TASK-011, 2026-06-04)

Investigated the S3 adapter against Oa17_radiance of the reference patch (t2 =
2025-04-01, the day an overpass covers the cell). **This source does NOT reach
bit-exactness** — the reasons are understood and the residual is out of scope.

**Recipe (`src/data/local_sources/s3.py`):**
1. Read SEN3 NetCDF via **h5py** directly (h5netcdf/xarray cannot resolve these files'
   HDF5 dimension-scale references — confirmed by traceback).
2. **Decode CF scaling.** Radiance: uint16 × `scale_factor` (Oa17 0.00493004, Oa21
   0.00324118). Geolocation: `geo_coordinates` lat/lon int32 × **1e-6** (the landmine
   that once clipped radiance to (0,0); decode before any geographic use).
3. Pick the overpass covering the cell (most valid pixels over the footprint), warp the
   curvilinear swath → cell UTM grid with `scipy.griddata(method="nearest")`.

**Investigation of the ~18% offset (per user request — checked the GEE catalog):**
- **Scale is correct.** The GEE `COPERNICUS/S3/OLCI` catalog lists Oa17 scale
  **0.00493004** — identical to the file `scale_factor`. The offset is NOT calibration.
- **Geolocation input is the best available.** `geo_coordinates.nc` is the **full
  per-pixel** grid (731×716, matches the radiance grid), better than the ×64-subsampled
  `tie_geo_coordinates.nc` (4091×77). Decoded ×1e-6 correctly.
- **Residual = OLCI's un-orthorectified geolocation.** Best result (S3A overpass):
  corr **0.666**, median |Δ| **35.4** (my median 152 vs GEE 192). A rigid grid shift
  (search ±600 m) lifts corr only to 0.73 at +150 m N and does NOT close median |Δ| —
  so it is per-pixel terrain distortion, not a translation. The GEE catalog confirms
  **SNAP** processing; GEE terrain-orthorectifies OLCI, which a plain swath-warp cannot
  reproduce (the **same SNAP-parity wall as S1**, §1–5). The reference patch is only
  ~3 OLCI pixels wide (~300 m px over a 1 km cell), so every pixel is an edge — the
  warped value set *overlaps* GEE's value set (132/147/154/165/187/196…), confirming the
  values are right but land on shifted cell pixels.

**Decision (user-approved):** ship the georeferencing-correct swath-warp adapter with a
**loose tolerance** (median |Δ| ≤ 60 radiance units, correlation floor 0.4); S3 has
**identity normalization** downstream with an explicit out-of-scope norm TODO. The
SNAP terrain-orthorectification of OLCI was *recorded* as a known follow-up (pairing it
with the S1 SNAP chain) — **but see §10.1: that follow-up has since been investigated and
rejected.** **S3 shipped (TASK-011), parity-loose by design.**

**Test (`tests/test_local_sources/test_s3_adapter.py`):** band order, `(2,H,W)` shape,
radiance-domain guard, missing-day → `-9999`, and a loose georeferencing-alignment
assertion (median |Δ| ≤ 60 + corr ≥ 0.4) — not bit-exactness.

### 10.1 SNAP ortho follow-up — investigated and REJECTED (2026-06-09)

Re-opened the §10 "SNAP terrain-orthorectification of OLCI is the closer for the
residual" follow-up after TASK-014 proved SNAP closes S1's parity wall. **It does not
transfer to S3.** The spike ran SNAP's OLCI ortho path and it went the *wrong* direction.

**Spike (`src/data/local_sources/parity/s3.py` +
`scripts/developer_scripts/bow_valley_inference_local/spikes/s3_olci_ortho_graph.xml`,
kept as evidence):** `Read → Subset(Oa17,Oa21 + AOI) → Reproject(orthorectify=true,
elevationModelName="SRTM 1Sec HGT", EPSG:32611, 300 m, Nearest) → Write`. SNAP's
`Reproject(orthorectify=true)` is the correct OLCI ortho operator — it uses the
product's tie-point geocoding + the DEM to terrain-correct the optical swath. (The SAR
`Terrain-Correction` / `Ellipsoid-Correction-GG` ops are radar-geometry only and reject
an OLCI product — verified.) Diffed against the same reference patch / day / cell as §10,
side-by-side with the production `griddata` swath-warp (identical 10403 co-valid pixels):

| band | swath-warp corr | SNAP-ortho corr | Δcorr | warp median \|Δ\| | ortho median \|Δ\| |
|------|-----------------|-----------------|-------|-------------------|--------------------|
| Oa17_radiance | 0.666 | 0.658 | **−0.008** | 35.39 | 36.08 |
| Oa21_radiance | 0.783 | 0.774 | **−0.009** | 22.77 | 24.01 |

**Conclusion — the §10 "un-orthorectified geolocation" hypothesis is wrong (or not the
dominant term).** If terrain distortion were the residual, SNAP's DEM ortho would have
closed it; instead corr was flat-to-slightly-worse on both bands. The real limit is
**sampling geometry**: the patch is ~3 OLCI pixels wide (~300 m px over a ~1 km cell), so
corr ~0.67 is a handful of edge pixels — too coarse to separate "good" from "perfect."
Both methods land the right *values* (the value sets overlap, §10) on a near-degenerate
coarse sample; neither can beat the source resolution. The **`spatial_kind="med"` 5×5
downsample erases any sub-pixel geolocation difference entirely** before the model sees it.

**Decision (user-approved):** keep the `griddata` swath-warp — marginally better, no
SNAP/DEM dependency, reads the **clipped** archive directly. (SNAP cannot: its
netCDF-Java reader throws `IllegalStateException: DataObject doesnt start with OHDR` on
the clip-rewritten HDF5 dimension-scale refs — the same landmine the adapter sidesteps
via `h5py`. The ortho spike had to source the **raw** product, an extra cost the
production adapter would inherit for zero gain.) **The SNAP-ortho follow-up is closed,
not deferred** — do not re-attempt it; the next lever for S3 parity is the un-fixed
identity-normalization TODO, not geolocation.
