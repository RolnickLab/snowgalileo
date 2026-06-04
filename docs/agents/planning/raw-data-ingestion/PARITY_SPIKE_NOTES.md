# S1 / S2 Parity Spike Notes (TASK-005)

> **Purpose** — de-risk the two highest value-domain-drift sources (S1 GRD, S2
> L1C) against the Phase-0 GEE reference patches *before* building the full
> adapter stack (TASK-012/013/014). This file records the spike design, the
> toolchain decision, the chosen tolerances, the measured per-band drift, and
> the **go/no-go verdict** (SPEC AC-3 / AC-14 / AC-15, spike form).
>
> The spike scripts (`scripts/spikes/`) are **throwaway** — they exist to
> measure drift, not to ship. The real adapters re-implement the recovered
> recipe.

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
(`scripts/spikes/s1_grd_snap_graph.xml`) chains **all six steps** — Apply-Orbit →
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

**Measured drift** (filled by the spike run — `scripts/spikes/*` emit `structlog`):

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
