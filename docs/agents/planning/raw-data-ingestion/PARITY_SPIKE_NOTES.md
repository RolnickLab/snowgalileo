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

- [ ] S1 spike run, drift recorded
- [x] S2 spike run, drift recorded — **within tolerance (GO)**, §4
- [ ] Go/no-go verdict stated (§5)

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

### Toolchain decision (user-approved 2026-06-04): **install `sarsen`, run the real spike**

`sarsen` 0.9.5 (+ `xarray-sentinel`, pinned `setuptools<81` for `pkg_resources`)
in the `spikes` dependency group. `sarsen.terrain_correction(...,
correct_radiometry=…)` does steps **4 + 5** (calibration + range-Doppler
geocoding to the Copernicus GLO-30 DEM already in the archive), the
load-bearing part of the chain.

**Documented gap (honest scope).** `sarsen` does **not** perform GRD border-noise
removal (step 2) or thermal-noise removal (step 3). Those primarily affect scene
edges and low-backscatter (water/shadow) pixels, which the `< -30 dB` edge mask
already suppresses. The measured drift (§4) tells us whether the gap is material
over this AOI; if VV/VH drift sits inside tolerance with the mask applied, the
missing noise steps are not a blocker for TASK-014 (they can be added with
`xarray-sentinel`'s noise LUTs if needed). This gap is called out so the verdict
is not over-claimed.

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

_S1 (per band, vs reference, valid pixels, −30 dB masked):_

| band | median |Δ| | p95 |Δ| | within tol? |
|------|----------|---------|-------------|
| VV   | _TBD_    | _TBD_   | _TBD_ |
| VH   | _TBD_    | _TBD_   | _TBD_ |
| angle| _TBD_    | _TBD_   | _TBD_ |

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

_TBD — stated after §4 is filled. If S1 or S2 drift is unrecoverable within
tolerance, escalate to the user **before** TASK-006 rather than proceeding._
