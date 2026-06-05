# Technical Review & Geospatial Audit: Bow Valley Ingestion & Inference Plan

This document presents a hyper-critical, technically precise quality assurance and architectural review of the plans, specifications, and task decompositions in `docs/agents/planning/raw-data-ingestion/`. 

This audit focuses on ensuring **zero-compromise correctness**, **geospatial precision**, and **scalable software architecture** in accordance with `docs/agents/KNOWLEDGE.md` and repository standards.

---

## 🚨 Critical Flaws (Blockers)

### 1. DEM Slope and Aspect Scale Distortion (Geospatial Error)
* **Status in Spec:** SPEC FR-15 states: *"The DEM adapter reprojects elevation to the 10 m cell grid before computing slope and aspect."*
* **The Flaw:** Reprojecting elevation to the `EPSG:4326` geographic grid (scale ≈ $0.0000898315^\circ$) and then calculating slope and aspect directly on that grid results in **massive numeric corruption**. Slope/aspect algorithms (e.g. Horn's or Zevenbergen-Thorne) calculate spatial derivatives ($\frac{dz}{dx}$, $\frac{dz}{dy}$). They assume that the horizontal coordinates ($x, y$) are in the **same units** as the vertical coordinate ($z$, meters).
  * In `EPSG:4326`, $x$ and $y$ are in decimal degrees, while $z$ is in meters. 
  * Because $1^\circ \approx 111,000\text{ m}$, a raw horizontal step of $0.000089^\circ$ is mathematically interpreted by standard derivative tools as $0.000089\text{ m}$ instead of $10\text{ m}$.
  * This scales calculated gradients by a factor of $\approx 111,000$, forcing all calculated slopes to approach $90^\circ$ (vertical cliffs) and corrupting aspect values completely.
* **Mitigation:** The DEM adapter **must calculate terrain metrics on a projected metric grid first**.
  1. Crop and reproject the Copernicus DEM to the target area in `EPSG:32611` (UTM Zone 11N) at a true 10 m pixel scale.
  2. Compute the slope and aspect on this UTM metric grid.
  3. Reproject the computed `slope` and `aspect` layers from `EPSG:32611` to `EPSG:4326` (scale=10) to match the cell grid.

### 2. Hardcoded UTM Zone 12N (`EPSG:32612`) for Landsat Clipping
* **Status in Spec:** `CLIPPING_PLAN.md §2.3` states: *"Reproject the WGS84 AOI polygon to S2's UTM Zone 11N CRS (EPSG:32611) ... Reproject the WGS84 AOI polygon to the band's UTM Zone 12N CRS (EPSG:32612)."*
* **The Flaw:** The Bow Valley AOI is situated between longitudes $-116.56^\circ$ and $-114.53^\circ$. UTM Zone 11 covers $-120^\circ$ to $-114^\circ$. Thus, the entire AOI is in **UTM Zone 11N (`EPSG:32611`)**. 
  * While some Landsat scenes in the wider archive might natively overlap the UTM Zone 12 boundary, hardcoding `EPSG:32612` for all Landsat scenes is a critical bug. 
  * If a Landsat scene is natively in UTM Zone 11N, reprojecting the AOI to `EPSG:32612` and attempting to crop the scene with a Zone 12N geometry will fail to align or crash `rasterio.mask.mask`.
* **Mitigation:** The clipping stage **must dynamically query the native CRS** of each Landsat TIFF band and reproject the WGS84 AOI to **that specific native CRS** before cropping. Never hardcode UTM projections for scene-by-scene operations.

---

## ⚠️ Major Risks (Performance & Numeric Parity)

### 3. Inode Exhaustion and Directory Performance in `.npz` Cache (Software Dev)
* **Status in Spec:** `PLAN_BOW_VALLEY_DATA.md §4` implements a per-modality per-(cell, day) `.npz` numpy cache in `data/bow_valley_processing/cube_cache/`.
* **The Flaw:** For $500$ cells, over $96$ days, with $10$ modalities, this results in up to **$480,000$ files** in the cache directory.
  * Storing half a million small files in a single flat directory will severely degrade filesystem performance on ext4/xfs, cause major latency in file lookups ($O(N)$ directory indexing degradation), and potentially exhaust the system's inode limits.
* **Mitigation:** Implement a sharded directory structure for the cache.
  * Instead of: `cube_cache/{cell_id}_{day}_{modality}.npz`
  * Use a two-tier nested layout: `cube_cache/{cell_id_prefix}/{cell_id}/{day}_{modality}.npz` where `cell_id_prefix` is the first 3 digits or a hash bucket. This keeps files per directory under $1,000$ and maintains $O(1)$ lookup speed.

### 4. MODIS Bilinear Resampling "Edge Bleeding" (Numeric Parity)
* **Status in Spec:** `SPEC FR-7` and `FR-10` require all continuous bands to use bilinear resampling and MODIS to preserve the `-28672` fill value.
* **The Flaw:** Bilinear resampling interpolates values across a 2x2 grid. If a pixel bordering the edge of a MODIS acquisition contains the fill value `-28672`, a bilinear resampler will interpolate between valid reflectance values (e.g. $1500$) and the large negative fill value $-28672$.
  * This results in **"edge bleeding"** where valid border pixels are corrupted to negative values (e.g., $-5000$), which bypasses the standard invalid-threshold checks and pollutes model inputs.
* **Mitigation:** 
  1. Read the raw MODIS band.
  2. Create a binary mask of valid pixels (where value $\ne -28672$ and value $\ne -9999$).
  3. Set all fill pixels to `NaN` (or mask them in a masked array).
  4. Perform the bilinear reprojection with a nodata-aware resampler, or apply standard bilinear interpolation on the masked array and re-fill missing values with `-28672`/`-9999` in the output grid.

### 5. Sentinel-1 GRD GCP-Based Swath vs. Projected GeoTIFFs
* **Status in Spec:** `CLIPPING_PLAN.md §2.5` details an elaborate custom GCP-based pixel grid slicing strategy for Sentinel-1: *"Find the min/max row and col of all GCPs ... Slice the pixel grid using this bounding box: array[:, row_min:row_max, col_min:col_max]"*.
* **The Flaw:** Sentinel-1 **GRD (Ground Range Detected)** products on land are already orthorectified and projected. Unlike SLC (Single Look Complex) products, GRD products in the Copernicus SAFE structure **already possess a standard map projection (usually UTM) and an affine transform** in their GeoTIFF headers.
  * Custom pixel-level slicing based on GCPs is unnecessary, highly error-prone, and risks misalignment because GCPs are coarsely spaced in SAFE metadata.
* **Mitigation:** Check if the S1 GRD GeoTIFFs already carry a valid CRS and transform. If they do (which is standard), treat them exactly like standard GeoTIFFs (DEM/WorldCover/S2) and use standard `rasterio.mask.mask` with projection alignment. Only fall back to manual GCP indexing if opening the file fails to resolve a spatial transform.

---

## 🔍 Minor Concerns & Spec Inconsistencies

### 6. Filename Format Discrepancy
* **Status in Spec:** `TASK-001` and `README.md` note that `export_from_csv_utm` (the GEE exporter) emits files named `PR_{date}_{cx}_{cy}.tif` (UTM coords), whereas `LocalSourceExporter` (TASK-004) is specified to emit `PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif` (geographic degrees).
* **The Flaw:** Both styles are parsed in `landsat_eval.py` via regex branches, but `LandsatEvalDataset` uses coordinate parsing to match inputs to labels.
  * If the local exporter uses decimal degrees and the GEE reference patches use UTM coordinates, the test harness will fail to align them for the full-stack parity tests.
* **Mitigation:** Ensure that `tests/test_local_sources/test_exporter_parity.py` translates coordinate representations explicitly before matching, or align both pipelines to use a unified filename layout.

### 7. S2 Processing Baseline Reading Location
* **Status in Spec:** SPEC FR-11 requires checking the Sentinel-2 processing baseline version and subtracting 1000 DN if $\ge 04.00$ (`N0511` in archive).
* **The Flaw:** The spec does not define *where* the baseline version is read. S2 SAFE archives contain this in `manifest.safe` under `<processingBaselineVersion>` and in `MTD_MSIL1C.xml` under `<PRODUCT_PROPERTIES><PROCESSING_BASELINE>`.
* **Mitigation:** In the S2 adapter, explicitly parse `MTD_MSIL1C.xml` or the SAFE folder metadata using an XML parser to extract the processing baseline string safely, and fall back to `04.00` if the tag is missing but baseline version `N0511` is part of the path.

---

## 📊 Summary of Quality and Severity

The planning documents (`PLAN`, `SPEC`, `FDD`) represent an exceptionally detailed, well-thought-out, and robust integration design. The strict division of Stage 0, 1, and 2 directories and the introduction of same-tile/date coalescing are superb engineering decisions. 

However, before implementation begins, the following actions must be taken to prevent immediate test failures and geospatial data corruption:

| Severity | Target File / Task | Issue | Action Required |
| :--- | :--- | :--- | :--- |
| **Critical** | `SPEC_BOW_VALLEY_DATA.md` / `TASK-007` | DEM Slope/Aspect geographic distortion | Calculate metrics in metric `EPSG:32611` before reprojecting. |
| **Critical** | `CLIPPING_PLAN.md` / `TASK-002` | Hardcoded Landsat Target CRS `EPSG:32612` | Query band CRS dynamically; reproject AOI to native band CRS. |
| **Major** | `PLAN_BOW_VALLEY_DATA.md` / `TASK-003` | Cache inode exhaustion (480k files) | Shard `.npz` cache directory structure. |
| **Major** | `SPEC_BOW_VALLEY_DATA.md` / `TASK-009` | MODIS bilinear fill-value bleed | Apply valid mask/NaNs to MODIS bands before bilinear interpolation. |
| **Minor** | `CLIPPING_PLAN.md` / `TASK-014` | Sentinel-1 GRD GCP Custom Slicing | Standardize on rasterio-based CRS crop for GRD. |

---

## ✅ Validation Verdict (Claude, 2026-06-01 — empirical re-check)

Each finding was re-validated against the **actual archive** (`gdalinfo`/`unzip`
on real `data/bow_valley_selection_raw` files) and the GEE reference code
(`src/data/earthengine/copernicus_dem.py`), not just the prose. Two of the
review's three "Critical/Major-blocker" geospatial claims (#2, #5) are
**empirically false for this archive** and their recommended fixes would have
*introduced* bugs. Evidence is recorded below so they are not re-raised.

| # | Review verdict | Validation | Evidence |
| :- | :- | :- | :- |
| 1 | DEM slope/aspect 4326 distortion | **Partially valid — but the proposed UTM fix is wrong** | `copernicus_dem.py:14-16` calls `ee.Terrain.slope/aspect` on the DEM's **native projection**; GEE's terrain ops are latitude-aware (true metric pixel spacing), then export resamples to the cell grid. The parity target therefore never computes terrain in EPSG:32611. The review's "compute in 32611, reproject" detour would **diverge from the GEE reference patches** AC-21 tests against. Real defect = loose FR-15 wording, not the algorithm. **Fix: tighten FR-15 to "latitude-correct metric pixel spacing matching `ee.Terrain`," not a UTM detour.** **⚠️ UPDATE 2026-06-04:** this verdict's original wording said the patches "never compute in a UTM frame / then export to 4326/scale=10" — that conflated the **terrain-computation frame** (native DEM, correct & unchanged) with the **export/cell grid CRS**. The cell grid is now confirmed **EPSG:32611** (UTM 11N), not 4326 (see KNOWLEDGE.md / PLAN §3). The verdict's *conclusion* is unchanged — compute terrain in the native lat-aware frame, do NOT compute in UTM — but the *reason* is "because `ee.Terrain` computes natively," not "because the export is 4326." The terrain RESAMPLE target is UTM; the terrain COMPUTATION frame is native. See TASK-007 §2. |
| 2 | **CRITICAL:** Landsat 32612 is a hardcode bug; AOI is UTM 11 | **FALSE (but "all 32612" half-claim is ALSO wrong — see 2026-06-05 correction)** | `gdalinfo` on `LC09_L1TP_042024_20250310_..._B4.TIF` → `ID["EPSG",32612]` (UTM 12N). USGS delivers WRS-2 path/row scenes in their assigned UTM zone regardless of AOI longitude. Clipping in native zone is correct + non-destructive; the adapter reprojects native→cell later (FR-5). The review's "hardcode 32612 for *all* scenes" fix is a bug **for the opposite reason it claimed**: the archive is mixed-zone, so the right rule is **query each band's CRS**. **⚠️ CORRECTION (2026-06-05, TASK-012b):** the original "**All** archive Landsat scenes (paths 042–044) are natively 32612" was inferred from one path-042024 sample and is **empirically FALSE**. USGS assigns zone by scene-center longitude, so the archive is **mixed UTM 11N/12N per scene**: paths **043/044 = EPSG:32611** (all rows verified), **042024 = 32612**, but **042025 = 32611** (adjacent rows straddle the 114°W boundary). The 7 TASK-012b scenes are all 32611. Adapter MUST read per-band CRS; never hardcode 32612. The clip stage already preserves native zone (raw 32611 → clipped 32611, no reprojection). See memory `landsat-mixed-utm-zone`. |
| 3 | **MAJOR:** `.npz` inode exhaustion | **VALID** | Flat `cube_cache/{cell}_{day}_{modality}.npz`: mode A ≈ 344 cells × 96 days × ~9 modalities ≈ 300k files in one dir → ext4 htree O(N) degradation. **Fix: shard `cube_cache/{cell_id}/{day}_{modality}.npz`** (per-cell subdir ≈ 864 files/dir; the review's extra hash-prefix tier is unnecessary). |
| 4 | **MAJOR:** MODIS bilinear fill-bleed | **VALID** | Bilinear across valid reflectance + `-28672` fill yields garbage negatives that bypass thresholds. Applies to **all** continuous bands near nodata edges, not MODIS only. **Fix: nodata-aware bilinear in the shared `base.py` resampler** (mask fill→NaN before warp, restore `-9999`/`-28672` after). |
| 5 | **MINOR:** S1 GRD already projected; GCP slicing over-engineered | **FALSE for this archive** | `gdalinfo` on 3 S1 measurement TIFFs → `GCP Projection = GEOGCRS["WGS 84"]`, 210 GCPs, **no `PROJCRS`**; `Upper Left (0.0,0.0) → Lower Right (26079,16708)` = raw pixel grid. These `S1C` GRD products are in **range geometry**, georeferenced by GCPs only. `CLIPPING_PLAN.md §2.5` GCP slicing is **correct + necessary**; "treat as standard GeoTIFF" would fail (no transform for `rasterio.mask.mask`). **Keep as-is; add CRS-first-then-GCP-fallback robustness note.** |
| 6 | **MINOR:** S2 baseline read location undefined | **VALID** | `unzip -p` → `MTD_MSIL1C.xml` contains `<PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>`. **Fix: FR-11 reads baseline from `MTD_MSIL1C.xml`, fall back to N0511-from-path** (Q8 already verified all 116 granules are N0511). |
| 7 | Filename UTM-vs-degree mismatch | **Non-issue** | Both GEE reference patches and local cubes are driven by the **same generated `cube_cells.csv`** (FR-19b); parity matches on shared CSV rows, not raw filename strings. One-line note added to the parity test. |

**Net:** valid catches = #3, #4 (the two `Major`s). The two confidently-stated
`Critical`/blocker geospatial claims (#2 Landsat, #5 S1) are **wrong** — the
reviewer reasoned from CRS theory without inspecting the files. #1 has correct
physics but the wrong fix relative to the GEE parity target. #6/#7 are minor.
**Do NOT apply #1's UTM detour, #2, or #5 as written — they would introduce
parity/clip failures.** Applied corrections: #1 (wording), #3, #4, #6, #7, plus
defensive notes for #2/#5.

---
*Caveman mode resume. Review validated against archive. Action plan corrected.*
