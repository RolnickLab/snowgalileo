# Ingestion Phase: Non-Destructive Spatial Clipping Plan

This plan outlines the design and implementation of a spatial clipping utility to crop all raw geospatial datasets in `data/bow_valley_selection_raw` to the Area of Interest (AOI) represented in `data/aoi.geojson`.

The transformation is strictly **non-destructive** (preserves native pixel values, projections, data formats, and coordinate reference systems) and writes clipped outputs to `data/clipped_bow_valley_selection_raw`.

> **Data-flow contract (RESOLVED).** This clip stage and the direct-source
> adapters in `PLAN_BOW_VALLEY_DATA.md §4` form a two-stage pipeline. The
> hand-off is now fixed: **the `LocalSource*` adapters read
> `data/clipped_bow_valley_selection_raw` (this stage's output), not the raw
> `data/bow_valley_selection_raw`.** This clip stage is therefore a **mandatory
> on-path prerequisite** (formalized as `PLAN_BOW_VALLEY_DATA.md §3 "Pipeline
> Stages & Data Flow"` and §7 Phase 0.5), not optional storage-shrink work, and
> `data/aoi.geojson` is the **single binding extent** for the whole pipeline. The
> §2.0 intersect gate below is consequently the *one* place footprint-vs-AOI
> filtering happens — adapters do not re-implement it. As of this writing
> `data/clipped_bow_valley_selection_raw` is empty (clip not yet run).
>
> **Partial AOI coverage per timestep (expected, not an error).** No
> observational source's per-day footprint is a superset of this AOI
> (~140 km × 175 km). Verified from the archive: Sentinel-2 covers all 4 AOI
> tiles on only 13 of its acquisition dates (3 tiles on 10, 2 on 12, 1 on 2);
> Sentinel-1 has just **16 acquisition dates** total over 2025-03→05, each a
> swath strip; MODIS/VIIRS are a single sinusoidal tile that does not span the
> full AOI. Consequences for this clip stage:
> 1. A clipped scene/granule will frequently cover only **part** of the AOI, or
>    none of it. The §2.0 intersect gate handles this: zero/degenerate overlap is
>    skipped (write nothing), but a *partial* overlap with real pixels must be
>    **kept and clipped to the intersection**, never skipped.
> 2. Clipping does not fabricate coverage. Days/areas with no acquisition stay
>    absent; the downstream adapters emit `-9999` placeholders there
>    (`PLAN_BOW_VALLEY_DATA.md` → "Per-Timestep, Per-Source Partial AOI
>    Coverage"). Do not pad clipped outputs to the full AOI.
> 3. Emit a per-source, per-date coverage manifest (which tiles/scenes
>    intersected the AOI, and their AOI-overlap fraction) so Phase 0 can profile
>    real input completeness over the 8-day windows.
>
> **Zone-mixing note (Landsat).** §2.3 clips Landsat in its native
> **EPSG:32612 (UTM 12N)**, while the cells, Sentinel-2, and the daily mosaic
> are **EPSG:32611 (UTM 11N)**. Clipping per-native-CRS is correct and
> non-destructive, but the downstream Landsat adapter must then reproject
> 32612 → the per-cell 4326 grid. This cross-zone reprojection is an explicit
> integration point for `PLAN_BOW_VALLEY_DATA.md §4`; verify it in the Landsat
> adapter parity test.

---

## 1. Bounding Box & Coordinate Specifications

The target AOI boundary parsed from [aoi.geojson](file:///home/dev/projects/presto-v3/data/aoi.geojson) is:
* **CRS:** `EPSG:4326` (WGS 84 geographic)
* **Longitude Range ($\lambda$):** `[-116.561936219710887, -114.527659450240762]`
* **Latitude Range ($\phi$):** `[50.729806886838752, 52.306672311654424]`

> **AOI authority & scope (CRS is law).** `data/aoi.geojson` is the
> **authoritative clip and inference boundary** for the whole pipeline. It is
> **not** the same extent as the cell-sampling bbox in
> `PLAN_BOW_VALLEY_DATA.md §3`. Reprojected to EPSG:4326, the 500 sampled cells
> span `lon=[-116.7408, -114.0104]`, `lat=[50.5121, 52.0046]` — **wider and
> further south than this AOI**. As a result **156 of 500 cells (31%) have their
> centre outside this AOI** and are **intentionally dropped**: the clipped
> archive will contain no data for them, and the grid generator filters them out
> (see `PLAN_BOW_VALLEY_DATA.md §3` grid-generator contract → 344 centre-in /
> 338 fully-in cells). This is by design. Do not widen the AOI to recover those
> cells without re-running the clip.
>
> **Static-layer coverage caveat.** This AOI reaches `lat_max = 52.31`. The
> DEM and WorldCover archives must mosaic to cover the full AOI to that
> latitude. The single-tile bounds quoted in `DATA_ANALYSIS.md` (DEM example
> tile north edge `51.0001`, WorldCover catalog north edge `51.0`) are
> *per-tile examples*, not the archive mosaic extent. Phase 0 audit MUST assert
> the DEM (9 `*_DEM.tif` elevation tiles) and WorldCover (4 `*_Map.tif` tiles)
> mosaics cover `[lon_min, lat_min, lon_max, lat_max]` before relying on them.
> (Verified in Phase 0: DEM mosaic `lat[50,53] lon[-117,-114]`, WorldCover
> `lat[48,54] lon[-120,-114]` — both contain the AOI to lat 52.31.)

---

## 2. Modality-Specific Clipping Strategies

To preserve data structures and prevent unnecessary transformations (such as resampling, reprojecting, or format changes), the clipping script will apply targeted strategies for each format type.

### 2.0 Mandatory Intersect Gate (applies to every source, before any clip)

**Problem.** Scene-collected sources (Landsat WRS path/rows, Sentinel-1/-2/-3
swaths and tiles) include products whose footprints lie partly — or, in a wider
archive pull, wholly — **outside the AOI**. Running `rasterio.mask.mask` (or any
slice) on a non-intersecting or barely-intersecting product produces an **empty
or all-`-9999` output**: a file that costs IO and storage, carries no signal,
and becomes pure noise / a degenerate placeholder downstream. Clipping must
**never manufacture such files.**

**Verified relevance (`data/bow_valley_selection_raw`).** The Landsat archive
spans **8 distinct WRS path/rows** (paths 042–044, rows 023–025). In this
*curated* archive every scene bbox-overlaps the AOI, but the **fraction of each
scene actually inside the AOI ranges 8%–42%** (parsed from `MTL.json` corner
coords). A wider or less-curated pull will contain wholly-outside scenes. The
gate must handle both: zero-overlap (skip) and tiny-but-nonzero overlap (keep
only if it yields real pixels).

**Gate — two-stage, run before the per-modality strategy in §2.1–§2.7:**

1. **Footprint vs AOI bbox test (cheap, metadata-only).** Read the product
   footprint from metadata **without decoding pixels** — Landsat `MTL.json`
   corner lon/lat; Sentinel `manifest.safe` / `xfdumanifest.xml` GML footprint;
   MODIS/VIIRS sinusoidal tile bounds; NetCDF coordinate ranges. Reproject the
   AOI to the product's CRS (or the footprint to 4326) and test polygon
   intersection — **not** just bbox-corner overlap, since swath footprints are
   non-rectangular. **No intersection → skip the entire product, log
   `SKIP_NO_OVERLAP`, write nothing.**

2. **Minimum-useful-overlap test (after intersection geometry is known).**
   Compute the intersection of the product footprint and the AOI polygon. Skip
   the product if **either**:
   - the intersection area is below `MIN_AOI_OVERLAP_AREA_KM2` (config, default
     e.g. `1 km²` — i.e. smaller than one grid cell, so it can populate no full
     cell), **or**
   - after the clip, the result contains **zero valid (non-nodata) pixels**
     (degenerate sliver that fell entirely on the product's own border nodata).

   Skipped → log `SKIP_DEGENERATE_OVERLAP` with the measured overlap, write
   nothing.

**Contract / config:**
- Thresholds are Pydantic settings (`MIN_AOI_OVERLAP_AREA_KM2`,
  `require_valid_pixels: bool = True`), not magic numbers.
- The gate is **fail-safe toward skipping**: a product that passes is clipped
  normally; a product that fails produces **no output file at all** (not an
  empty one). This keeps `data/clipped_bow_valley_selection_raw` free of
  zero-signal artifacts.
- Emit a per-source **clip manifest** row for every input product:
  `{product_id, footprint_bbox, intersects: bool, aoi_overlap_km2,
  valid_pixel_count, action: CLIP|SKIP_NO_OVERLAP|SKIP_DEGENERATE_OVERLAP}`.
  This manifest is the audit artifact Phase 0 consumes and the proof the clip
  stage created no noise.
- A clipped output that *passes* the gate but is still **partial** (the common
  case, 8–42% for Landsat here) is **kept and clipped to the intersection** —
  partial real data is valid; only *empty/degenerate* outputs are suppressed.
  Do not confuse "partial coverage" (keep) with "no useful overlap" (skip).

### 2.1 Standard GeoTIFFs (DEM & WorldCover)
* **Sources:** `dem`, `worldcover`
* **Format:** Local `.tif` files.
* **Note:** These are **multi-tile** archives (DEM: 9 `*_DEM.tif` elevation tiles, in a nested SAFE layout with ~196 files total incl. KML/XML/PDF + auxiliary FLM/EDM/HEM/WBM rasters; WorldCover: 4 `*_Map.tif` tiles, plus 4 `*_InputQuality.tif` companions = 8 tifs). Clip only the `*_DEM.tif` / `*_Map.tif` rasters. Individual tiles can lie fully outside the AOI — the §2.0 gate must run per tile so non-intersecting tiles are skipped, not clipped to empty.
* **Strategy:**
  1. Open the file with `rasterio`; read its bounds.
  2. **Apply the §2.0 intersect gate** per tile. On `SKIP_*`, write nothing.
  3. Project the WGS84 AOI polygon to the TIFF's CRS (`EPSG:4326` for both).
  4. Crop using `rasterio.mask.mask` with `crop=True`.
  5. Write the clipped array to the destination path using the source's original `profile` (updating `width`, `height`, and `transform`).

### 2.2 Climate NetCDF (ERA5-Land)
* **Source:** `era5`
* **Format:** NetCDF-4 (`.nc`) files.
* **Strategy:**
  1. Open the NetCDF file using `xarray` with `h5netcdf` engine.
  2. Slice along spatial dimensions:
     - `latitude` slice: `slice(lat_max, lat_min)` (since latitude is in descending order).
     - `longitude` slice: `slice(lon_min, lon_max)`.
  3. Save the clipped Dataset to the destination directory using `to_netcdf(..., engine="h5netcdf")`.

### 2.3 Landsat Tarballs (Landsat 8 & 9)
* **Sources:** `landsat8`, `landsat9`
* **Format:** `.tar` archives containing band GeoTIFFs (`_B*.TIF`) and text metadata.
* **Strategy:**
  1. Open the input `.tar` file; read the `MTL.json` corner lon/lat for the footprint.
  2. **Apply the §2.0 intersect gate** (polygon intersection + min-overlap). On `SKIP_*`, log and write nothing — do **not** create an output tarball. Note: this archive's Landsat scenes span 8 WRS path/rows with 8–42% AOI overlap, so most pass but are partial; partial passes are clipped to the intersection.
  3. Create a new `.tar` archive in the output folder **only after the gate passes**.
  4. Iterate through archive members:
     - If the member is a `.TIF` or `.tif` file:
       - Extract to a temporary directory.
       - **Read the band's native CRS from its header** and reproject the WGS84
         AOI polygon to that CRS. For this archive every Landsat scene is
         natively **`EPSG:32612` (UTM 12N)** — verified by `gdalinfo` on
         `LC09_..._B4.TIF` (`ID["EPSG",32612]`) and recorded in
         `DATA_ANALYSIS.md:545-546`. USGS delivers each WRS-2 path/row in its
         **assigned** UTM zone regardless of AOI longitude, so 32612 is correct
         here even though the AOI's longitude band (−116.5°…−114.5°) would suggest
         UTM 11. **Querying the band CRS dynamically (rather than asserting 32612)
         is the defensive choice** — it keeps the clip correct if a future, less
         curated pull mixes zones. Do not hardcode a single UTM zone for the AOI
         reprojection.
       - Crop using `rasterio.mask.mask` with `crop=True`.
       - Write the cropped TIFF, and add it to the output tarball.
     - Otherwise (MTL text, angles, XML):
       - Extract and add it to the output tarball unchanged.

### 2.4 Sentinel-2 Granules (Sentinel-2)
* **Source:** `sentinel2`
* **Format:** `.zip` archives containing the SAFE product folder with JPEG 2000 (`.jp2`) band images.
* **Strategy:**
  1. Open the `.zip` archive.
  2. Parse the spatial footprint from the embedded `manifest.safe` or `MTD_MSIL1C.xml` file.
  3. **Apply the §2.0 intersect gate** (polygon intersection + min-overlap). On `SKIP_*`, log and write nothing — do **not** create an output zip.
  4. Create a new `.zip` archive in the output folder **only after the gate passes**.
  5. For each file member inside the zip:
     - If the member is a JP2 band (`.jp2`):
       - Extract to a temporary directory.
       - Reproject the WGS84 AOI to S2's UTM Zone 11N CRS (`EPSG:32611`).
       - Open with `rasterio`, crop using `rasterio.mask.mask` with `crop=True`.
       - Write the cropped image back as JP2 using `driver="JP2OpenJPEG"` (OpenJPEG rwv driver) or TIFF if needed, and write to the output `.zip` archive under the original relative path.
     - Otherwise (XML, manifest):
       - Copy directly to the output zip file unchanged.

### 2.5 Sentinel-1 GCP-Based Swaths (Sentinel-1)
* **Source:** `sentinel1`
* **Format:** `.zip` archives containing the SAFE product with `.tiff` measurements in range geometry (`CRS: None` with GCPs).
* **Strategy:**
  1. Open the Sentinel-1 `.zip` file. Parse the geographic coordinates from `manifest.safe` or `preview/map-overlay.kml`.
  2. **Apply the §2.0 intersect gate** (polygon intersection + min-overlap) against the GML footprint. On `SKIP_*`, log and write nothing — do **not** create an output zip. (S1 swaths are the most likely to miss the AOI; the gate matters most here.)
  3. Create a new output `.zip` archive **only after the gate passes**.
  > **Range-geometry verified — GCP slicing is required, not over-engineering.**
  > A prior external review claimed S1 GRD-on-land ships orthorectified UTM
  > GeoTIFFs and that this GCP path is unnecessary. **Refuted empirically:**
  > `gdalinfo` on three archive measurement TIFFs shows only a `GCP Projection =
  > GEOGCRS["WGS 84"]` with 210 GCPs and **no `PROJCRS`**, and a raw pixel grid
  > (`Upper Left (0.0, 0.0) → Lower Right (26079, 16708)`). These `S1C` GRD
  > products are in **range geometry**, georeferenced by GCPs only — there is no
  > affine transform for `rasterio.mask.mask` to use, so the GCP-based slice below
  > is necessary. **Defensive fallback:** open the TIFF and check for a resolvable
  > CRS+transform first; only if absent (the case here) fall back to the GCP
  > slicing. This future-proofs a mixed archive without breaking the current one.
  >
  4. For each file member inside the zip:
     - If it is a `.tiff` file in the `measurement/` directory:
       - Extract the file.
       - **If the TIFF resolves a real CRS + affine transform, clip it like a
         standard GeoTIFF (§2.1).** Otherwise (the verified case for this
         archive — range geometry, GCPs only):
       - Extract all Ground Control Points (GCPs) from the TIFF header.
       - Find the min/max `row` and `col` of all GCPs whose geographical `x` (lon) and `y` (lat) overlap the WGS84 AOI (expanded with a 200-pixel buffer).
       - Slice the pixel grid using this bounding box: `array[:, row_min:row_max, col_min:col_max]`.
       - Shift the GCP pixel coordinates: `col_new = col - col_min` and `row_new = row - row_min`.
       - Save the cropped array with the shifted GCPs to the output zip under the original relative path.
     - Otherwise:
       - Copy to the output zip file unchanged.

### 2.6 Sentinel-3 OLCI Swaths (Sentinel-3)
* **Source:** `sentinel3`
* **Format:** `.zip` archives containing SAFE product with NetCDF (`.nc`) band radiance files georeferenced by separate tie-point grids.
* **Strategy:**
  1. Open the Sentinel-3 `.zip` file. Parse geographic coordinates from `xfdumanifest.xml`.
  2. **Apply the §2.0 intersect gate** (polygon intersection + min-overlap) against the manifest footprint. On `SKIP_*`, log and write nothing — do **not** create an output zip.
  3. Create a new output `.zip` archive **only after the gate passes**.
  4. Extract `geo_coordinates.nc` first. Read 2D `latitude` and `longitude` grids.
  5. Find the bounding box `[row_min, col_min, row_max, col_max]` of all indices where:
     - `lon_min <= longitude[row, col] <= lon_max` and `lat_min <= latitude[row, col] <= lat_max` (expanded with a 10-pixel buffer).
  6. For each `.nc` file in the zip:
     - Extract, open with `h5py`.
     - Slice all 2D datasets along the `rows` and `columns` dimensions to `[row_min:row_max, col_min:col_max]`.
     - Write the cropped `.nc` file to the output zip under the original relative path.
     - Ensure the attributes and structure are copied identically.

### 2.7 MODIS & VIIRS Sinusoidal Tiles (MODIS & VIIRS)
* **Sources:** `modis`, `viirs`
* **Format:** HDF4 (`.hdf`) and HDF5 (`.h5`) files on standard Sinusoidal Tile `h10v03`.
* **CRITICAL — multiple grid resolutions per file.** MOD09GA / VNP09GA each
  contain **two co-registered sinusoidal grids**, not one (verified via
  `gdalinfo` on `MOD09GA.A2025060.h10v03`):
  * **1 km grid** (`MODIS_Grid_1km_2D`): **1200 × 1200** px, ~926.625 m/px.
    Holds `state_1km` (cloud flag), geometry/angle bands.
  * **500 m grid** (`MODIS_Grid_500m_2D`): **2400 × 2400** px, ~463.31 m/px.
    Holds the science bands we actually need: `sur_refl_b01` … `sur_refl_b07`.

  A single resolution constant (`926.625 m`) and a single `[0, 1200]` clamp —
  as in the prior draft — **mis-clip the 500 m science bands**: the same
  geographic AOI maps to a `2400`-wide index range on that grid, so clamping to
  `1200` discards the eastern/southern half of every reflectance band. This was
  a correctness bug; the strategy below computes indices **per grid**.

* **Strategy:**
  1. Open the HDF4/HDF5 file and enumerate subdatasets; **group them by their
     native grid** (1 km vs 500 m). Read each grid's pixel dimensions and
     upper-left sinusoidal coordinate from the subdataset geotransform — do
     **not** hardcode.
  2. Reproject the WGS84 AOI corners to the MODIS Sinusoidal projection
     (`+proj=sinu +R=6371007.181`). **Apply the §2.0 intersect gate** against the
     tile's sinusoidal bounds: tile `h10v03` does **not** span the full AOI, so a
     file whose tile footprint misses the AOI is skipped (write nothing). After
     index clamping, if the clamped window is empty on both grids → `SKIP_*`.
  3. For **each grid independently**, compute pixel indices from *that grid's*
     resolution (`dx`/`dy`) and origin:
     - `col = (x - upper_left_x) / dx`
     - `row = (upper_left_y - y) / dy`
     - The 500 m grid uses `dx = dy ≈ 463.31 m` and dimensions `2400`; the 1 km
       grid uses `≈ 926.625 m` and dimensions `1200`.
  4. Clamp each grid's bounds to **`[0, grid_dim]`** (i.e. `[0, 2400]` for 500 m,
     `[0, 1200]` for 1 km) — never a single hardcoded `1200`.
  5. Subset each 2D science dataset using the pixel bounds of **its own** grid.
  6. Save the cropped datasets to HDF5/HDF4 format (using system
     `gdal_translate` per-subdataset, or `h5py` for VIIRS) into the destination
     directory, preserving each subdataset's grid association and geotransform.

  > **Validation gate:** assert that, for a known AOI corner, the 500 m index is
  > ~2× the 1 km index for the same lon/lat. A test that crops a single MODIS
  > file and checks both grids' output extents must pass before bulk runs.

---

## 3. Implementation Workflow

1. **Typer CLI Script (`scripts/developer_scripts/clip_dataset.py`):**
   * Uses `typer` to provide a robust CLI with commands `clip-all` and `clip-source`.
   * Accepts `--aoi-path`, `--input-dir`, and `--output-dir` arguments.
   * Leverages verbose logging (`structlog` or `logging`) to output detailed step-by-step progress.
   * Includes strict validations and assertions (checking file existence, geometry types, and projection alignment).

2. **Validation Script (`scripts/developer_scripts/test_clip_dataset.py`):**
   * Tests the clipping pipeline on a single small sample (e.g. one ERA5 file, one DEM file) to verify output bounds and dimensions.
   * **Intersect-gate tests (§2.0) — mandatory:**
     - A synthetic product footprint **fully outside** the AOI yields `SKIP_NO_OVERLAP` and **no output file** is written.
     - A footprint with a **sub-`MIN_AOI_OVERLAP_AREA_KM2`** intersection yields `SKIP_DEGENERATE_OVERLAP` and no output file.
     - A **partially-overlapping** real product (e.g. a Landsat scene with ~8% AOI overlap) is **kept**, clipped to the intersection, and its output contains > 0 valid pixels.
     - The clip manifest records one row per input with the correct `action` and measured `aoi_overlap_km2` / `valid_pixel_count`.
   * **Post-run audit:** after any bulk clip, assert `data/clipped_bow_valley_selection_raw` contains **zero** all-nodata / zero-valid-pixel outputs — the gate's purpose is to guarantee this.
