# Ingestion Phase: Non-Destructive Spatial Clipping Plan

This plan outlines the design and implementation of a spatial clipping utility to crop all raw geospatial datasets in `data/bow_valley_selection_raw` to the Area of Interest (AOI) represented in `data/aoi.geojson`.

The transformation is strictly **non-destructive** (preserves native pixel values, projections, data formats, and coordinate reference systems) and writes clipped outputs to `data/clipped_bow_valley_selection_raw`.

---

## 1. Bounding Box & Coordinate Specifications

The target AOI boundary parsed from [aoi.geojson](file:///home/dev/projects/presto-v3/data/aoi.geojson) is:
* **CRS:** `EPSG:4326` (WGS 84 geographic)
* **Longitude Range ($\lambda$):** `[-116.561936219710887, -114.527659450240762]`
* **Latitude Range ($\phi$):** `[50.729806886838752, 52.306672311654424]`

---

## 2. Modality-Specific Clipping Strategies

To preserve data structures and prevent unnecessary transformations (such as resampling, reprojecting, or format changes), the clipping script will apply targeted strategies for each format type.

### 2.1 Standard GeoTIFFs (DEM & WorldCover)
* **Sources:** `dem`, `worldcover`
* **Format:** Local `.tif` files.
* **Strategy:**
  1. Open the file with `rasterio`.
  2. Project the WGS84 AOI polygon to the TIFF's CRS (`EPSG:4326` for both).
  3. Crop using `rasterio.mask.mask` with `crop=True`.
  4. Write the clipped array to the destination path using the source's original `profile` (updating `width`, `height`, and `transform`).

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
  1. Open the input `.tar` file.
  2. If the scene bounds do not overlap the WGS84 AOI, skip the file.
  3. Create a new `.tar` archive in the output folder.
  4. Iterate through archive members:
     - If the member is a `.TIF` or `.tif` file:
       - Extract to a temporary directory.
       - Reproject the WGS84 AOI polygon to the band's UTM Zone 12N CRS (`EPSG:32612`).
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
  3. If there is no overlap with the WGS84 AOI, skip the file.
  4. Create a new `.zip` archive in the output folder.
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
  2. If there is no overlap with the WGS84 AOI, skip the file.
  3. Create a new output `.zip` archive.
  4. For each file member inside the zip:
     - If it is a `.tiff` file in the `measurement/` directory:
       - Extract the file.
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
  2. If there is no overlap with the WGS84 AOI, skip the file.
  3. Create a new output `.zip` archive.
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
* **Strategy:**
  1. Open the HDF4/HDF5 file.
  2. Reproject the WGS84 AOI coordinates to Sinusoidal projection.
  3. Calculate pixel indices `(row, col)` using standard sinusoidal grid cell bounds and grid cell resolution ($926.625\text{ m}$ per pixel):
     - `col = (x - upper_left_x) / dx`
     - `row = (upper_left_y - y) / dy`
  4. Clamp the pixel bounds to `[0, 1200]`.
  5. Subset all 2D science datasets using these pixel bounds.
  6. Save the cropped datasets to HDF5/HDF4 format (using system `gdal_translate` or `h5py` for VIIRS) into the destination directory.

---

## 3. Implementation Workflow

1. **Typer CLI Script (`scripts/developer_scripts/clip_dataset.py`):**
   * Uses `typer` to provide a robust CLI with commands `clip-all` and `clip-source`.
   * Accepts `--aoi-path`, `--input-dir`, and `--output-dir` arguments.
   * Leverages verbose logging (`structlog` or `logging`) to output detailed step-by-step progress.
   * Includes strict validations and assertions (checking file existence, geometry types, and projection alignment).

2. **Validation Script (`scripts/developer_scripts/test_clip_dataset.py`):**
   * Tests the clipping pipeline on a single small sample (e.g. one ERA5 file, one DEM file) to verify output bounds and dimensions.
