import os
import re
import json
import shutil
import tempfile
import tarfile
import zipfile
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import typer
import structlog
import rasterio
import rasterio.mask
from rasterio.windows import Window
import xarray as xr
import h5py
import pyproj
from shapely.geometry import shape, box, Polygon

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if os.environ.get("LOG_JSON") else structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

app = typer.Typer(help="Non-destructive Spatial Clipping Utility for Bow Valley Raw Datasets")

# Target AOI Coordinates in EPSG:4326
AOI_LON_MIN, AOI_LON_MAX = -116.561936219710887, -114.527659450240762
AOI_LAT_MIN, AOI_LAT_MAX = 50.729806886838752, 52.306672311654424

# Sinusoidal projection definitions for MODIS/VIIRS
sinu_crs = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
wgs84_crs = "epsg:4326"
wgs84_to_sinu = pyproj.Transformer.from_crs(wgs84_crs, sinu_crs, always_xy=True)

# Standard sinusoidal bounds for tile h10v03
SINU_X_MIN, SINU_Y_MAX = -8895604.157, 6671703.118
SINU_X_MAX, SINU_Y_MIN = -7783653.638, 5559752.598
SINU_WIDTH = SINU_X_MAX - SINU_X_MIN
SINU_HEIGHT = SINU_Y_MAX - SINU_Y_MIN

def get_aoi_geometry(aoi_path: Path) -> Dict[str, Any]:
    assert aoi_path.exists(), f"AOI file {aoi_path} does not exist"
    with open(aoi_path, "r") as f:
        geojson = json.load(f)
    features = geojson.get("features", [])
    assert len(features) > 0, "No features found in AOI GeoJSON"
    geom = features[0].get("geometry")
    assert geom is not None, "First feature lacks geometry"
    assert geom["type"] == "Polygon", "AOI must be a Polygon"
    return geom

def check_overlap(item_bbox: List[float], aoi_bbox: List[float]) -> bool:
    # item_bbox: [lon_min, lat_min, lon_max, lat_max]
    # aoi_bbox: [lon_min, lat_min, lon_max, lat_max]
    b1 = box(*item_bbox)
    b2 = box(*aoi_bbox)
    return b1.intersects(b2)

# --- Standard GeoTIFF (dem, worldcover) ---
def clip_geotiff(input_path: Path, output_path: Path, aoi_geom: Dict[str, Any]):
    logger.info("Clipping GeoTIFF", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        # Reproject AOI geometry to TIFF's CRS if different
        if src.crs != pyproj.CRS.from_epsg(4326):
            projector = pyproj.Transformer.from_crs("epsg:4326", src.crs, always_xy=True)
            coords = aoi_geom["coordinates"][0]
            projected_coords = [projector.transform(x, y) for x, y in coords]
            geom = {"type": "Polygon", "coordinates": [projected_coords]}
        else:
            geom = aoi_geom
            
        out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
        out_meta = src.meta.copy()
        
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        assert out_image.shape[1] > 0 and out_image.shape[2] > 0, "Clipped GeoTIFF has degenerate size"
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    logger.info("Successfully saved clipped GeoTIFF", path=str(output_path))

# --- Climate NetCDF (era5) ---
def clip_era5(input_path: Path, output_path: Path):
    logger.info("Clipping ERA5 NetCDF", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with xr.open_dataset(input_path, engine="h5netcdf") as ds:
        # Check coordinates and slice
        lat_coords = ds.coords["latitude"].values
        lon_coords = ds.coords["longitude"].values
        
        # Slicing along lat/lon. Note: latitude might be descending or ascending
        if lat_coords[0] > lat_coords[-1]:
            ds_clipped = ds.sel(latitude=slice(AOI_LAT_MAX, AOI_LAT_MIN))
        else:
            ds_clipped = ds.sel(latitude=slice(AOI_LAT_MIN, AOI_LAT_MAX))
            
        ds_clipped = ds_clipped.sel(longitude=slice(AOI_LON_MIN, AOI_LON_MAX))
        
        assert len(ds_clipped.coords["latitude"]) > 0 and len(ds_clipped.coords["longitude"]) > 0, "Clipped ERA5 dataset has empty dimensions"
        
        ds_clipped.to_netcdf(output_path, engine="h5netcdf")
    logger.info("Successfully saved clipped NetCDF", path=str(output_path))

# --- Landsat Tarballs (landsat8, landsat9) ---
def clip_landsat_tar(input_path: Path, output_path: Path, aoi_geom: Dict[str, Any]):
    logger.info("Clipping Landsat Tar archive", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(input_path, "r") as src_tar, tarfile.open(output_path, "w") as dest_tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            for member in src_tar.getmembers():
                if member.name.lower().endswith((".tif", ".tiff")):
                    logger.debug("Clipping internal Landsat band", band=member.name)
                    src_tar.extract(member, path=tmpdir)
                    band_path = Path(tmpdir) / member.name
                    clipped_band_path = Path(tmpdir) / f"clipped_{member.name}"
                    
                    # Open, clip and save
                    clip_geotiff(band_path, clipped_band_path, aoi_geom)
                    
                    # Add clipped band to output tar
                    dest_tar.add(clipped_band_path, arcname=member.name)
                else:
                    # Non-raster metadata or text files, extract and copy directly
                    logger.debug("Copying internal Landsat metadata", file=member.name)
                    src_tar.extract(member, path=tmpdir)
                    dest_tar.add(Path(tmpdir) / member.name, arcname=member.name)
    logger.info("Successfully saved clipped Landsat Tarball", path=str(output_path))

# --- Sentinel-2 Zip (sentinel2) ---
def clip_sentinel2_zip(input_path: Path, output_path: Path, aoi_geom: Dict[str, Any]):
    logger.info("Clipping Sentinel-2 Zip archive", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(input_path, "r") as src_zip, zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in src_zip.namelist():
                if name.lower().endswith(".jp2"):
                    logger.debug("Clipping internal Sentinel-2 JP2 band", band=name)
                    src_zip.extract(name, path=tmpdir)
                    band_path = Path(tmpdir) / name
                    clipped_band_path = Path(tmpdir) / f"clipped_{Path(name).name}"
                    
                    # Clip JP2 and write as JP2OpenJPEG format
                    clip_geotiff(band_path, clipped_band_path, aoi_geom)
                    
                    # Add to output zip under original relative path
                    dest_zip.write(clipped_band_path, arcname=name)
                else:
                    logger.debug("Copying internal Sentinel-2 metadata", file=name)
                    # Copy non-raster file directly
                    src_zip.extract(name, path=tmpdir)
                    dest_zip.write(Path(tmpdir) / name, arcname=name)
    logger.info("Successfully saved clipped Sentinel-2 Zip", path=str(output_path))

# --- Sentinel-1 Zip (sentinel1) ---
def clip_sentinel1_zip(input_path: Path, output_path: Path):
    logger.info("Clipping Sentinel-1 Zip archive (GCP-based Pixel Crop)", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(input_path, "r") as src_zip, zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in src_zip.namelist():
                if name.lower().endswith((".tiff", ".tif")) and "measurement/" in name:
                    logger.debug("Clipping internal Sentinel-1 SAR TIFF", band=name)
                    src_zip.extract(name, path=tmpdir)
                    tiff_path = Path(tmpdir) / name
                    clipped_tiff_path = Path(tmpdir) / f"clipped_{Path(name).name}"
                    
                    # Open and read GCPs
                    with rasterio.open(tiff_path) as src:
                        gcps = src.gcps[0]
                        gcp_crs = src.gcps[1]
                        
                        # Find matching GCP indices that overlap the WGS84 AOI
                        overlapping_indices = []
                        for idx, g in enumerate(gcps):
                            if AOI_LON_MIN <= g.x <= AOI_LON_MAX and AOI_LAT_MIN <= g.y <= AOI_LAT_MAX:
                                overlapping_indices.append(idx)
                                
                        if not overlapping_indices:
                            logger.warning("No S1 GCPs found inside the AOI. Swath may not overlap. Clipping to empty/default.", band=name)
                            # Clip to very small region at center
                            col_min, row_min, col_max, row_max = 0, 0, 100, 100
                        else:
                            # Extract bounds
                            overlapping_gcps = [gcps[i] for i in overlapping_indices]
                            col_min = int(min(g.col for g in overlapping_gcps))
                            row_min = int(min(g.row for g in overlapping_gcps))
                            col_max = int(max(g.col for g in overlapping_gcps))
                            row_max = int(max(g.row for g in overlapping_gcps))
                            
                            # Expand buffer (e.g. 200 pixels) to avoid boundary cuts
                            col_min = max(0, col_min - 200)
                            row_min = max(0, row_min - 200)
                            col_max = min(src.width, col_max + 200)
                            row_max = min(src.height, row_max + 200)
                            
                        # Crop pixel array
                        logger.debug("S1 Pixel crop window coordinates", col_min=col_min, col_max=col_max, row_min=row_min, row_max=row_max)
                        crop_window = Window(col_min, row_min, col_max - col_min, row_max - row_min)
                        data = src.read(window=crop_window)
                        
                        # Shift GCPs
                        shifted_gcps = []
                        for g in gcps:
                            # Keep only GCPs within the clipped bounds
                            if col_min <= g.col <= col_max and row_min <= g.row <= row_max:
                                shifted_g = rasterio.control.GroundControlPoint(
                                    row=g.row - row_min,
                                    col=g.col - col_min,
                                    x=g.x,
                                    y=g.y,
                                    z=g.z,
                                    id=g.id
                                )
                                shifted_gcps.append(shifted_g)
                                
                        # Write the cropped TIFF
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "height": data.shape[1],
                            "width": data.shape[2],
                            "transform": rasterio.transform.Affine.identity() # Range geometry has no affine transform
                        })
                        
                        with rasterio.open(clipped_tiff_path, "w", **out_meta) as dest:
                            dest.write(data)
                            dest.gcps = (shifted_gcps, gcp_crs)
                            
                    # Write to output zip
                    dest_zip.write(clipped_tiff_path, arcname=name)
                else:
                    logger.debug("Copying internal Sentinel-1 metadata", file=name)
                    src_zip.extract(name, path=tmpdir)
                    dest_zip.write(Path(tmpdir) / name, arcname=name)
    logger.info("Successfully saved clipped Sentinel-1 Zip", path=str(output_path))

# --- Sentinel-3 Zip (sentinel3) ---
def clip_sentinel3_zip(input_path: Path, output_path: Path):
    logger.info("Clipping Sentinel-3 Zip archive (Coordinate-based Swath Crop)", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(input_path, "r") as src_zip, zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract geo_coordinates first to determine spatial indices bounds
            geo_nc_name = [x for x in src_zip.namelist() if "geo_coordinates.nc" in x][0]
            src_zip.extract(geo_nc_name, path=tmpdir)
            geo_nc_path = Path(tmpdir) / geo_nc_name
            
            with h5py.File(geo_nc_path, "r") as f:
                latitudes = f["latitude"][:]
                longitudes = f["longitude"][:]
                
            # Find row/col grid coordinates overlapping the WGS84 AOI
            overlap = (AOI_LON_MIN <= longitudes) & (longitudes <= AOI_LON_MAX) & (AOI_LAT_MIN <= latitudes) & (latitudes <= AOI_LAT_MAX)
            rows, cols = overlap.nonzero()
            
            if len(rows) == 0:
                logger.warning("No S3 coordinates overlap the AOI bounds. Clipping to tiny region.", file=input_path.name)
                row_min, row_max, col_min, col_max = 0, 100, 0, 100
            else:
                row_min, row_max = int(rows.min()), int(rows.max())
                col_min, col_max = int(cols.min()), int(cols.max())
                
                # Buffer (e.g. 10 pixels)
                row_min = max(0, row_min - 10)
                row_max = min(latitudes.shape[0], row_max + 10)
                col_min = max(0, col_min - 10)
                col_max = min(latitudes.shape[1], col_max + 10)
                
            logger.debug("S3 Grid crop coordinates", row_min=row_min, row_max=row_max, col_min=col_min, col_max=col_max)
            
            # Loop through all files in zip
            for name in src_zip.namelist():
                if name.lower().endswith(".nc"):
                    logger.debug("Clipping internal Sentinel-3 NetCDF band", file=name)
                    src_zip.extract(name, path=tmpdir)
                    nc_path = Path(tmpdir) / name
                    clipped_nc_path = Path(tmpdir) / f"clipped_{Path(name).name}"
                    
                    with h5py.File(nc_path, "r") as src_f, h5py.File(clipped_nc_path, "w") as dest_f:
                        # Copy global attributes
                        for attr_name, attr_val in src_f.attrs.items():
                            dest_f.attrs[attr_name] = attr_val
                            
                        # Traverse and crop datasets
                        def visitor(obj_name, obj):
                            if isinstance(obj, h5py.Dataset):
                                ds_shape = obj.shape
                                if len(ds_shape) == 2 and ds_shape[0] == latitudes.shape[0] and ds_shape[1] == latitudes.shape[1]:
                                    # Crop 2D bands matching rows x columns
                                    data_cropped = obj[row_min:row_max, col_min:col_max]
                                    dest_ds = dest_f.create_dataset(obj_name, data=data_cropped, dtype=obj.dtype, compression=obj.compression)
                                elif len(ds_shape) == 1 and ds_shape[0] == latitudes.shape[0]:
                                    # Crop 1D along row dimension
                                    data_cropped = obj[row_min:row_max]
                                    dest_ds = dest_f.create_dataset(obj_name, data=data_cropped, dtype=obj.dtype, compression=obj.compression)
                                elif len(ds_shape) == 1 and ds_shape[0] == latitudes.shape[1]:
                                    # Crop 1D along column dimension
                                    data_cropped = obj[col_min:col_max]
                                    dest_ds = dest_f.create_dataset(obj_name, data=data_cropped, dtype=obj.dtype, compression=obj.compression)
                                else:
                                    # Copy other static arrays/dimensions unchanged
                                    dest_ds = dest_f.create_dataset(obj_name, data=obj[()], dtype=obj.dtype)
                                    
                                for attr_name, attr_val in obj.attrs.items():
                                    dest_ds.attrs[attr_name] = attr_val
                                    
                        src_f.visititems(visitor)
                        
                    # Write to output zip
                    dest_zip.write(clipped_nc_path, arcname=name)
                else:
                    logger.debug("Copying internal Sentinel-3 metadata", file=name)
                    src_zip.extract(name, path=tmpdir)
                    dest_zip.write(Path(tmpdir) / name, arcname=name)
    logger.info("Successfully saved clipped Sentinel-3 Zip", path=str(output_path))

# --- VIIRS (viirs) & MODIS (modis) Sinusoidal HDF/H5 ---
def clip_sinusoidal_hdf5(input_path: Path, output_path: Path):
    logger.info("Clipping VIIRS/MODIS Sinusoidal H5/HDF5 file", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate pixel indices using sinusoidal coordinates transformer
    # AOI Lon/Lat corners reprojected to sinusoidal meters
    x_coords = []
    y_coords = []
    for lon, lat in [(AOI_LON_MIN, AOI_LAT_MIN), (AOI_LON_MAX, AOI_LAT_MIN), (AOI_LON_MAX, AOI_LAT_MAX), (AOI_LON_MIN, AOI_LAT_MAX)]:
        x, y = wgs84_to_sinu.transform(lon, lat)
        x_coords.append(x)
        y_coords.append(y)
        
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # h10v03 coordinates to pixel conversions
    # 1. For 1km grid (1200 x 1200)
    dx_1k = SINU_WIDTH / 1200
    dy_1k = SINU_HEIGHT / 1200
    col_min_1k = max(0, int((x_min - SINU_X_MIN) / dx_1k))
    col_max_1k = min(1200, int((x_max - SINU_X_MIN) / dx_1k) + 1)
    row_min_1k = max(0, int((SINU_Y_MAX - y_max) / dy_1k))
    row_max_1k = min(1200, int((SINU_Y_MAX - y_min) / dy_1k) + 1)
    
    # 2. For 500m grid (2400 x 2400)
    col_min_500 = col_min_1k * 2
    col_max_500 = col_max_1k * 2
    row_min_500 = row_min_1k * 2
    row_max_500 = row_max_1k * 2
    
    logger.debug("Sinusoidal pixel crop bounds", 
                 col_min_1k=col_min_1k, col_max_1k=col_max_1k, 
                 row_min_1k=row_min_1k, row_max_1k=row_max_1k)
                 
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dest:
        # Copy attributes
        for attr_name, attr_val in src.attrs.items():
            dest.attrs[attr_name] = attr_val
            
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                ds_shape = obj.shape
                # Check grid resolution and slice accordingly
                if len(ds_shape) == 2 and ds_shape[0] == 1200 and ds_shape[1] == 1200:
                    data = obj[row_min_1k:row_max_1k, col_min_1k:col_max_1k]
                    dest_ds = dest.create_dataset(name, data=data, dtype=obj.dtype)
                elif len(ds_shape) == 2 and ds_shape[0] == 2400 and ds_shape[1] == 2400:
                    data = obj[row_min_500:row_max_500, col_min_500:col_max_500]
                    dest_ds = dest.create_dataset(name, data=data, dtype=obj.dtype)
                else:
                    # Static/scalar datasets
                    dest_ds = dest.create_dataset(name, data=obj[()], dtype=obj.dtype)
                    
                for attr_name, attr_val in obj.attrs.items():
                    dest_ds.attrs[attr_name] = attr_val
                    
        src.visititems(visitor)
    logger.info("Successfully saved clipped Sinusoidal H5 file", path=str(output_path))

def clip_modis_hdf4(input_path: Path, output_path: Path):
    # Since writing HDF4 in Python is generally not supported, we convert the HDF4 tile
    # to a standardized, modern HDF5 file during the non-destructive clipping process!
    # This is fully compatible with our local source adapters and is highly standard.
    logger.info("Clipping MODIS HDF4 tile and converting to HDF5", input=str(input_path), output=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate pixel indices using sinusoidal coordinates transformer (1200 x 1200 and 2400 x 2400)
    x_coords = []
    y_coords = []
    for lon, lat in [(AOI_LON_MIN, AOI_LAT_MIN), (AOI_LON_MAX, AOI_LAT_MIN), (AOI_LON_MAX, AOI_LAT_MAX), (AOI_LON_MIN, AOI_LAT_MAX)]:
        x, y = wgs84_to_sinu.transform(lon, lat)
        x_coords.append(x)
        y_coords.append(y)
        
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    dx_1k = SINU_WIDTH / 1200
    dy_1k = SINU_HEIGHT / 1200
    col_min_1k = max(0, int((x_min - SINU_X_MIN) / dx_1k))
    col_max_1k = min(1200, int((x_max - SINU_X_MIN) / dx_1k) + 1)
    row_min_1k = max(0, int((SINU_Y_MAX - y_max) / dy_1k))
    row_max_1k = min(1200, int((SINU_Y_MAX - y_min) / dy_1k) + 1)
    
    col_min_2k = col_min_1k * 2
    col_max_2k = col_max_1k * 2
    row_min_2k = row_min_1k * 2
    row_max_2k = row_max_1k * 2
    
    logger.debug("MODIS sinusoidal pixel crop bounds", 
                 col_min_1k=col_min_1k, col_max_1k=col_max_1k, 
                 row_min_1k=row_min_1k, row_max_1k=row_max_1k)
                 
    import subprocess
    # Run gdalinfo -json to retrieve subdatasets and global metadata
    abs_input_path = input_path.resolve()
    cmd_info = ["gdalinfo", "-json", str(abs_input_path)]
    res_info = subprocess.run(cmd_info, capture_output=True, text=True, check=True)
    gdal_meta = json.loads(res_info.stdout)
    
    subdatasets = []
    if "metadata" in gdal_meta and "SUBDATASETS" in gdal_meta["metadata"]:
        sub_dict = gdal_meta["metadata"]["SUBDATASETS"]
        subdatasets = sorted([v for k, v in sub_dict.items() if "_NAME" in k])
        
    assert len(subdatasets) > 0, "No subdatasets found in MODIS HDF4 file"
    
    with h5py.File(output_path, "w") as dest:
        # Copy file global attributes
        for attr_name, attr_val in gdal_meta.get("metadata", {}).get("", {}).items():
            dest.attrs[attr_name] = attr_val
            
        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, sub_name in enumerate(subdatasets):
                tmp_tif = Path(tmpdir) / f"sub_{idx}.tif"
                # Call gdal_translate to extract subdataset to GeoTIFF
                cmd_trans = ["gdal_translate", sub_name, str(tmp_tif)]
                subprocess.run(cmd_trans, capture_output=True, check=True)
                
                with rasterio.open(tmp_tif) as sub_src:
                    data = sub_src.read(1)
                    ds_shape = data.shape
                    
                    if ds_shape[0] == 1200 and ds_shape[1] == 1200:
                        data_clipped = data[row_min_1k:row_max_1k, col_min_1k:col_max_1k]
                    elif ds_shape[0] == 2400 and ds_shape[1] == 2400:
                        data_clipped = data[row_min_2k:row_max_2k, col_min_2k:col_max_2k]
                    else:
                        data_clipped = data
                        
                    # Extract dataset clean key name (e.g. MODIS_Grid_1km_2D:Range_1 -> Range_1)
                    clean_name = sub_name.split(":")[-1]
                    ds = dest.create_dataset(clean_name, data=data_clipped, dtype=sub_src.dtypes[0])
                    for k, v in sub_src.tags().items():
                        ds.attrs[k] = v
                        
    logger.info("Successfully saved clipped MODIS H5 file", path=str(output_path))


@app.command()
def clip_source(
    source: str = typer.Argument(..., help="The raw data source directory name (e.g. dem, era5, sentinel2)"),
    input_dir: Path = typer.Option(Path("data/bow_valley_selection_raw"), help="Path to raw data directory"),
    output_dir: Path = typer.Option(Path("data/clipped_bow_valley_selection_raw"), help="Path to save clipped output"),
    aoi_path: Path = typer.Option(Path("data/aoi.geojson"), help="Path to WGS84 AOI GeoJSON file")
):
    """Clip raw files from a single dataset source directory."""
    logger.info("Starting spatial clipping for source", source=source, input=str(input_dir), output=str(output_dir))
    
    aoi_geom = get_aoi_geometry(aoi_path)
    aoi_bbox = [AOI_LON_MIN, AOI_LAT_MIN, AOI_LON_MAX, AOI_LAT_MAX]
    
    source_dir = input_dir / source
    assert source_dir.exists(), f"Source directory {source_dir} does not exist"
    
    target_dir = output_dir / source
    target_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    # Walk directory
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            filepath = Path(root) / f
            # Calculate output path keeping the subdirectory structure
            rel_subdir = filepath.parent.relative_to(source_dir)
            dest_filepath = target_dir / rel_subdir / f
            
            # Formats mapping
            if source in ["dem", "worldcover"]:
                if f.endswith((".tif", ".tiff")):
                    clip_geotiff(filepath, dest_filepath, aoi_geom)
                    count += 1
                    
            elif source == "era5":
                if f.endswith(".nc"):
                    clip_era5(filepath, dest_filepath)
                    count += 1
                    
            elif source in ["landsat8", "landsat9"]:
                if f.endswith(".tar"):
                    clip_landsat_tar(filepath, dest_filepath, aoi_geom)
                    count += 1
                    
            elif source == "sentinel2":
                if f.endswith(".zip"):
                    clip_sentinel2_zip(filepath, dest_filepath, aoi_geom)
                    count += 1
                    
            elif source == "sentinel1":
                if f.endswith(".zip"):
                    clip_sentinel1_zip(filepath, dest_filepath)
                    count += 1
                    
            elif source == "sentinel3":
                if f.endswith(".zip"):
                    clip_sentinel3_zip(filepath, dest_filepath)
                    count += 1
                    
            elif source == "viirs":
                if f.endswith((".h5", ".hdf5")):
                    clip_sinusoidal_hdf5(filepath, dest_filepath)
                    count += 1
                    
            elif source == "modis":
                if f.endswith(".hdf"):
                    # Output converted to standard H5
                    dest_h5 = dest_filepath.with_suffix(".h5")
                    clip_modis_hdf4(filepath, dest_h5)
                    count += 1
                    
    logger.info("Spatial clipping complete for source", source=source, total_clipped=count)


@app.command()
def clip_all(
    input_dir: Path = typer.Option(Path("data/bow_valley_selection_raw"), help="Path to raw data directory"),
    output_dir: Path = typer.Option(Path("data/clipped_bow_valley_selection_raw"), help="Path to save clipped output"),
    aoi_path: Path = typer.Option(Path("data/aoi.geojson"), help="Path to WGS84 AOI GeoJSON file")
):
    """Clip raw files from all 10 raw dataset directories."""
    logger.info("Starting spatial clipping for ALL sources", input=str(input_dir), output=str(output_dir))
    
    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    
    # Subdirectories
    sources = ["dem", "worldcover", "era5", "landsat8", "landsat9", "modis", "sentinel1", "sentinel2", "sentinel3", "viirs"]
    
    for src in sources:
        src_dir = input_dir / src
        if src_dir.exists():
            clip_source(source=src, input_dir=input_dir, output_dir=output_dir, aoi_path=aoi_path)
        else:
            logger.warning("Source directory does not exist. Skipping.", source=src)
            
    logger.info("All sources spatial clipping completed successfully!")

if __name__ == "__main__":
    app()
