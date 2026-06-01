import os
import sys
import tempfile
import zipfile
import tarfile
from pathlib import Path
import datetime
import rasterio
import xarray as xr
import h5py
import structlog
import shutil
from typing import Dict, Any
from shapely.geometry import box

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

# Add script location to path to allow imports
sys.path.append(str(Path(__file__).parent))
from clip_dataset import (
    clip_geotiff,
    clip_era5,
    clip_landsat_tar,
    clip_sentinel2_zip,
    clip_sentinel1_zip,
    clip_sentinel3_zip,
    clip_sinusoidal_hdf5,
    clip_modis_hdf4,
    get_aoi_geometry
)

# Paths
aoi_path = Path("data/aoi.geojson")
raw_dir = Path("data/bow_valley_selection_raw")
test_output_dir = Path("data/test_clipped_output")

# Bounding box limits in WGS84
AOI_LON_MIN, AOI_LON_MAX = -116.561936219710887, -114.527659450240762
AOI_LAT_MIN, AOI_LAT_MAX = 50.729806886838752, 52.306672311654424

# Samples Dictionary
samples = {
    "dem": raw_dir / "dem/DEM1_SAR_DGE_30_20110206T013549_20140827T013828_ADS_000000_O98e_079cf4dc/Copernicus_DSM_10_N50_00_W117_00/DEM/Copernicus_DSM_10_N50_00_W117_00_DEM.tif",
    "worldcover": raw_dir / "worldcover/ESA_WorldCover_10m_2021_v200_N48W117_Map/ESA_WorldCover_10m_2021_v200_N48W117_Map.tif",
    "era5": raw_dir / "era5/202503_ERA5LAND_totalprecip.nc",
    "landsat8": raw_dir / "landsat8/LC08_L1TP_042024_20250302_20250311_02_T1.tar",
    "landsat9": raw_dir / "landsat9/LC09_L1TP_042024_20250310_20250310_02_T1.tar",
    "modis": raw_dir / "modis/MOD09GA.A2025060.h10v03.061.2025062032054.hdf",
    "sentinel1": raw_dir / "sentinel1/S1C_IW_GRDH_1SDV_20250330T013724_20250330T013749_001664_002BB2_88AD.zip",
    "sentinel2": raw_dir / "sentinel2/S2B_MSIL1C_20250301T184219_N0511_R070_T11UPS_20250301T221310.zip",
    "sentinel3": raw_dir / "sentinel3/S3A_OL_1_EFR____20250301T183505_20250301T183805_20250302T193356_0179_123_127_1980_PS1_O_NT_004.zip",
    "viirs": raw_dir / "viirs/VNP09GA.A2025060.h10v03.002.2025061083831.h5"
}

def test_dem(aoi_geom):
    logger.info("--- Testing DEM clipping ---")
    in_path = samples["dem"]
    if not in_path.exists():
        logger.warning("DEM sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "dem_clipped.tif"
    clip_geotiff(in_path, out_path, aoi_geom)
    assert out_path.exists(), "Clipped DEM was not created"
    with rasterio.open(in_path) as orig, rasterio.open(out_path) as clipped:
        logger.info("DEM verified", orig_shape=orig.shape, clipped_shape=clipped.shape)
        assert clipped.shape[0] < orig.shape[0] and clipped.shape[1] < orig.shape[1], "DEM dimensions not subsetted"
        assert clipped.crs == orig.crs, "CRS changed"
        assert abs(clipped.bounds.left - max(orig.bounds.left, AOI_LON_MIN)) < 0.005, "Bounds mismatch"

def test_worldcover(aoi_geom):
    logger.info("--- Testing WorldCover clipping ---")
    in_path = samples["worldcover"]
    if not in_path.exists():
        logger.warning("WorldCover sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "worldcover_clipped.tif"
    clip_geotiff(in_path, out_path, aoi_geom)
    assert out_path.exists(), "Clipped WorldCover was not created"
    with rasterio.open(in_path) as orig, rasterio.open(out_path) as clipped:
        logger.info("WorldCover verified", orig_shape=orig.shape, clipped_shape=clipped.shape)
        assert clipped.shape[0] < orig.shape[0] and clipped.shape[1] < orig.shape[1], "WorldCover not subsetted"
        assert clipped.crs == orig.crs, "CRS changed"

def test_era5():
    logger.info("--- Testing ERA5 NetCDF clipping ---")
    in_path = samples["era5"]
    if not in_path.exists():
        logger.warning("ERA5 sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "era5_clipped.nc"
    clip_era5(in_path, out_path)
    assert out_path.exists(), "Clipped ERA5 was not created"
    with xr.open_dataset(in_path, engine="h5netcdf") as orig, xr.open_dataset(out_path, engine="h5netcdf") as clipped:
        logger.info("ERA5 verified", orig_dims=dict(orig.sizes), clipped_dims=dict(clipped.sizes))
        assert clipped.sizes["latitude"] < orig.sizes["latitude"], "Latitude not subsetted"
        assert clipped.sizes["longitude"] < orig.sizes["longitude"], "Longitude not subsetted"

def test_landsat(source_id: str, aoi_geom: Dict[str, Any]):
    logger.info(f"--- Testing {source_id.upper()} Tar clipping ---")
    in_path = samples[source_id]
    if not in_path.exists():
        logger.warning(f"{source_id} sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / f"{source_id}_clipped.tar"
    clip_landsat_tar(in_path, out_path, aoi_geom)
    assert out_path.exists(), f"Clipped {source_id} was not created"
    
    # Open and verify tar contents
    with tarfile.open(in_path, "r") as orig_tar, tarfile.open(out_path, "r") as clipped_tar:
        orig_names = orig_tar.getnames()
        clipped_names = clipped_tar.getnames()
        assert len(orig_names) == len(clipped_names), "File counts do not match"
        
        # Open first band and verify dimensions are smaller
        tif_bands = [x for x in clipped_names if x.endswith(".TIF")]
        if tif_bands:
            first_band = tif_bands[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract original first band
                orig_tar.extract(first_band, path=tmpdir)
                orig_tif = Path(tmpdir) / first_band
                # Extract clipped first band
                clipped_tar.extract(first_band, path=tmpdir)
                clipped_tif = Path(tmpdir) / f"clipped_{first_band}"
                shutil.move(Path(tmpdir) / first_band, clipped_tif) # Avoid overwriting
                
                # Re-extract original band to check shape
                orig_tar.extract(first_band, path=tmpdir)
                
                with rasterio.open(orig_tif) as orig, rasterio.open(clipped_tif) as clipped:
                    logger.info(f"{source_id} band verified", band=first_band, orig_shape=orig.shape, clipped_shape=clipped.shape)
                    assert clipped.shape[0] < orig.shape[0] and clipped.shape[1] < orig.shape[1], f"{source_id} band not subsetted"

def test_sentinel2(aoi_geom):
    logger.info("--- Testing Sentinel-2 Zip clipping ---")
    in_path = samples["sentinel2"]
    if not in_path.exists():
        logger.warning("Sentinel-2 sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "sentinel2_clipped.zip"
    clip_sentinel2_zip(in_path, out_path, aoi_geom)
    assert out_path.exists(), "Clipped Sentinel-2 zip was not created"
    
    with zipfile.ZipFile(in_path, "r") as orig_zip, zipfile.ZipFile(out_path, "r") as clipped_zip:
        orig_names = orig_zip.namelist()
        clipped_names = clipped_zip.namelist()
        assert len(orig_names) == len(clipped_names), "S2 zip file counts do not match"
        
        # Verify first jp2 band
        jp2_bands = [x for x in clipped_names if x.endswith(".jp2")]
        if jp2_bands:
            first_band = jp2_bands[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_zip.extract(first_band, path=tmpdir)
                orig_jp2 = Path(tmpdir) / first_band
                
                clipped_zip.extract(first_band, path=tmpdir)
                clipped_jp2 = Path(tmpdir) / f"clipped_{Path(first_band).name}"
                shutil.move(Path(tmpdir) / first_band, clipped_jp2)
                
                orig_zip.extract(first_band, path=tmpdir)
                
                with rasterio.open(orig_jp2) as orig, rasterio.open(clipped_jp2) as clipped:
                    logger.info("Sentinel-2 band verified", band=first_band, orig_shape=orig.shape, clipped_shape=clipped.shape)
                    assert clipped.shape[0] < orig.shape[0] and clipped.shape[1] < orig.shape[1], "S2 band not subsetted"

def test_sentinel1():
    logger.info("--- Testing Sentinel-1 SAR Zip clipping ---")
    in_path = samples["sentinel1"]
    if not in_path.exists():
        logger.warning("Sentinel-1 sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "sentinel1_clipped.zip"
    clip_sentinel1_zip(in_path, out_path)
    assert out_path.exists(), "Clipped Sentinel-1 zip was not created"
    
    with zipfile.ZipFile(in_path, "r") as orig_zip, zipfile.ZipFile(out_path, "r") as clipped_zip:
        orig_names = orig_zip.namelist()
        clipped_names = clipped_zip.namelist()
        assert len(orig_names) == len(clipped_names), "S1 zip file counts do not match"
        
        tiff_files = [x for x in clipped_names if x.endswith(".tiff") and "measurement/" in x]
        if tiff_files:
            first_tiff = tiff_files[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_zip.extract(first_tiff, path=tmpdir)
                orig_tif = Path(tmpdir) / first_tiff
                
                clipped_zip.extract(first_tiff, path=tmpdir)
                clipped_tif = Path(tmpdir) / f"clipped_{Path(first_tiff).name}"
                shutil.move(Path(tmpdir) / first_tiff, clipped_tif)
                
                orig_zip.extract(first_tiff, path=tmpdir)
                
                with rasterio.open(orig_tif) as orig, rasterio.open(clipped_tif) as clipped:
                    logger.info("Sentinel-1 band verified", band=first_tiff, orig_shape=orig.shape, clipped_shape=clipped.shape, orig_gcp_count=len(orig.gcps[0]), clipped_gcp_count=len(clipped.gcps[0]))
                    assert clipped.shape[0] < orig.shape[0] and clipped.shape[1] < orig.shape[1], "S1 band pixel array not cropped"
                    assert len(clipped.gcps[0]) <= len(orig.gcps[0]), "Clipped GCP count must be <= original GCP count"

def test_sentinel3():
    logger.info("--- Testing Sentinel-3 Swath Zip clipping ---")
    in_path = samples["sentinel3"]
    if not in_path.exists():
        logger.warning("Sentinel-3 sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "sentinel3_clipped.zip"
    clip_sentinel3_zip(in_path, out_path)
    assert out_path.exists(), "Clipped Sentinel-3 zip was not created"
    
    with zipfile.ZipFile(in_path, "r") as orig_zip, zipfile.ZipFile(out_path, "r") as clipped_zip:
        orig_names = orig_zip.namelist()
        clipped_names = clipped_zip.namelist()
        assert len(orig_names) == len(clipped_names), "S3 zip file counts do not match"
        
        nc_files = [x for x in clipped_names if x.endswith(".nc") and "Oa01_radiance.nc" in x]
        if nc_files:
            first_nc = nc_files[0]
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_zip.extract(first_nc, path=tmpdir)
                orig_nc = Path(tmpdir) / first_nc
                
                clipped_zip.extract(first_nc, path=tmpdir)
                clipped_nc = Path(tmpdir) / f"clipped_{Path(first_nc).name}"
                shutil.move(Path(tmpdir) / first_nc, clipped_nc)
                
                orig_zip.extract(first_nc, path=tmpdir)
                
                with h5py.File(orig_nc, "r") as orig, h5py.File(clipped_nc, "r") as clipped:
                    orig_ds = orig["Oa01_radiance"]
                    clipped_ds = clipped["Oa01_radiance"]
                    logger.info("Sentinel-3 band verified", band=first_nc, orig_shape=orig_ds.shape, clipped_shape=clipped_ds.shape)
                    assert clipped_ds.shape[0] < orig_ds.shape[0] and clipped_ds.shape[1] < orig_ds.shape[1], "S3 NetCDF not subsetted along grid"

def test_viirs():
    logger.info("--- Testing VIIRS Sinusoidal H5 clipping ---")
    in_path = samples["viirs"]
    if not in_path.exists():
        logger.warning("VIIRS sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "viirs_clipped.h5"
    clip_sinusoidal_hdf5(in_path, out_path)
    assert out_path.exists(), "Clipped VIIRS H5 file was not created"
    
    with h5py.File(in_path, "r") as orig, h5py.File(out_path, "r") as clipped:
        orig_ds = orig["HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SensorAzimuth_1"]
        clipped_ds = clipped["HDFEOS/GRIDS/VIIRS_Grid_1km_2D/Data Fields/SensorAzimuth_1"]
        logger.info("VIIRS verified", orig_shape=orig_ds.shape, clipped_shape=clipped_ds.shape)
        assert clipped_ds.shape[0] < orig_ds.shape[0] and clipped_ds.shape[1] < orig_ds.shape[1], "VIIRS Sinusoidal grid not subsetted"

def test_modis():
    logger.info("--- Testing MODIS Sinusoidal HDF4 clipping and H5 conversion ---")
    in_path = samples["modis"]
    if not in_path.exists():
        logger.warning("MODIS sample not found. Skipping.", path=str(in_path))
        return
    out_path = test_output_dir / "modis_clipped.h5"
    clip_modis_hdf4(in_path, out_path)
    assert out_path.exists(), "Clipped MODIS H5 was not created"
    
    with h5py.File(out_path, "r") as clipped:
        clipped_ds = clipped["Range_1"]
        logger.info("MODIS verified", clipped_shape=clipped_ds.shape)
        assert clipped_ds.shape == (190, 451), f"MODIS Sinusoidal grid has shape {clipped_ds.shape}, expected (190, 451)"


def main():
    logger.info("Starting explicit validation tests for all 10 raw datasets...")
    
    # 1. Parse AOI
    aoi_geom = get_aoi_geometry(aoi_path)
    
    # Clean output dir
    if test_output_dir.exists():
        import shutil
        shutil.rmtree(test_output_dir)
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests sequentially
    test_dem(aoi_geom)
    test_worldcover(aoi_geom)
    test_era5()
    test_landsat("landsat8", aoi_geom)
    test_landsat("landsat9", aoi_geom)
    test_modis()
    test_sentinel1()
    test_sentinel2(aoi_geom)
    test_sentinel3()
    test_viirs()
    
    logger.info("All 10 raw dataset clipping tests completed and validated successfully!")

if __name__ == "__main__":
    main()
