import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import tempfile
import os
import shutil

def resample_resolution(tif_path):
    with rasterio.open(tif_path) as src:
        if src.crs.to_string() == 'EPSG:4326':
            print(f"File '{tif_path}' is already in EPSG:4326. No reprojection needed.")
            return
        
        transform, width, height = calculate_default_transform(
            src.crs, 'EPSG:4326', src.width, src.height, *src.bounds)
        print(height, width)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': 'EPSG:4326',
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Use a temporary file to avoid overwriting during processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmpfile:
            temp_path = tmpfile.name
        
        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:4326',
                    resampling=Resampling.nearest)
        
        shutil.move(temp_path, tif_path)
        print(f"Reprojection complete. Input file '{tif_path}' has been updated.")