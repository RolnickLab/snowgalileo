import os
import rasterio
import numpy as np
import argparse
from src.data.config import DATA_FOLDER
from pyproj import Transformer

argparser = argparse.ArgumentParser()
argparser.add_argument("--mask_folder", type=str, default="patches_UTM_1_99")

mask_folder = argparser.parse_args().__dict__["mask_folder"]
mask_path = os.path.join(DATA_FOLDER, "landsat_eval_masks", "all", mask_folder)

latitudes = []
longitudes = []

for filename in os.listdir(mask_path):
    # open file to get crs
    with rasterio.open(os.path.join(mask_path, filename)) as src:
        crs = src.crs

    utm_lat = filename.split('_')[3]
    utm_lon = filename.split('_')[4].split('.tif')[0]
    print(utm_lat, utm_lon)

    # transform from UTM to lat/lon
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(utm_lon, utm_lat)
    latitudes.append(float(lat))
    longitudes.append(float(lon))

print(len(latitudes))
print(len(longitudes))

latitudes = np.array(latitudes)
longitudes = np.array(longitudes)

np.save(os.path.join(DATA_FOLDER, "landsat_eval_masks", f'latitudes_{mask_folder}.npy'), latitudes)
np.save(os.path.join(DATA_FOLDER, "landsat_eval_masks", f'longitudes_{mask_folder}.npy'), longitudes)