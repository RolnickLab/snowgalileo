import argparse
import os

import numpy as np
import rasterio

from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--mask_folder", type=str, default="patches_UTM_1_99")

mask_folder = argparser.parse_args().__dict__["mask_folder"]
mask_path = os.path.join(DATA_FOLDER, mask_folder)

latitudes = []
longitudes = []

for filename in os.listdir(mask_path):
    if not filename.lower().endswith(".tif"):
        continue
    with rasterio.open(os.path.join(mask_path, filename)) as src:
        crs = src.crs

    lat = filename.split("_")[3]
    lon = filename.split("_")[4].split(".tif")[0]

    # transform from UTM to lat/lon
    # transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # lon, lat = transformer.transform(utm_lon, utm_lat)
    latitudes.append(float(lat))
    longitudes.append(float(lon))

print(len(latitudes))
print(len(longitudes))

np.save(
    os.path.join(DATA_FOLDER, "latitudes.npy"),
    np.array(latitudes),
)
np.save(
    os.path.join(DATA_FOLDER, "longitudes.npy"),
    np.array(longitudes),
)
