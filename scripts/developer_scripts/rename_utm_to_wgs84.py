import argparse
from pathlib import Path

import rasterio
from pyproj import Transformer

from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--folder", type=str, default="landsat_eval_masks/patches_UTM_5_95")

args = argparser.parse_args().__dict__

path = Path(DATA_FOLDER / args["folder"])

# connvert all files in the folder from _UTM_ to _WGS84_ in their filename
for file in path.glob("*.tif"):
    with rasterio.open(file) as src:
        crs = src.crs.to_string()

    old_name = file.name
    parts = old_name.split(".tif")[0].split("_")
    utm_lat = float(parts[3])
    utm_lon = float(parts[4])

    # NOTE: always_xy=True ensures that the first coordinate is always in northerly direction
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(utm_lon, utm_lat)

    new_name = file.name.replace(str(utm_lon), str(lon)).replace(str(utm_lat), str(lat))
    file.rename(path / new_name)
