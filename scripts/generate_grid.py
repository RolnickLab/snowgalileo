# Generates a csv file with the class pixel amount for each grid in ESA WorldCover map
# Requires: pip install geopandas rioxarray tqdm -q
# Usage: python3 generate_grid.py
# Output: esa_grid.csv
# Once outputted, the file can be uploaded and made public by running:
# gcloud storage cp esa_grid.csv gs://lem-assets
# gcloud storage acl ch -u AllUsers:R gs://lem-assets/esa_grid.csv
# Author: Ivan Zvonkov

from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from tqdm import tqdm

TILE_SIZE = 1000
GRID_PATH = Path()

legend = {
    10: "Trees",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Barren / sparse vegetation",
    70: "Snow and ice",
    80: "Open water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
url = f"{s3_url_prefix}/v100/2020/esa_worldcover_2020_grid.geojson"
grid = gpd.read_file(url)

output_dir = Path(__file__).parents[1] / "data/esa_grid_granular.csv"
if output_dir.exists():
    print(f"Found file at {output_dir}. Resuming")
    output_dict: Dict[str, List] = pd.read_csv(output_dir).to_dict("list")
else:
    print(f"No file found at {output_dir}. Starting from scratch")
    output_dict = {"tile_id": [], "lat": [], "lon": []}

for k in legend.keys():
    output_dict[f"class_{k}"] = []

for tile_i in tqdm(range(len(grid))):
    tile_name = grid.iloc[tile_i]["ll_tile"]
    if tile_i in output_dict["tile_id"]:
        print(f"Skipping {tile_name}")
        continue
    url = f"{s3_url_prefix}/v100/2020/map/ESA_WorldCover_10m_2020_v100_{tile_name}_Map.tif"
    tif_file = rioxarray.open_rasterio(url, cache=False)

    for x_i in tqdm(range(0, len(tif_file.x), TILE_SIZE), leave=False):  # type: ignore
        for y_i in tqdm(range(0, len(tif_file.y), TILE_SIZE), leave=False):  # type: ignore
            sub_tile = tif_file.isel(x=slice(x_i, x_i + TILE_SIZE), y=slice(y_i, y_i + TILE_SIZE))  # type: ignore
            keys, amounts = np.unique(sub_tile, return_counts=True)

            if (len(keys) == 1) and (keys[0] == 0):
                continue

            output_dict["tile_id"].append(tile_name)
            output_dict["lat"].append(sub_tile.y.mean().item())  # type: ignore
            output_dict["lon"].append(sub_tile.x.mean().item())  # type: ignore
            for k in legend.keys():
                if k in keys:
                    output_dict[f"class_{k}"].append(amounts[keys == k][0])
                else:
                    output_dict[f"class_{k}"].append(0)
    pd.DataFrame(output_dict).to_csv(output_dir, index=False)
