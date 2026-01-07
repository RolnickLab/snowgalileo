import argparse
import os

import numpy as np
import rasterio

from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--mask_folder", type=str, default="patches_UTM_1_99")

mask_folder = argparser.parse_args().__dict__["mask_folder"]
mask_path = os.path.join(DATA_FOLDER, "landsat_eval_masks", "all", mask_folder)

targets_list = []

for i in os.listdir(mask_path):
    print(i)
    with rasterio.open(os.path.join(mask_path, i)) as src:
        data = src.read()
        targets_list.append(data)

targets = np.concatenate(targets_list, axis=0)
print(len(targets_list))
np.save(os.path.join(DATA_FOLDER, "landsat_eval_masks", f"all_targets_{mask_folder}.npy"), targets)
print("Saved all targets to", os.path.join(mask_path, f"all_targets_{mask_folder}.npy"))
