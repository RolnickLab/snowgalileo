import os
import numpy as np
import rioxarray
import xarray as xr

import argparse
import os
from typing import cast

from src.data.config import TIFS_FOLDER, NO_DATA_VALUE

argparser = argparse.ArgumentParser()
argparser.add_argument("--tif_folder", type=str, default=TIFS_FOLDER)


def count_geotiff_values_and_nans(folder_path):
    """
    Counts the total number of values and NaNs in all GeoTIFF files in a folder.

    Parameters:
        folder_path (str): Path to the folder containing GeoTIFF files.

    Returns:
        dict: A dictionary with the total number of values and NaNs.
    """
    total_values = 0
    total_nans = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(folder_path, filename)
            try:
                with cast(xr.Dataset, rioxarray.open_rasterio(file_path)) as data:
                    values = cast(np.ndarray, data.values)
                    total_values += values.size
                    total_nans += np.count_nonzero(np.isnan(values))
                    total_nodata = np.count_nonzero(values == NO_DATA_VALUE)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return {
        "total_files": len(os.listdir(folder_path)),
        "total_values": total_values,
        "total_nans": total_nans,
        "total_nodata": total_nodata,
    }


if __name__ == "__main__":
    args = argparser.parse_args().__dict__
    result = count_geotiff_values_and_nans(args["tif_folder"])
    print("Total values:", result["total_values"])
    print("Total NaNs:", result["total_nans"])
    print("Total NoData values:", result["total_nodata"])
    print("Total files:", result["total_files"])