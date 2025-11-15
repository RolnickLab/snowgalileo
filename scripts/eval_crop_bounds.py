# TODO: integrate this more beautifully with the rest of the codebase!

import gc
import os
import re
from pathlib import Path
from typing import Union

import rasterio

from src.data.config import DATA_FOLDER


def crop_input_to_mask_bounds(input_data, bounds, transform):
    west, south, east, north = bounds
    row_start, col_start = ~transform * (west, north)
    row_end, col_end = ~transform * (east, south)

    row_start, row_end = int(row_start), int(row_end)
    col_start, col_end = int(col_start), int(col_end)

    return input_data[:, row_start:row_end, col_start:col_end]


def get_filename_without_epsg_extension(x):
    return re.sub(r"_EPSG:\d+\.tif{1,2}f?$", "", x)


exported_tif_path = Path(DATA_FOLDER / "landsat_eval_tifs" / "patches_UTM_5_95_sorted")
mask_path = Path(DATA_FOLDER / "landsat_eval_masks" / "all" / "patches_UTM_5_95_subset")
output_folder = Path(DATA_FOLDER / "landsat_eval_tifs" / "patches_UTM_5_95_cropped")

output_folder.mkdir(parents=True, exist_ok=True)

# check the number of tifs and masks that are left to be processed
output_files: Union[list[Path], list[str]] = list(output_folder.glob("*.tif")) + list(
    output_folder.glob("*.tiff")
)
tif_files = list(exported_tif_path.glob("*.tif")) + list(exported_tif_path.glob("*.tiff"))
mask_files = [f for f in mask_path.glob("*.tif") if f.name not in output_files] + [
    f for f in mask_path.glob("*.tiff") if f.name not in output_files
]

print(f"Number of remaining TIF files: {len(tif_files)}")
print(f"Number of remaining Mask files: {len(mask_files)}")
print(f"Number of already cropped Output files: {len(output_files)}")

output_files = os.listdir(output_folder)
output_location_season = {f for f in output_files}

for file_name in os.listdir(mask_path):
    mask_file = os.path.join(mask_path, file_name)
    if os.path.isfile(mask_file):
        # check if identifier is in output_location_season, where the actual filename can be longer
        if any(file_name in fname for fname in output_location_season):
            print(f"Duplicate found, skipping: {file_name}")
        else:
            matching_input_files = [f for f in tif_files if file_name in os.path.basename(f)]
            for input_file in matching_input_files:
                with rasterio.open(input_file) as input_src:
                    input_data = input_src.read()
                    input_crs = input_src.crs

                    with rasterio.open(mask_file) as mask_src:
                        mask_bounds = mask_src.bounds
                        mask_transform = mask_src.transform
                        mask_crs = mask_src.crs

                        # TODO: an die Remote Sensing Leute: funktioniert das so?
                        transformed_mask_transform = mask_transform * rasterio.Affine.scale(
                            0.1, 0.1
                        )

                        assert input_crs == mask_crs, "Input and mask CRS do not match."

                        cropped_data = crop_input_to_mask_bounds(
                            input_data, mask_bounds, transformed_mask_transform
                        )

                        assert cropped_data.shape[1:] == (100, 100), (
                            "Cropped data shape does not match mask shape."
                        )

                        dest_file = os.path.join(output_folder, file_name)

                        # Save the cropped data to the output folder
                        with rasterio.open(
                            dest_file,
                            "w",
                            driver="GTiff",
                            height=cropped_data.shape[1],
                            width=cropped_data.shape[2],
                            count=cropped_data.shape[0],
                            dtype=cropped_data.dtype,
                            crs=input_src.crs,
                            transform=transformed_mask_transform,
                        ) as dst:
                            dst.write(cropped_data)
                            print(f"Copied and cropped: {input_file} to {dest_file}")
                del input_data, cropped_data
                gc.collect()
