# TODO: integrate this more beautifully with the rest of the codebase!

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Union

from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--exported_tif_path", type=str, default="fsc_train_100m_tifs/all_tifs")
argparser.add_argument("--dest_folder", type=str, default="fsc_train_100m_patches")
argparser.add_argument("--output_folder", type=str, default="fsc_train_100m_masks")


# Source - https://stackoverflow.com/a
# Posted by Gareth Latty, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-06, License - CC BY-SA 4.0
def get_filename_without_epsg_extension(x):
    return re.sub(r"_EPSG:\d+\.tif{1,2}f?$", "", x)


args = argparser.parse_args()

exported_tif_path = Path(DATA_FOLDER / args.exported_tif_path)
mask_path = Path(DATA_FOLDER / args.mask_path)
output_folder = Path(DATA_FOLDER / args.output_folder)

output_folder.mkdir(parents=True, exist_ok=True)

# check the number of tifs and masks
tif_files = list(exported_tif_path.glob("*.tif")) + list(exported_tif_path.glob("*.tiff"))
mask_files = list(mask_path.glob("*.tif")) + list(mask_path.glob("*.tiff"))
output_files: Union[list[Path], list[str]] = list(output_folder.glob("*.tif")) + list(
    output_folder.glob("*.tiff")
)

print(f"Number of TIF files: {len(tif_files)}")
print(f"Number of Mask files: {len(mask_files)}")
print(f"Number of Output files: {len(output_files)}")

output_files = os.listdir(output_folder)
output_location_season = {f for f in output_files}

for file_name in os.listdir(exported_tif_path):
    src_file = os.path.join(exported_tif_path, file_name)
    if os.path.isfile(src_file):
        identifier = get_filename_without_epsg_extension(file_name)
        if identifier in output_location_season:
            print(f"Duplicate found, skipping: {identifier}")
        else:
            mask_file = mask_path / identifier
            if not mask_file.exists():
                print(f"Mask file {mask_file} does not exist. Skipping {identifier}.")
                continue
            dest_file = os.path.join(output_folder, identifier)
            shutil.copy2(mask_file, dest_file)  # Copy the file
            print(f"Copied: {mask_file} to {dest_file}")


def crop_input_to_mask_bounds(input_data, bounds, transform):
    west, south, east, north = bounds
    row_start, col_start = ~transform * (west, north)
    row_end, col_end = ~transform * (east, south)

    row_start, row_end = int(row_start), int(row_end)
    col_start, col_end = int(col_start), int(col_end)

    return input_data[:, row_start:row_end, col_start:col_end]
