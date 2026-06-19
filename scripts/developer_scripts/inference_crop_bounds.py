# TODO: integrate this more beautifully with the rest of the codebase!

import argparse
import gc
from pathlib import Path

import rasterio
from rasterio.transform import Affine

from src.data.config import DATA_FOLDER


def crop_center(input_data, crop_height=100, crop_width=100):
    """
    Crop the center region of shape (crop_height, crop_width)
    from input_data with shape (C, H, W).
    """
    _, height, width = input_data.shape

    row_start = max((height - crop_height) // 2, 0)
    col_start = max((width - crop_width) // 2, 0)

    row_end = row_start + crop_height
    col_end = col_start + crop_width

    cropped = input_data[:, row_start:row_end, col_start:col_end]

    return cropped, row_start, col_start


def update_transform_for_crop(transform, row_start, col_start):
    """Update geotransform after cropping."""
    return transform * Affine.translation(col_start, row_start)


argparser = argparse.ArgumentParser()
argparser.add_argument("--exported_tif_path", type=str, default="fsc_train_100m_tifs/all_tifs")
argparser.add_argument("--cropped_path", type=str, default="fsc_train_100m_cropped")
argparser.add_argument("--crop_size", type=int, default=100)

args = argparser.parse_args()

exported_tif_path = Path(DATA_FOLDER / args.exported_tif_path)
output_folder = Path(DATA_FOLDER / args.cropped_path)
crop_size = args.crop_size

output_folder.mkdir(parents=True, exist_ok=True)

# Input and output files
output_files = list(output_folder.glob("*.tif")) + list(output_folder.glob("*.tiff"))
tif_files = list(exported_tif_path.glob("*.tif")) + list(exported_tif_path.glob("*.tiff"))

processed_files = {f.name for f in output_files}

print(f"Number of input TIF files: {len(tif_files)}")
print(f"Number of already cropped Output files: {len(output_files)}")

for input_file in tif_files:
    file_name = input_file.name

    if file_name in processed_files:
        print(f"Skipping already processed file: {file_name}")
        continue

    try:
        with rasterio.open(input_file) as input_src:
            input_data = input_src.read()

            cropped_data, row_start, col_start = crop_center(
                input_data,
                crop_height=crop_size,
                crop_width=crop_size,
            )

            new_transform = update_transform_for_crop(
                input_src.transform,
                row_start,
                col_start,
            )

            dest_file = output_folder / file_name

            with rasterio.open(
                dest_file,
                "w",
                driver="GTiff",
                height=cropped_data.shape[1],
                width=cropped_data.shape[2],
                count=cropped_data.shape[0],
                dtype=cropped_data.dtype,
                crs=input_src.crs,
                transform=new_transform,
            ) as dst:
                dst.write(cropped_data)

            print(f"Cropped center 100x100: {input_file} -> {dest_file}")

        del input_data, cropped_data
        gc.collect()

    except Exception as e:
        print(f"Failed processing {input_file}: {e}")
