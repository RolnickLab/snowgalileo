import argparse
from pathlib import Path

import rasterio
from rasterio.windows import Window

from src.data.config import DATA_FOLDER


def center_crop_tifs(input_folder, output_folder, crop_height=100, crop_width=100):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_folder.glob("*.tif"))

    for tif_file in tif_files:
        print(f"Processing {tif_file.name}")

        with rasterio.open(tif_file) as src:
            height = src.height
            width = src.width

            if height < crop_height or width < crop_width:
                print(f"Skipping {tif_file.name}: smaller than crop size")
                continue

            # Compute center crop coordinates
            row_start = (height - crop_height) // 2
            col_start = (width - crop_width) // 2

            window = Window(
                col_off=col_start, row_off=row_start, width=crop_width, height=crop_height
            )

            cropped = src.read(window=window)

            # Update transform
            transform = src.window_transform(window)

            profile = src.profile.copy()
            profile.update({"height": crop_height, "width": crop_width, "transform": transform})

            output_path = output_folder / tif_file.name

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(cropped)

    print("Done.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--exported_tif_path", type=str, default="fsc_train_100m_tifs/all_tifs")
    argparser.add_argument("--cropped_path", type=str, default="fsc_train_100m_cropped")
    args = argparser.parse_args()

    center_crop_tifs(
        input_folder=Path(DATA_FOLDER / args.exported_tif_path),
        output_folder=Path(DATA_FOLDER / args.cropped_path),
        crop_height=100,
        crop_width=100,
    )
