import os
from datetime import datetime
from pathlib import Path

import rasterio

from src.data.config import DATA_FOLDER
from pystac_client import Client
from pyproj import Transformer


def export_from_filename_for_folder(
    folder,
    start_idx: int = 0,
) -> None:
    """
    Export GlobalSnowpack cutouts that match the bounds for each file in the given folder.
    Expected filename format is L0*_YYYYMMDD_FSC[a number between 0 and 100]_LAT_LON.tif.
    """

    # Collect all filenames in the folder that match the expected format
    filenames = []
    folder = Path(DATA_FOLDER / folder)

    for filename in os.listdir(folder):
        if not filename.startswith("LC0") or not filename.endswith(".tif"):
            print(f"Format error: Filename {filename} does not start with LC0_ or end with .tif")
            continue
        parts = filename.split("_")
        if len(parts) != 5:
            print(f"Format error: Filename {filename} does not have 5 parts")
            continue
        filenames.append(filename)

    filenames = sorted(filenames)[start_idx:]
    print(f"Exporting {len(filenames)} cutouts: ")

    # Initialize the STAC client
    stac_api = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1")
    collection = stac_api.get_collection("GSP_SCE_P1D")
    items = collection.get_items()

    # Initialize the output folder
    output_folder = DATA_FOLDER / "globalsnowpack_exports"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filename = output_folder / f"gsp_{filename}"

    for filename in filenames:
        date = datetime.strptime(parts[1], "%Y%m%d").date()

        with rasterio.open(folder / filename) as src:
            min_yy, max_yy = src.bounds.bottom, src.bounds.top
            min_xx, max_xx = src.bounds.left, src.bounds.right
            crs = src.crs.to_string()

            # Reproject to EPSG:4326
            # NOTE: always_xy=True ensures that the first coordinate is always in northerly direction
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(min_xx, min_yy)
            max_lon, max_lat = transformer.transform(max_xx, max_yy)

        # Find the stac item with the matching date
        item = None
        for it in items:
            item_date = datetime.strptime(it.properties["datetime"][:10], "%Y-%m-%d").date()
            if item_date == date:
                item = it
                break

        if item is None:
            print(f"No item found for date {date}")
            continue

        # Crop the item to the lat, lon bounds with rasterio
        asset = item.assets["sce"]
        href = asset.href
        with rasterio.open(href) as src:
            out_image, out_transform = rasterio.mask.mask(
                src,
                [
                    {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [min_lon, min_lat],
                                [min_lon, max_lat],
                                [max_lon, max_lat],
                                [max_lon, min_lat],
                                [min_lon, min_lat],
                            ]
                        ],
                    }
                ],
                crop=True,
            )
            out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(out_image)