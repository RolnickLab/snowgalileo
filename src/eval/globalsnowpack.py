import os
from datetime import datetime
from pathlib import Path

import rasterio

from src.data.config import DATA_FOLDER
from pystac_client import Client


def export_from_filename_for_folder(
    folder,
    start_idx: int = 0,
) -> None:
    """
    Export boxes with length and width EXPORTED_HEIGHT_WIDTH_METRES
    for the latlons specified in the filename of each file in the given folder.
    """

    # check that each file in the folder has a filename with the format L0*_YYYYMMDD_LAT_LON_SC[a number between 0 and 100]
    # and that the lat and lon are in the format of a string
    # e.g. LC09_20220101_FSC0_50.1234_8.1234.tif
    # also, create a pandas dataframe with all filenames in the format of a string
    filenames = []
    folder = Path(DATA_FOLDER / folder)

    for filename in os.listdir(folder):
        if not filename.startswith("LC0") or not filename.endswith(".tif"):
            print(f"Filename {filename} does not start with LC0_ or end with .tif")
            continue
        parts = filename.split("_")
        if len(parts) != 5:
            print(f"Filename {filename} does not have 5 parts")
            continue
        filenames.append(filename)

    filenames = sorted(filenames)[start_idx:]
    print(f"Exporting {len(filenames)} latlons: ")

    for filename in filenames:
        date = datetime.strptime(parts[1], "%Y%m%d").date()
        parts = filename.split("_")

        with rasterio.open(folder / filename) as src:
            min_yy, max_yy = src.bounds.bottom, src.bounds.top
            min_xx, max_xx = src.bounds.left, src.bounds.right
            crs = src.crs.to_string()

            # reproject to EPSG:4326
            print(f"Converting {crs} to EPSG:4326")
            from pyproj import Transformer

            # NOTE: always_xy=True ensures that the first coordinate is always in northerly direction
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(min_xx, min_yy)
            max_lon, max_lat = transformer.transform(max_xx, max_yy)

        # read in the date file with pystac and crop to lat, lon bounds with rasterio
        stac_api = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1")
        collection = stac_api.get_collection("GSP_SCE_P1D")
        items = collection.get_items()

        # find the item with the matching date
        item = None
        for it in items:
            item_date = datetime.strptime(it.properties["datetime"][:10], "%Y-%m-%d").date()
            if item_date == date:
                item = it
                break

        if item is None:
            print(f"No item found for date {date}")
            continue

        # crop the item to the lat, lon bounds with rasterio
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

        output_folder = DATA_FOLDER / "globalsnowpack_exports"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_filename = output_folder / f"gsp_{filename}"
        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(out_image)