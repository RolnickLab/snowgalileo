import os
from datetime import datetime
from pathlib import Path

import rasterio

from src.data.config import DATA_FOLDER
from pystac_client import Client
from pyproj import Transformer
from shapely.geometry import box, mapping


def export_from_filename_for_folder(
    folder: str,
    start_idx: int = 0,
) -> None:
    """
    Export GlobalSnowpack cutouts that match the bounds for each file in the given folder.
    Expected filename format is L0*_YYYYMMDD_FSC[a number between 0 and 100]_LAT_LON.tif.
    """

    # Collect all filenames in the folder that match the expected format
    filenames = []
    folder = Path(DATA_FOLDER / folder)

    for path in folder.iterdir():
        if not path.name.startswith("LC0") or not path.name.endswith(".tif"):
            continue
        parts = path.name.split("_")
        if len(parts) != 5:
            continue
        filenames.append(path.name)

    filenames = sorted(filenames)[start_idx:]
    print(f"Exporting {len(filenames)} cutouts: ")

    # Initialize the STAC client
    stac_api = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1")

    # Initialize the output folder
    output_folder = DATA_FOLDER / "globalsnowpack_exports"
    output_folder.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        date = datetime.strptime(parts[1], "%Y%m%d").date()

        with rasterio.open(folder / filename) as src:
            bounds = src.bounds
            crs = src.crs

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
            max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

        # Search by date
        search = stac_api.search(
            collections=["GSP_SCE_P1D"],
            datetime=date.isoformat(),
        )
        items = list(search.items())

        if not items:
            print(f"No item found for {date}")
            continue

        item = items[0]
        href = item.assets["sce"].href

        polygon = mapping(box(min_lon, min_lat, max_lon, max_lat))

        with rasterio.open(href) as src:
            out_image, out_transform = rasterio.mask.mask(
                src,
                [polygon],
                crop=True,
            )
            out_meta = src.meta.copy()

        out_meta.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )
        
        output_filename = output_folder / f"gsp_{filename}"
        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(out_image)