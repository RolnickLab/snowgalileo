from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, mapping

from src.data.config import DATA_FOLDER


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

        with rasterio.open(folder / filename) as landsat_src:
            landsat_bounds = landsat_src.bounds
            landsat_crs = landsat_src.crs
            landsat_transform = landsat_src.transform
            landsat_height = landsat_src.height
            landsat_width = landsat_src.width

            transformer = Transformer.from_crs(landsat_crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(landsat_bounds.left, landsat_bounds.bottom)
            max_lon, max_lat = transformer.transform(landsat_bounds.right, landsat_bounds.top)

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

        with rasterio.open(href) as gsp_src:
            gsp_image, gsp_transform = rasterio.mask.mask(
                gsp_src,
                [polygon],
                crop=True,
            )
            gsp_meta = gsp_src.meta.copy()
            gsp_crs = gsp_src.crs

        reprojected_cutout = np.empty((landsat_height, landsat_width), dtype=gsp_image.dtype)

        reproject(
            source=gsp_image,
            destination=reprojected_cutout,
            src_transform=gsp_transform,
            src_crs=gsp_crs,
            dst_transform=landsat_transform,
            dst_crs=landsat_crs,
            resampling=Resampling.nearest,
        )

        gsp_meta.update(
            height=reprojected_cutout.shape[0],
            width=reprojected_cutout.shape[1],
            crs=landsat_crs,
            transform=landsat_transform
        )

        output_filename = output_folder / f"gsp_{filename}"
        with rasterio.open(output_filename, "w", **gsp_meta) as dest:
            dest.write(reprojected_cutout)
