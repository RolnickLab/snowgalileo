from datetime import datetime
from pathlib import Path
from einops import rearrange
import json
from src.eval.metrics import compute_classification_metrics

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
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Export GlobalSnowpack cutouts that match the bounds for each file in the given folder.
    Expected filename format is LC0*_YYYYMMDD_FSC[a number between 0 and 100]_LAT_LON.tif.
    """

    all_landsat_labels = []
    all_gsp_labels = []

    folder = Path(DATA_FOLDER / folder)

    # Collect valid filenames
    filenames = []
    for path in folder.iterdir():
        if not path.name.startswith("LC0") or not path.name.endswith(".tif"):
            continue
        parts = path.name.split("_")
        if len(parts) != 5:
            continue
        filenames.append(path.name)

    filenames = sorted(filenames)[start_idx:]
    print(f"Exporting {len(filenames)} cutouts")

    # STAC client (Global SnowPack via DLR)
    stac_api = Client.open("https://geoservice.dlr.de/eoc/ogc/stac/v1")

    output_folder = DATA_FOLDER / "globalsnowpack_exports"
    output_folder.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        parts = filename.split("_")
        date = datetime.strptime(parts[1], "%Y%m%d").date()

        with rasterio.open(folder / filename) as landsat_src:
            landsat_labels = landsat_src.read(1)
        all_landsat_labels.append(
            rearrange(landsat_labels, "h w -> (h w)")
        )

        output_filename = output_folder / f"gsp_{filename}"

        if output_filename.exists():
            print(f"Skipping existing file: {output_filename.name}")
            with rasterio.open(output_filename) as existing:
                existing_data = existing.read(1)
            all_gsp_labels.append(
                rearrange(existing_data, "h w -> (h w)")
            )
            continue

        # ---- Read Landsat ----
        with rasterio.open(folder / filename) as landsat_src:
            landsat_labels = landsat_src.read(1)
            landsat_bounds = landsat_src.bounds
            landsat_crs = landsat_src.crs
            landsat_transform = landsat_src.transform
            landsat_height = landsat_src.height
            landsat_width = landsat_src.width

        all_landsat_labels.append(
            rearrange(landsat_labels, "h w -> (h w)")
        )

        # Transform bounds to WGS84
        transformer = Transformer.from_crs(
            landsat_crs, "EPSG:4326", always_xy=True
        )

        min_lon, min_lat = transformer.transform(
            landsat_bounds.left, landsat_bounds.bottom
        )
        max_lon, max_lat = transformer.transform(
            landsat_bounds.right, landsat_bounds.top
        )

        # ---- Search GlobalSnowpack ----
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

        gsp_image = gsp_image[0]

        reprojected_cutout = np.empty(
            (landsat_height, landsat_width),
            dtype=gsp_image.dtype,
        )

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
            height=landsat_height,
            width=landsat_width,
            crs=landsat_crs,
            transform=landsat_transform,
            count=1
        )

        all_gsp_labels.append(
            rearrange(reprojected_cutout, "h w -> (h w)")
        )

        with rasterio.open(output_filename, "w", **gsp_meta) as dest:
            dest.write(reprojected_cutout, 1)

    return all_landsat_labels, all_gsp_labels


if __name__ == "__main__":
    # Mapping from https://download.geoservice.dlr.de/GSP/files/daily/GSPDAILY_README.txt 
    # Fill value 0.0 will be mapped to -1, which will be discarded in metric computations
    def gsp_binary_mapping(arr, fill_value = -1):
        result = np.full_like(arr, fill_value=fill_value)
        result[(1 < arr) & (arr < 64)] = 0
        result[(64 <= arr)] = 1
        return result

    def landsat_binary_mapping(arr, fill_value= -1):
        result = np.full_like(arr, fill_value=fill_value)
        result[(0 == arr)] = 0
        result[(0 < arr)] = 1
        return result

    fill_value = -1

    labels_folder = Path(DATA_FOLDER / "fsc_test_rockies_100m_masks/test")

    all_landsat_labels, all_gsp_labels = export_from_filename_for_folder(labels_folder)

    assert len(all_gsp_labels) == len(all_landsat_labels)

    gsp_labels = gsp_binary_mapping(np.concatenate(all_gsp_labels), fill_value=fill_value)
    landsat_labels = landsat_binary_mapping(np.concatenate(all_landsat_labels), fill_value=fill_value)

    valid_data_mask = gsp_labels != fill_value

    results = compute_classification_metrics(gsp_labels[valid_data_mask], landsat_labels[valid_data_mask])

    results_path = Path(f"./globalsnowpack_results.json")
    with results_path.open("w") as f:
        json.dump(results, f)