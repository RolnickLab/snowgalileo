# https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/ee_exporter.py
import os
import shutil
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, List, Optional, Union
from typing import OrderedDict as OrderedDictType

import ee
import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from pandas.compat._optional import import_optional_dependency
from tqdm import tqdm

from src.config import DEFAULT_SEED
from src.data.config import (
    DATA_FOLDER,
    DAYS_PER_TIMESTEP,
    EE_BUCKET_TIFS,
    EE_DRIVE_FOLDER_NAME,
    EE_FOLDER_TIFS,
    EE_PROJECT,
    END_YEAR,
    EXPORTED_HEIGHT_WIDTH_METRES,
    MODALITIES,
    NO_DATA_VALUE,
    NORTH_HEM_SEASONS,
    NUM_TIMESTEPS,
    SOUTH_HEM_SEASONS,
    START_YEAR,
    TIFS_FOLDER,
)
from src.data.earthengine.copernicus_dem import (
    DEM_BANDS,
    DEM_DIV_VALUES,
    DEM_SHIFT_VALUES,
    get_single_dem_image,
)
from src.data.earthengine.ee_bbox import EEBoundingBox
from src.data.earthengine.era5 import (
    ERA5_BANDS,
    ERA5_DIV_VALUES,
    ERA5_SHIFT_VALUES,
    get_single_era5_image,
)
from src.data.earthengine.esa_worldcover import (
    WC_BANDS,
    WC_DIV_VALUES,
    WC_SHIFT_VALUES,
    get_single_wc_image,
)
from src.data.earthengine.landsat import (
    LANDSAT_BANDS,
    LANDSAT_CLOUD_FLAG_BANDS,
    LANDSAT_DIV_VALUES,
    LANDSAT_SHIFT_VALUES,
    get_landsat_cloud_flag,
    get_single_landsat_image,
)
from src.data.earthengine.modis import (
    MODIS_BANDS,
    MODIS_CLOUD_FLAG_BANDS,
    MODIS_DIV_VALUES,
    MODIS_SHIFT_VALUES,
    get_modis_cloud_flag,
    get_single_modis_image,
)
from src.data.earthengine.s1 import S1_BANDS, S1_DIV_VALUES, S1_SHIFT_VALUES, get_single_s1_image
from src.data.earthengine.s2 import (
    S2_BANDS,
    S2_CLOUD_FLAG_BANDS,
    S2_DIV_VALUES,
    S2_SHIFT_VALUES,
    get_s2_cloud_flag,
    get_single_s2_image,
)
from src.data.earthengine.s3 import S3_BANDS, S3_DIV_VALUES, S3_SHIFT_VALUES, get_single_s3_image
from src.data.earthengine.utils import (
    get_ee_credentials,
    get_location_season_identifier,
    sample_season_year,
    sample_time_window,
)
from src.data.earthengine.viirs import (
    VIIRS_COARSE_BANDS,
    VIIRS_COARSE_DIV_VALUES,
    VIIRS_COARSE_SHIFT_VALUES,
    VIIRS_FINE_BANDS,
    VIIRS_FINE_DIV_VALUES,
    VIIRS_FINE_SHIFT_VALUES,
    get_single_viirs_coarse_image,
    get_single_viirs_fine_image,
)

# dataframe constants when exporting the labels
LAT = "Latitude"
LON = "Longitude"
START_DATE = date(START_YEAR, 1, 1)
END_DATE = date(END_YEAR, 12, 31)

# construct time image functions
TIME_IMAGE_FUNCTIONS = []
SPACE_IMAGE_FUNCTIONS = []

SPACE_TIME_HIGH_RES_BANDS = []
SPACE_TIME_HIGH_RES_SHIFT_VALUES = []
SPACE_TIME_HIGH_RES_DIV_VALUES = []

SPACE_TIME_MED_RES_BANDS = []
SPACE_TIME_MED_RES_SHIFT_VALUES = []
SPACE_TIME_MED_RES_DIV_VALUES = []

SPACE_TIME_LOW_RES_BANDS = []
SPACE_TIME_LOW_RES_SHIFT_VALUES = []
SPACE_TIME_LOW_RES_DIV_VALUES = []

TIME_BANDS = []
TIME_SHIFT_VALUES = []
TIME_DIV_VALUES = []

SPACE_BANDS = []
SPACE_SHIFT_VALUES = []
SPACE_DIV_VALUES = []

CLOUD_BANDS = []

for modality in MODALITIES:
    if MODALITIES[modality].get("active") and MODALITIES[modality].get("export"):
        print(MODALITIES[modality])
        try:
            band_list = globals()[f"{modality.upper()}_BANDS"]
            shift_values = globals()[f"{modality.upper()}_SHIFT_VALUES"]
            div_values = globals()[f"{modality.upper()}_DIV_VALUES"]

            if MODALITIES[modality].get("shape_type") == "s_t_h_x":
                SPACE_TIME_HIGH_RES_BANDS.extend(band_list)
                SPACE_TIME_HIGH_RES_SHIFT_VALUES.extend(shift_values)
                SPACE_TIME_HIGH_RES_DIV_VALUES.extend(div_values)

                function = globals()[f"get_single_{modality}_image"]
                TIME_IMAGE_FUNCTIONS.append(function)

            elif MODALITIES[modality].get("shape_type") == "s_t_m_x":
                SPACE_TIME_MED_RES_BANDS.extend(band_list)
                SPACE_TIME_MED_RES_SHIFT_VALUES.extend(shift_values)
                SPACE_TIME_MED_RES_DIV_VALUES.extend(div_values)

                function = globals()[f"get_single_{modality}_image"]
                TIME_IMAGE_FUNCTIONS.append(function)

            elif MODALITIES[modality].get("shape_type") == "s_t_l_x":
                SPACE_TIME_LOW_RES_BANDS.extend(band_list)
                SPACE_TIME_LOW_RES_SHIFT_VALUES.extend(shift_values)
                SPACE_TIME_LOW_RES_DIV_VALUES.extend(div_values)

                function = globals()[f"get_single_{modality}_image"]
                TIME_IMAGE_FUNCTIONS.append(function)

            elif MODALITIES[modality].get("shape_type") == "t_x":
                TIME_BANDS.extend(band_list)
                TIME_SHIFT_VALUES.extend(shift_values)
                TIME_DIV_VALUES.extend(div_values)

                function = globals()[f"get_single_{modality}_image"]
                TIME_IMAGE_FUNCTIONS.append(function)

            elif MODALITIES[modality].get("shape_type") == "sp_x":
                SPACE_BANDS.extend(band_list)
                SPACE_SHIFT_VALUES.extend(shift_values)
                SPACE_DIV_VALUES.extend(div_values)

                function = globals()[f"get_single_{modality}_image"]
                SPACE_IMAGE_FUNCTIONS.append(function)

        except KeyError:
            # TODO: make this more pretty
            if MODALITIES[modality].get("shape_type") == "clouds":
                band_list = globals()[f"{modality.upper()}_BANDS"]
                CLOUD_BANDS.extend(band_list)

                function = globals()[f"get_{modality}"]
                TIME_IMAGE_FUNCTIONS.append(function)
            else:
                print(f"Warning: Check modality '{modality}'.")

# TODO: remove this hacky assert and add a better test
assert TIME_IMAGE_FUNCTIONS == [
    get_single_s1_image,
    get_single_s2_image,
    get_single_landsat_image,
    get_single_s3_image,
    get_single_modis_image,
    get_single_viirs_fine_image,
    get_single_viirs_coarse_image,
    get_single_era5_image,
    get_modis_cloud_flag,
    get_s2_cloud_flag,
    get_landsat_cloud_flag,
]
assert SPACE_IMAGE_FUNCTIONS == [get_single_dem_image, get_single_wc_image]
assert SPACE_TIME_HIGH_RES_BANDS == S1_BANDS + S2_BANDS + LANDSAT_BANDS
assert SPACE_TIME_HIGH_RES_SHIFT_VALUES == S1_SHIFT_VALUES + S2_SHIFT_VALUES + LANDSAT_SHIFT_VALUES
assert SPACE_TIME_HIGH_RES_DIV_VALUES == S1_DIV_VALUES + S2_DIV_VALUES + LANDSAT_DIV_VALUES
assert SPACE_TIME_MED_RES_BANDS == S3_BANDS
assert SPACE_TIME_MED_RES_SHIFT_VALUES == S3_SHIFT_VALUES
assert SPACE_TIME_MED_RES_DIV_VALUES == S3_DIV_VALUES
assert SPACE_TIME_LOW_RES_BANDS == MODIS_BANDS + VIIRS_FINE_BANDS
assert SPACE_TIME_LOW_RES_SHIFT_VALUES == MODIS_SHIFT_VALUES + VIIRS_FINE_SHIFT_VALUES
assert SPACE_TIME_LOW_RES_DIV_VALUES == MODIS_DIV_VALUES + VIIRS_FINE_DIV_VALUES
assert TIME_BANDS == VIIRS_COARSE_BANDS + ERA5_BANDS
assert TIME_SHIFT_VALUES == VIIRS_COARSE_SHIFT_VALUES + ERA5_SHIFT_VALUES
assert TIME_DIV_VALUES == VIIRS_COARSE_DIV_VALUES + ERA5_DIV_VALUES
assert SPACE_BANDS == DEM_BANDS + WC_BANDS
assert SPACE_SHIFT_VALUES == DEM_SHIFT_VALUES + WC_SHIFT_VALUES
assert SPACE_DIV_VALUES == DEM_DIV_VALUES + WC_DIV_VALUES
assert CLOUD_BANDS == MODIS_CLOUD_FLAG_BANDS + S2_CLOUD_FLAG_BANDS + LANDSAT_CLOUD_FLAG_BANDS

SPACE_TIME_HIGH_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_HIGH_RES_SHIFT_VALUES)
SPACE_TIME_HIGH_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_HIGH_RES_DIV_VALUES)
SPACE_TIME_MED_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_MED_RES_SHIFT_VALUES)
SPACE_TIME_MED_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_MED_RES_DIV_VALUES)
SPACE_TIME_LOW_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_LOW_RES_SHIFT_VALUES)
SPACE_TIME_LOW_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_LOW_RES_DIV_VALUES)
TIME_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(TIME_SHIFT_VALUES)
TIME_DIV_VALUES_NP: npt.NDArray[Any] = np.array(TIME_DIV_VALUES)
SPACE_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(DEM_SHIFT_VALUES + WC_SHIFT_VALUES)
SPACE_DIV_VALUES_NP: npt.NDArray[Any] = np.array(DEM_DIV_VALUES + WC_DIV_VALUES)

# we will add latlons in dataset.py function
LOCATION_BANDS = ["x", "y", "z"]
STATIC_BANDS = LOCATION_BANDS
STATIC_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array([0, 0, 0])
STATIC_DIV_VALUES_NP: npt.NDArray[Any] = np.array([1, 1, 1])

EO_SPACE_TIME_LOW_RES_BANDS = SPACE_TIME_LOW_RES_BANDS

if MODALITIES["ndsi"].get("active"):
    SPACE_TIME_LOW_RES_BANDS = SPACE_TIME_LOW_RES_BANDS + ["NDSI"]
    SPACE_TIME_LOW_RES_SHIFT_VALUES_NP = np.append(SPACE_TIME_LOW_RES_SHIFT_VALUES_NP, [0])
    SPACE_TIME_LOW_RES_DIV_VALUES_NP = np.append(SPACE_TIME_LOW_RES_DIV_VALUES_NP, [1])

if MODALITIES["ndvi"].get("active"):
    SPACE_TIME_LOW_RES_BANDS = SPACE_TIME_LOW_RES_BANDS + ["NDVI"]
    SPACE_TIME_LOW_RES_SHIFT_VALUES_NP = np.append(SPACE_TIME_LOW_RES_SHIFT_VALUES_NP, [0])
    SPACE_TIME_LOW_RES_DIV_VALUES_NP = np.append(SPACE_TIME_LOW_RES_DIV_VALUES_NP, [1])

EO_ALL_DYNAMIC_IN_TIME_BANDS = (
    SPACE_TIME_HIGH_RES_BANDS
    + SPACE_TIME_MED_RES_BANDS
    + EO_SPACE_TIME_LOW_RES_BANDS
    + TIME_BANDS
    + CLOUD_BANDS
)

EO_ALL_DYNAMIC_IN_TIME_BANDS_NP = np.array(EO_ALL_DYNAMIC_IN_TIME_BANDS)

# spatial resolution per pixel: 10m, 20m, or 30m
SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_NIR": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B8"]],
        "S2_SWIR": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B11", "B12"]],
        "L_RGB": [
            SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B2_landsat", "B3_landsat", "B4_landsat"]
        ],
        "L_NIR": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B5_landsat"]],
        "L_SWIR": [SPACE_TIME_HIGH_RES_BANDS.index(b) for b in ["B6_landsat", "B7_landsat"]],
    }
)

SPACE_TIME_MED_RES_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S3_NIR": [SPACE_TIME_MED_RES_BANDS.index(b) for b in ["Oa17_radiance", "Oa21_radiance"]],
    }
)

SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "MODIS_RGB": [
            SPACE_TIME_LOW_RES_BANDS.index(b)
            for b in ["sur_refl_b01", "sur_refl_b03", "sur_refl_b04"]
        ],
        "MODIS_NIR": [SPACE_TIME_LOW_RES_BANDS.index(b) for b in ["sur_refl_b02"]],
        "MODIS_SWIR": [
            SPACE_TIME_LOW_RES_BANDS.index(b)
            for b in ["sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
        ],
        "VIIRS_RGB_FINE": [SPACE_TIME_LOW_RES_BANDS.index(b) for b in ["I1"]],
        "VIIRS_VNIR_FINE": [SPACE_TIME_LOW_RES_BANDS.index(b) for b in ["I3"]],
    }
)

if MODALITIES["ndsi"].get("active"):
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.update({"NDSI": [SPACE_TIME_LOW_RES_BANDS.index("NDSI")]})

if MODALITIES["ndvi"].get("active"):
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.update({"NDVI": [SPACE_TIME_LOW_RES_BANDS.index("NDVI")]})

TIME_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "VIIRS_RGB_COARSE": [TIME_BANDS.index(b) for b in ["M5", "M7"]],
        "VIIRS_VNIR_COARSE": [TIME_BANDS.index(b) for b in ["M10"]],
        "VIIRS_SWIR_COARSE": [TIME_BANDS.index(b) for b in ["M11"]],
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
    }
)

# spatial resolution per pixel: 30m
SPACE_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "DEM": [SPACE_BANDS.index(b) for b in DEM_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
    }
)


def get_ee_task_list(key: str = "description") -> List[str]:
    """Gets a list of all active tasks in the EE task list."""
    task_list = ee.data.getTaskList()
    return [
        task[key]
        for task in tqdm(task_list, desc="Loading Earth Engine tasks")
        if task["state"] in ["READY", "RUNNING", "FAILED"]
    ]


def get_ee_task_amount(prefix: Optional[str] = None) -> int:
    """
    Gets amount of active tasks in Earth Engine.
    Args:
        prefix: Prefix to filter tasks.
    Returns:
        Amount of active tasks.
    """
    ee_prefix = None if prefix is None else ee_safe_str(prefix)
    amount = 0
    task_list = ee.data.getTaskList()
    for t in tqdm(task_list):
        valid_state = t["state"] in ["READY", "RUNNING"]
        if valid_state and (ee_prefix is None or t["description"].startswith(ee_prefix)):
            amount += 1
    return amount


def get_cloud_tif_list(
    dest_bucket: str, prefix: str = EE_FOLDER_TIFS, region: str = "us-central1"
) -> List[str]:
    """Gets a list of all cloud-free TIFs in a bucket."""
    storage = import_optional_dependency("google.cloud.storage")
    cloud_tif_list_iterator = storage.Client().list_blobs(dest_bucket, prefix=prefix)
    try:
        tif_list = [
            blob.name
            for blob in tqdm(cloud_tif_list_iterator, desc="Loading tifs already on Google Cloud")
        ]
    except Exception as e:
        raise Exception(
            f"{e}\nPlease create the Google Cloud bucket: {dest_bucket}"
            + f"\nCommand: gsutil mb -l {region} gs://{dest_bucket}"
        )
    print(f"Found {len(tif_list)} already exported tifs")
    return tif_list


def make_combine_bands_function(bands: List[str]):
    def combine_bands(current, previous):
        # Transforms an Image Collection with 1 band per Image into a single
        # Image with items as bands
        # Author: Jamie Vleeshouwer

        # Rename the band
        previous = ee.Image(previous)
        current = current.select(bands)
        # Append it to the result (Note: only return current item on first
        # element/iteration)
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )

    return combine_bands


def ee_safe_str(s: str):
    """Earth Engine descriptions only allow certain characters"""
    return s.replace(".", "-").replace("=", "-").replace("/", "-")[:100]


def create_ee_image(
    polygon: ee.Geometry,
    interval_start_date: date,
    interval_end_date: date,
    days_per_timestep: int = DAYS_PER_TIMESTEP,
) -> ee.Image:
    # TODO: change function header
    """
    Returns an ee.Image which we can then export.
    This image will contain S1, S2, ERA5 and Dynamic World data
    between start_date and end_date, in intervals of
    days_per_timestep. Each timestep will be a different channel in the
    image (e.g. if I have 3 timesteps, then I'll have VV, VV_1, VV_2 for the
    S1 VV bands). The static in time SRTM bands will also be in the image.
    """
    image_collection_list: List[ee.Image] = []
    cur_date = interval_start_date
    cur_end_date = cur_date + timedelta(days=days_per_timestep)

    # Note: we add a day to the end date to make sure we get the last day inclusive
    # (the ee.filterDate function is exclusive)
    # TODO: check if this makes sense if days_per_timestep is greater than 1
    while cur_end_date <= interval_end_date + timedelta(days=days_per_timestep):
        image_list: List[ee.Image] = []

        for image_function in TIME_IMAGE_FUNCTIONS:
            image_list.append(
                image_function(
                    region=polygon,
                    start_date=cur_date.strftime("%Y-%m-%d"),
                    end_date=cur_end_date.strftime("%Y-%m-%d"),
                )
            )

        image_collection_list.append(ee.Image.cat(image_list))
        cur_date += timedelta(days=days_per_timestep)
        cur_end_date += timedelta(days=days_per_timestep)

    # now, we want to take our image collection and append the bands into a single image
    imcoll = ee.ImageCollection(image_collection_list)
    combine_bands_function = make_combine_bands_function(EO_ALL_DYNAMIC_IN_TIME_BANDS)
    img = ee.Image(imcoll.iterate(combine_bands_function))

    # we add the static in time images
    total_image_list: List[ee.Image] = [img]
    for space_image_function in SPACE_IMAGE_FUNCTIONS:
        total_image_list.append(
            space_image_function(
                region=polygon,
                start_date=cur_date.strftime("%Y-%m-%d"),
                end_date=cur_end_date.strftime("%Y-%m-%d"),
            )
        )

    return ee.Image.cat(total_image_list)


class EarthEngineExporter:
    """
    Export satellite data from Earth engine. It's called using the following
    script:
    ```
    from src.data import EarthEngineExporter
    EarthEngineExporter(dest_bucket="bucket_name").export_for_labels(df)
    ```
    :param check_ee: Whether to check Earth Engine before exporting
    :param check_gcp: Whether to check Google Cloud Storage before exporting,
        google-cloud-storage must be installed.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    :param dest_bucket: The bucket to export to, google-cloud-storage must be installed.
    """

    def __init__(
        self,
        dest_bucket=EE_BUCKET_TIFS,
        dest_drive_folder: str = EE_DRIVE_FOLDER_NAME,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
        mode: str = "drive",
        no_data_val: int = NO_DATA_VALUE,
        tifs_folder=None,
    ) -> None:
        assert mode in ["cloud", "drive", "url"]
        self.mode = mode
        if tifs_folder is None:
            self.tifs_folder = Path(TIFS_FOLDER)
        else:
            self.tifs_folder = DATA_FOLDER / tifs_folder
        if mode == "url":
            print(
                f"Mode: url. Files will be saved to {self.tifs_folder} and rsynced to google cloud"
            )
        self.surrounding_metres = EXPORTED_HEIGHT_WIDTH_METRES / 2
        self.dest_bucket = dest_bucket
        self.dest_drive_folder = dest_drive_folder

        initialize_args = {
            "credentials": credentials if credentials else get_ee_credentials(),
            "project": EE_PROJECT,
        }
        if mode == "url":
            initialize_args["opt_url"] = "https://earthengine-highvolume.googleapis.com"
        ee.Initialize(**initialize_args)
        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []
        self.check_gcp = check_gcp
        self.cloud_tif_list = get_cloud_tif_list(dest_bucket) if self.check_gcp else []
        self.local_tif_list = [x.name for x in self.tifs_folder.glob("*.tif*")]
        self.cloud_location_season_tif_list = [
            get_location_season_identifier(x) for x in self.cloud_tif_list
        ]
        self.local_location_season_tif_list = [
            get_location_season_identifier(x) for x in self.local_tif_list
        ]
        self.no_data_val = no_data_val

    def sync_local_and_gcloud(self):
        os.system(
            f"gcloud storage rsync -r {self.tifs_folder} gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS}"
        )

    def _export_for_polygon(
        self,
        polygon: ee.Geometry,
        polygon_identifier: Union[int, str],
        interval_start_date: date,
        interval_end_date: date,
        file_dimensions: Optional[int] = None,
    ) -> bool:
        cloud_filename = f"{str(polygon_identifier)}"
        local_filename = f"{str(polygon_identifier).replace('/', '_')}.tif"
        location_season_identifier = get_location_season_identifier(local_filename)

        # Description of the export cannot contain certrain characters
        description = ee_safe_str(cloud_filename)

        if cloud_filename in self.cloud_tif_list:
            # checks that we haven't already exported this file
            print(f"{cloud_filename}.tif already in cloud_tif_files", flush=True)
            return False

        if local_filename in self.local_tif_list:
            # checks that we haven't already exported this file
            print(f"{local_filename} already in local_tif_files, but not in the cloud", flush=True)
            return False

        if location_season_identifier in self.local_location_season_tif_list:
            # checks that we haven't already exported this file
            print(f"{location_season_identifier} already in local_tif_files", flush=True)
            return False

        # Check if task is already started in EarthEngine
        if description in self.ee_task_list:
            print(f"{description} already in ee task list", flush=True)
            return False

        if len(self.ee_task_list) >= 3000:
            # we can only have 3000 running exports at once
            print("3000 exports started", flush=True)
            return False

        img = create_ee_image(polygon, interval_start_date, interval_end_date)

        if self.mode == "cloud":
            try:
                ee.batch.Export.image.toCloudStorage(
                    bucket=self.dest_bucket,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    crs="EPSG:4326",
                    scale=10,
                    region=polygon,
                    maxPixels=1e13,
                    fileDimensions=file_dimensions,
                    formatOptions={"noData": self.no_data_val},
                ).start()
                self.ee_task_list.append(description)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}", flush=True)
                return False
        elif self.mode == "drive":
            try:
                ee.batch.Export.image.toDrive(
                    folder=self.dest_drive_folder,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    crs="EPSG:4326",
                    scale=10,
                    region=polygon,
                    maxPixels=1e13,
                    fileDimensions=file_dimensions,
                    formatOptions={"noData": self.no_data_val},
                ).start()
                self.ee_task_list.append(description)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}", flush=True)
                return False
        elif self.mode == "url":
            try:
                url = img.getDownloadURL(
                    {
                        "region": polygon,
                        "crs": "EPSG:4326",
                        "scale": 10,
                        "filePerBand": False,
                        "format": "GEO_TIFF",
                    }
                )
                r = requests.get(url, stream=True)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}", flush=True)
                return False
            if r.status_code != 200:
                print(f"Task failed with status {r.status_code}", flush=True)
                return False
            else:
                local_path = Path(self.tifs_folder / local_filename)
                with local_path.open("wb") as f:
                    shutil.copyfileobj(r.raw, f)
                    print("Downloaded file " + local_filename, flush=True)
        return True

    def export_for_latlons(
        self,
        latlons: pd.DataFrame,
        num_exports_to_start: int = 3000,
    ) -> None:
        """
        Export boxes with length and width EXPORTED_HEIGHT_WIDTH_METRES
        for the points in latlons (where latlons is a dataframe with
        the columns "lat" and "lon")
        """
        for expected_column in [LAT, LON]:
            assert expected_column in latlons

        exports_started = 0
        print(f"Exporting {len(latlons)} latlons: ")

        for i, row in tqdm(latlons.iterrows(), desc="Exporting", total=len(latlons)):
            ee_bbox = EEBoundingBox.from_centre(
                # worldstrat points are strings
                mid_lat=float(row[LAT]),
                mid_lon=float(row[LON]),
                surrounding_metres=int(self.surrounding_metres),
            )

            seed = DEFAULT_SEED + i

            # sample seasons based on the hemisphere
            if float(row[LAT]) < 0:
                SEASONS = SOUTH_HEM_SEASONS
            else:
                SEASONS = NORTH_HEM_SEASONS

            # Sample each point for each season
            for season in SEASONS.items():
                season_key = season[0]
                # randomly choose year to sample from
                sampled_season = sample_season_year(season, START_YEAR, END_YEAR, seed=seed)

                SEASON_START_DATE = sampled_season[0]
                SEASON_END_DATE = sampled_season[1]

                WINDOW_START_DATE, WINDOW_END_DATE = sample_time_window(
                    SEASON_START_DATE, SEASON_END_DATE, NUM_TIMESTEPS, seed=seed
                )

                export_started = self._export_for_polygon(
                    polygon=ee_bbox.to_ee_polygon(),
                    polygon_identifier=ee_bbox.get_identifier(
                        season_key, WINDOW_START_DATE, WINDOW_END_DATE
                    ),
                    interval_start_date=WINDOW_START_DATE,
                    interval_end_date=WINDOW_END_DATE,
                )
                if export_started:
                    exports_started += 1
                    if (
                        num_exports_to_start is not None
                        and exports_started >= num_exports_to_start
                    ):
                        print(f"Started {exports_started} exports. Ending export")
                        return None

        if self.mode == "url":
            print("Export finished. Syncing to google cloud")
            self.sync_local_and_gcloud()
            print("Finished sync")
