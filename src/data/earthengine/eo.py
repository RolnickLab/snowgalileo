# https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/ee_exporter.py
import os
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Union

import ee
import numpy as np
import pandas as pd
import requests
from pandas.compat._optional import import_optional_dependency
from tqdm import tqdm

from ..config import (
    DAYS_PER_TIMESTEP,
    EE_BUCKET_TIFS,
    EE_DRIVE_FOLDER_NAME,
    EE_FOLDER_TIFS,
    EE_PROJECT,
    END_YEAR,
    EXPORTED_HEIGHT_WIDTH_METRES,
    NO_DATA_VALUE,
    NUM_TIMESTEPS,
    SEASONS,
    START_YEAR,
    TIFS_FOLDER,
)
from .ee_bbox import EEBoundingBox
from .era5 import ERA5_BANDS, ERA5_DIV_VALUES, ERA5_SHIFT_VALUES, get_single_era5_image
from .modis import MODIS_BANDS, MODIS_DIV_VALUES, MODIS_SHIFT_VALUES, get_single_modis_image
from .s1 import S1_BANDS, S1_DIV_VALUES, S1_SHIFT_VALUES, get_single_s1_image
from .s2 import S2_BANDS, S2_DIV_VALUES, S2_SHIFT_VALUES, get_single_s2_image
from .s3 import S3_BANDS, S3_DIV_VALUES, S3_SHIFT_VALUES, get_single_s3_image
from .srtm import SRTM_BANDS, SRTM_DIV_VALUES, SRTM_SHIFT_VALUES, get_single_srtm_image
from .utils import get_ee_credentials, sample_time_window, sample_season_year
from .viirs import (
    VIIRS_500m_DIV_VALUES,
    VIIRS_500m_SHIFT_VALUES,
    VIIRS_1000m_DIV_VALUES,
    VIIRS_1000m_SHIFT_VALUES,
    VIIRS_BANDS_500m,
    VIIRS_BANDS_1000m,
    get_single_viirs_500m_image,
    get_single_viirs_1000m_image,
)

# dataframe constants when exporting the labels
LAT = "Latitude"
LON = "Longitude"
START_DATE = date(START_YEAR, 1, 1)
END_DATE = date(END_YEAR, 12, 31)

TIME_IMAGE_FUNCTIONS = [
    get_single_s1_image,
    get_single_s2_image,
    get_single_s3_image,
    get_single_era5_image,
    get_single_modis_image,
    get_single_viirs_500m_image,
    get_single_viirs_1000m_image,
]

SPACE_TIME_HIGH_RES_BANDS = S1_BANDS + S2_BANDS
SPACE_TIME_HIGH_RES_SHIFT_VALUES = S1_SHIFT_VALUES + S2_SHIFT_VALUES
SPACE_TIME_HIGH_RES_DIV_VALUES = S1_DIV_VALUES + S2_DIV_VALUES

SPACE_TIME_MED_RES_BANDS = S3_BANDS
SPACE_TIME_MED_RES_SHIFT_VALUES = S3_SHIFT_VALUES
SPACE_TIME_MED_RES_DIV_VALUES = S3_DIV_VALUES

SPACE_TIME_LOW_RES_BANDS = MODIS_BANDS + VIIRS_BANDS_500m
SPACE_TIME_LOW_RES_SHIFT_VALUES = MODIS_SHIFT_VALUES + VIIRS_500m_SHIFT_VALUES
SPACE_TIME_LOW_RES_DIV_VALUES = MODIS_DIV_VALUES + VIIRS_500m_DIV_VALUES

SPACE_TIME_BANDS = SPACE_TIME_HIGH_RES_BANDS + SPACE_TIME_MED_RES_BANDS + SPACE_TIME_LOW_RES_BANDS
SPACE_TIME_SHIFT_VALUES = np.array(
    SPACE_TIME_HIGH_RES_SHIFT_VALUES
    + SPACE_TIME_MED_RES_SHIFT_VALUES
    + SPACE_TIME_LOW_RES_SHIFT_VALUES
)
SPACE_TIME_DIV_VALUES = np.array(
    SPACE_TIME_HIGH_RES_DIV_VALUES + SPACE_TIME_MED_RES_DIV_VALUES + SPACE_TIME_LOW_RES_DIV_VALUES
)

TIME_BANDS = ERA5_BANDS + VIIRS_BANDS_1000m
TIME_SHIFT_VALUES = np.array(ERA5_SHIFT_VALUES + VIIRS_1000m_SHIFT_VALUES)
TIME_DIV_VALUES = np.array(ERA5_DIV_VALUES + VIIRS_1000m_DIV_VALUES)

ALL_DYNAMIC_IN_TIME_BANDS = (
    SPACE_TIME_HIGH_RES_BANDS + SPACE_TIME_MED_RES_BANDS + SPACE_TIME_LOW_RES_BANDS + TIME_BANDS
)

SPACE_BANDS = SRTM_BANDS
SPACE_IMAGE_FUNCTIONS = [get_single_srtm_image]
SPACE_SHIFT_VALUES = np.array(SRTM_SHIFT_VALUES)
SPACE_DIV_VALUES = np.array(SRTM_DIV_VALUES)

# we will add latlons in dataset.py function
LOCATION_BANDS = ["x", "y", "z"]
STATIC_BANDS = LOCATION_BANDS
STATIC_SHIFT_VALUES = np.array([0, 0, 0])
STATIC_DIV_VALUES = np.array([1, 1, 1])


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
                image_function(region=polygon, start_date=cur_date, end_date=cur_end_date)
            )

        image_collection_list.append(ee.Image.cat(image_list))
        cur_date += timedelta(days=days_per_timestep)
        cur_end_date += timedelta(days=days_per_timestep)

    # now, we want to take our image collection and append the bands into a single image
    imcoll = ee.ImageCollection(image_collection_list)
    combine_bands_function = make_combine_bands_function(ALL_DYNAMIC_IN_TIME_BANDS)
    img = ee.Image(imcoll.iterate(combine_bands_function))

    # finally, we add the static in time images
    total_image_list: List[ee.Image] = [img]
    for space_image_function in SPACE_IMAGE_FUNCTIONS:
        total_image_list.append(
            space_image_function(
                region=polygon,
                start_date=interval_start_date - timedelta(days=31),
                end_date=interval_end_date + timedelta(days=31),
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
        dest_bucket: str = EE_BUCKET_TIFS,
        dest_drive_folder: str = EE_DRIVE_FOLDER_NAME,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
        mode: str = "drive",
        no_data_val: int = NO_DATA_VALUE,
    ) -> None:
        assert mode in ["cloud", "drive", "url"]
        self.mode = mode
        if mode == "url":
            print(f"Mode: url. Files will be saved to {TIFS_FOLDER} and rsynced to google cloud")
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
        self.local_tif_list = [x.name for x in TIFS_FOLDER.glob("*.tif*")]
        self.no_data_val = no_data_val

    def sync_local_and_gcloud(self):
        os.system(f"gcloud storage rsync -r {TIFS_FOLDER} gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS}")

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

        # Description of the export cannot contain certrain characters
        description = ee_safe_str(cloud_filename)

        if f"{cloud_filename}.tif" in self.cloud_tif_list:
            # checks that we haven't already exported this file
            print(f"{cloud_filename}.tif already in cloud_tif_files")
            return False

        if local_filename in self.local_tif_list:
            # checks that we haven't already exported this file
            print(f"{local_filename} already in local_tif_files, but not in the cloud")
            return False

        # Check if task is already started in EarthEngine
        if description in self.ee_task_list:
            print(f"{description} already in ee task list")
            return False

        if len(self.ee_task_list) >= 3000:
            # we can only have 3000 running exports at once
            print("3000 exports started")
            return False

        img = create_ee_image(polygon, interval_start_date, interval_end_date)

        # TODO: check if we can use the scale parameter of should use crs and crs_transform instead
        if self.mode == "cloud":
            try:
                ee.batch.Export.image.toCloudStorage(
                    bucket=self.dest_bucket,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    scale=10,
                    region=polygon,
                    maxPixels=1e13,
                    fileDimensions=file_dimensions,
                    formatOptions={"noData": self.no_data_val},
                ).start()
                self.ee_task_list.append(description)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}")
                return False
        elif self.mode == "drive":
            try:
                ee.batch.Export.image.toDrive(
                    folder=self.dest_drive_folder,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    scale=10,
                    region=polygon,
                    maxPixels=1e13,
                    fileDimensions=file_dimensions,
                    formatOptions={"noData": self.no_data_val},
                ).start()
                self.ee_task_list.append(description)
            except ee.ee_exception.EEException as e:
                print(f"Task not started! Got exception {e}")
                return False
        elif self.mode == "url":
            try:
                url = img.getDownloadURL(
                    {
                        "region": polygon,
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
                local_path = Path(TIFS_FOLDER / local_filename)
                with local_path.open("wb") as f:
                    shutil.copyfileobj(r.raw, f)
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

        for _, row in tqdm(latlons.iterrows(), desc="Exporting", total=len(latlons)):
            ee_bbox = EEBoundingBox.from_centre(
                # worldstrat points are strings
                mid_lat=float(row[LAT]),
                mid_lon=float(row[LON]),
                surrounding_metres=int(self.surrounding_metres),
            )

            # Sample each point for each season
            for season in SEASONS.items():
                # randomly choose year to sample from
                season = sample_season_year(season, START_YEAR, END_YEAR)

                SEASON_START_DATE = season[0]
                SEASON_END_DATE = season[1]

                WINDOW_START_DATE, WINDOW_END_DATE = sample_time_window(
                    SEASON_START_DATE, SEASON_END_DATE, NUM_TIMESTEPS
                )

                export_started = self._export_for_polygon(
                    polygon=ee_bbox.to_ee_polygon(),
                    polygon_identifier=ee_bbox.get_identifier(WINDOW_START_DATE, WINDOW_END_DATE),
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
