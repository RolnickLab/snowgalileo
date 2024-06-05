# https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/ee_exporter.py
import json
import os
from datetime import date, timedelta
from typing import List, Optional, Union

import ee
import numpy as np
import pandas as pd
from pandas.compat._optional import import_optional_dependency
from tqdm import tqdm

from ..config import (
    DAYS_PER_TIMESTEP,
    EE_BUCKET_TIFS,
    EE_FOLDER_TIFS,
    EE_PROJECT,
    END_YEAR,
    EXPORTED_HEIGHT_WIDTH_METRES,
    START_YEAR,
)
from .dynamic_world import (
    DW_BANDS,
    DW_DIV_VALUES,
    DW_SHIFT_VALUES,
    get_single_dw_image,
)
from .ee_bbox import EEBoundingBox
from .era5 import ERA5_BANDS, ERA5_DIV_VALUES, ERA5_SHIFT_VALUES, get_single_era5_image
from .s1 import (
    S1_BANDS,
    S1_DIV_VALUES,
    S1_SHIFT_VALUES,
    get_s1_image_collection,
    get_single_s1_image,
)
from .s2 import S2_BANDS, S2_DIV_VALUES, S2_SHIFT_VALUES, get_single_s2_image
from .srtm import SRTM_BANDS, SRTM_DIV_VALUES, SRTM_SHIFT_VALUES, get_single_srtm_image

# dataframe constants when exporting the labels
LAT = "lat"
LON = "lon"
SURROUNDING_METRES = EXPORTED_HEIGHT_WIDTH_METRES / 2
START_DATE = date(START_YEAR, 1, 1)
END_DATE = date(END_YEAR, 12, 31)

TIME_IMAGE_FUNCTIONS = [
    get_single_s2_image,
    get_single_era5_image,
]
SPACE_TIME_BANDS = S1_BANDS + S2_BANDS
SPACE_TIME_SHIFT_VALUES = np.array(S1_SHIFT_VALUES + S2_SHIFT_VALUES)
SPACE_TIME_DIV_VALUES = np.array(S1_DIV_VALUES + S2_DIV_VALUES)

TIME_BANDS = ERA5_BANDS
TIME_SHIFT_VALUES = np.array(ERA5_SHIFT_VALUES)
TIME_DIV_VALUES = np.array(ERA5_DIV_VALUES)

ALL_DYNAMIC_IN_TIME_BANDS = SPACE_TIME_BANDS + TIME_BANDS

SPACE_BANDS = SRTM_BANDS + DW_BANDS
SPACE_IMAGE_FUNCTIONS = [get_single_srtm_image, get_single_dw_image]
SPACE_SHIFT_VALUES = np.array(SRTM_SHIFT_VALUES + DW_SHIFT_VALUES)
SPACE_DIV_VALUES = np.array(SRTM_DIV_VALUES + DW_DIV_VALUES)


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
    start_date: date,
    end_date: date,
    days_per_timestep: int = DAYS_PER_TIMESTEP,
) -> ee.Image:
    """
    Returns an ee.Image which we can then export.
    This image will contain S1, S2, ERA5 and Dynamic World data
    between start_date and end_date, in intervals of
    days_per_timestep. Each timestep will be a different channel in the
    image (e.g. if I have 3 timesteps, then I'll have VV, VV_1, VV_2 for the
    S1 VV bands). The static in time SRTM bands will also be in the image.
    """
    image_collection_list: List[ee.Image] = []
    cur_date = start_date
    cur_end_date = cur_date + timedelta(days=days_per_timestep)

    # We get all the S1 images in an exaggerated date range. We do this because
    # S1 data is sparser, so we will pull from outside the days_per_timestep
    # range if we are missing data within that range
    vv_imcol, vh_imcol = get_s1_image_collection(
        polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
    )

    while cur_end_date <= end_date:
        image_list: List[ee.Image] = []

        # first, the S1 image which gets the entire s1 collection
        image_list.append(
            get_single_s1_image(
                region=polygon,
                start_date=cur_date,
                end_date=cur_end_date,
                vv_imcol=vv_imcol,
                vh_imcol=vh_imcol,
            )
        )
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
    for static_image_function in SPACE_IMAGE_FUNCTIONS:
        total_image_list.append(
            static_image_function(
                region=polygon,
                start_date=start_date - timedelta(days=31),
                end_date=end_date + timedelta(days=31),
            )
        )

    return ee.Image.cat(total_image_list)


def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Logging into EarthEngine with {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Logging into EarthEngine with default credentials")
        return "persistent"


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
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
    ) -> None:
        self.dest_bucket = dest_bucket
        ee.Initialize(
            credentials=credentials if credentials else get_ee_credentials(), project=EE_PROJECT
        )
        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []
        self.check_gcp = check_gcp
        self.cloud_tif_list = get_cloud_tif_list(dest_bucket) if self.check_gcp else []

    def _export_for_polygon(
        self,
        polygon: ee.Geometry,
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        file_dimensions: Optional[int] = None,
    ) -> bool:
        filename = f"{EE_FOLDER_TIFS}/{str(polygon_identifier)}"

        # Description of the export cannot contain certrain characters
        description = ee_safe_str(filename)

        if f"{filename}.tif" in self.cloud_tif_list:
            # checks that we haven't already exported this file
            print(f"{filename}.tif already in cloud_tif_files")
            return False

        # Check if task is already started in EarthEngine
        if description in self.ee_task_list:
            print(f"{description} already in ee task list")
            return False

        if len(self.ee_task_list) >= 3000:
            # we can only have 3000 running exports at once
            print("3000 exports started")
            return False

        img = create_ee_image(polygon, start_date, end_date)

        try:
            ee.batch.Export.image.toCloudStorage(
                bucket=self.dest_bucket,
                fileNamePrefix=filename,
                image=img.clip(polygon),
                description=description,
                scale=10,
                region=polygon,
                maxPixels=1e13,
                fileDimensions=file_dimensions,
            ).start()
            self.ee_task_list.append(description)
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")

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
                surrounding_metres=int(SURROUNDING_METRES),
            )

            export_started = self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=ee_bbox.get_identifier(START_DATE, END_DATE),
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if export_started:
                exports_started += 1
                if num_exports_to_start is not None and exports_started >= num_exports_to_start:
                    print(f"Started {exports_started} exports. Ending export")
                    return None
