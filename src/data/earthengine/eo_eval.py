### Original Code:
### Copyright (c) 2024 Presto Authors
### Licensed under the MIT License.
### A copy of the MIT License is available in the LICENSE file in the root directory of this project.

# Similar to eo.py, but for evaluation.
# Adds functions to export input data from given label polygons and in crs other than EPSG:4326,
# which is needed for evaluation on the test set (since the test set labels are in a different crs).
# We can also export data in various ways from CSV files, which is useful for inference.
# Still, a lot of this script is redundant with eo.py and needs to be cleaned --> refactor in the future.

# https://github.com/nasaharvest/openmapflow/blob/main/openmapflow/ee_exporter.py
import os
import shutil
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Union
from typing import OrderedDict as OrderedDictType

import ee
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import requests  # type: ignore

from src.data.config import (
    DATA_FOLDER,
    EE_BUCKET_TIFS,
    EE_DRIVE_FOLDER_NAME,
    MODALITIES,
    NO_DATA_VALUE,
    NUM_TIMESTEPS,
)
from src.data.earthengine.copernicus_dem import (
    DEM_BANDS,
    DEM_DIV_VALUES,
    DEM_SHIFT_VALUES,
    get_single_dem_image,
)
from src.data.earthengine.ee_bbox import EEBoundingBox, EEGeometry
from src.data.earthengine.eo import (
    EarthEngineExporter,
    create_ee_image,
    ee_safe_str,
)
from src.data.earthengine.era5 import (
    ERA5_BANDS,
    ERA5_DIV_VALUES,
    ERA5_SHIFT_VALUES,
    get_single_era5_image,
)
from src.data.earthengine.esa_worldcover import (
    EE_WC_BANDS,
    EE_WC_DIV_VALUES,
    EE_WC_SHIFT_VALUES,
    WC_BANDS_NAMES,
    get_single_ee_wc_image,
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

EE_SPACE_BANDS = []
SPACE_SHIFT_VALUES = []
SPACE_DIV_VALUES = []

CLOUD_BANDS = []

for modality in MODALITIES:
    if MODALITIES[modality].get("active") and MODALITIES[modality].get("export"):
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
                EE_SPACE_BANDS.extend(band_list)
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

# NOTE: This changes once the input sources are modified
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
assert SPACE_IMAGE_FUNCTIONS == [get_single_dem_image, get_single_ee_wc_image]
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
assert EE_SPACE_BANDS == DEM_BANDS + EE_WC_BANDS
assert SPACE_SHIFT_VALUES == DEM_SHIFT_VALUES + EE_WC_SHIFT_VALUES
assert SPACE_DIV_VALUES == DEM_DIV_VALUES + EE_WC_DIV_VALUES
assert CLOUD_BANDS == MODIS_CLOUD_FLAG_BANDS + S2_CLOUD_FLAG_BANDS + LANDSAT_CLOUD_FLAG_BANDS

SPACE_TIME_HIGH_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_HIGH_RES_SHIFT_VALUES)
SPACE_TIME_HIGH_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_HIGH_RES_DIV_VALUES)
SPACE_TIME_MED_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_MED_RES_SHIFT_VALUES)
SPACE_TIME_MED_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_MED_RES_DIV_VALUES)
SPACE_TIME_LOW_RES_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_LOW_RES_SHIFT_VALUES)
SPACE_TIME_LOW_RES_DIV_VALUES_NP: npt.NDArray[Any] = np.array(SPACE_TIME_LOW_RES_DIV_VALUES)
TIME_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(TIME_SHIFT_VALUES)
TIME_DIV_VALUES_NP: npt.NDArray[Any] = np.array(TIME_DIV_VALUES)
SPACE_SHIFT_VALUES_NP: npt.NDArray[Any] = np.array(DEM_SHIFT_VALUES + EE_WC_SHIFT_VALUES)
SPACE_DIV_VALUES_NP: npt.NDArray[Any] = np.array(DEM_DIV_VALUES + EE_WC_DIV_VALUES)

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

# we create a new list for one-hot encoded space bands
SPACE_BANDS = DEM_BANDS + WC_BANDS_NAMES

# hacky, but we need to reduce one shift/div value because we already had one for the "Map" band
SPACE_SHIFT_VALUES_NP = np.append(SPACE_SHIFT_VALUES_NP, [0] * (len(WC_BANDS_NAMES) - 1))
SPACE_DIV_VALUES_NP = np.append(SPACE_DIV_VALUES_NP, [1] * (len(WC_BANDS_NAMES) - 1))

# index of the ESA Worldcover band in the SPACE_BANDS list, needed for one-hot encoding
ESA_WORLDCOVER_BAND_INDEX = EE_SPACE_BANDS.index("Map")

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
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS_NAMES],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
    }
)


class EarthEngineExporterEval(EarthEngineExporter):
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
        super().__init__(
            dest_bucket=dest_bucket,
            dest_drive_folder=dest_drive_folder,
            check_ee=check_ee,
            check_gcp=check_gcp,
            credentials=credentials,
            mode=mode,
            no_data_val=no_data_val,
            tifs_folder=tifs_folder,
        )

    def _export_for_polygon(
        self,
        polygon: ee.Geometry,
        polygon_identifier: Union[int, str],
        interval_start_date: date,
        interval_end_date: date,
        file_dimensions: Optional[int] = None,
        crs: Optional[str] = "EPSG:4326",
    ) -> bool:
        cloud_filename = f"{str(polygon_identifier)}"
        local_filename = f"{str(polygon_identifier).replace('/', '_')}_{crs}.tif"

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

        if polygon_identifier in self.local_location_season_tif_list:
            # checks that we haven't already exported this file
            print(f"{polygon_identifier} already in local_tif_files", flush=True)
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

        # important so we control the no data value
        # NOTE: in reality, GEE might still write values to zero with URL downloads
        img = img.unmask(self.no_data_val)  # type: ignore[attr-defined]

        print("Exporting image in crs", crs, flush=True)

        if self.mode == "cloud":
            try:
                ee.batch.Export.image.toCloudStorage(
                    bucket=self.dest_bucket,
                    fileNamePrefix=cloud_filename,
                    image=img.clip(polygon),
                    description=description,
                    crs=crs,
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
                    image=img.toDouble().clip(polygon),
                    description=description,
                    crs=crs,
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
                        "crs": crs,
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

    def export_from_filename_for_folder(
        self,
        folder,
        start_idx: int = 0,
    ) -> None:
        """
        Export boxes with the bounds of the files given in the current folder.
        NOTE: The resulting exports will not be exactly in rectangular format, so we will have to crop them afterwards.
        """
        # check that each file in the folder has a filename with the format L0*_YYYYMMDD_LAT_LON_SC[a number between 0 and 100]
        # and that the lat and lon are in the format of a string
        # e.g. LC09_20220101_FSC0_50.1234_8.1234.tif
        # also, create a pandas dataframe with all filenames in the format of a string
        filenames = []
        folder = Path(DATA_FOLDER / folder)

        # NOTE: specific to the filename format of our finetuning / evaluation data
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
        exports_started = 0
        print(f"Exporting {len(filenames)} latlons: ")

        for filename in filenames:
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

            ee_bbox = EEGeometry.from_coord_bounds(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                proj="EPSG:4326",
            )

            WINDOW_END_DATE = datetime.strptime(parts[1], "%Y%m%d").date()
            WINDOW_START_DATE = WINDOW_END_DATE - timedelta(days=NUM_TIMESTEPS - 1)

            export_started = self._export_for_polygon(
                polygon=ee_bbox,
                polygon_identifier=filename,
                interval_start_date=WINDOW_START_DATE,
                interval_end_date=WINDOW_END_DATE,
                crs=crs,
            )
            if export_started:
                exports_started += 1

        if self.mode == "url":
            print("Export finished.")

    def export_from_csv_wgs84(self, csv_file) -> None:
        """Export from center coordinates and dates passed by a csv file."""
        df = pd.read_csv(csv_file)
        dates = df["date"].tolist()
        lats = df["latitude"].tolist()
        lons = df["longitude"].tolist()

        exports_started = 0
        print(f"Exporting {len(dates)} files: ")

        for i, dat in enumerate(dates):
            ee_bbox = EEBoundingBox.from_centre(
                # worldstrat points are strings
                mid_lat=float(lats[i]),
                mid_lon=float(lons[i]),
                surrounding_metres=int(self.surrounding_metres),
            )

            WINDOW_END_DATE = datetime.strptime(str(dat), "%Y%m%d").date()
            WINDOW_START_DATE = WINDOW_END_DATE - timedelta(days=NUM_TIMESTEPS - 1)

            export_started = self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=ee_bbox.get_identifier(
                    WINDOW_START_DATE, WINDOW_END_DATE, for_eval_from_csv=True
                ),
                interval_start_date=WINDOW_START_DATE,
                interval_end_date=WINDOW_END_DATE,
            )
            if export_started:
                exports_started += 1

        if self.mode == "url":
            print("Export finished. Syncing to google cloud")
            self.sync_local_and_gcloud()
            print("Finished sync")

    def export_from_csv_utm(self, csv_file) -> None:
        """Export from UTM bounds and dates passed by a csv file."""
        df = pd.read_csv(csv_file)
        dates = df["date"].tolist()
        coordinate_system = df["crs"].tolist()
        center_x = df["center_lat"].tolist()
        center_y = df["center_lon"].tolist()
        min_x = df["min_x"].tolist()
        max_x = df["max_x"].tolist()
        min_y = df["min_y"].tolist()
        max_y = df["max_y"].tolist()

        exports_started = 0
        print(f"Exporting {len(dates)} files: ")

        for i, dat in enumerate(dates):
            min_yy, max_yy = min_y[i], max_y[i]
            min_xx, max_xx = min_x[i], max_x[i]
            crs = coordinate_system[i]

            # reproject to EPSG:4326
            print(f"Converting {crs} to EPSG:4326")
            from pyproj import Transformer

            filename = f"PR_{dat}_{center_x[i]:.16f}_{center_y[i]:.16f}.tif"

            # NOTE: always_xy=True ensures that the first coordinate is always in northerly direction
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(min_xx, min_yy)
            max_lon, max_lat = transformer.transform(max_xx, max_yy)

            ee_bbox = EEGeometry.from_coord_bounds(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                proj="EPSG:4326",
            )

            WINDOW_END_DATE = datetime.strptime(str(dat), "%Y%m%d").date()
            WINDOW_START_DATE = WINDOW_END_DATE - timedelta(days=NUM_TIMESTEPS - 1)

            export_started = self._export_for_polygon(
                polygon=ee_bbox,
                polygon_identifier=filename,
                interval_start_date=WINDOW_START_DATE,
                interval_end_date=WINDOW_END_DATE,
                crs=crs,
            )
            if export_started:
                exports_started += 1

        if self.mode == "url":
            print("Export finished.")
