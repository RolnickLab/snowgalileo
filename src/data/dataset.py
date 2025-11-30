import json
import logging
import math
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from random import sample
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, cast

import h5py
import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange, repeat
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from src.data.config import (
    CHANNEL_WISE_INVALID_DATA_THRESHOLDS,
    DATASET_OUTPUT_HW_HIGH_RES,
    DATASET_OUTPUT_HW_LOW_RES,
    DATASET_OUTPUT_HW_MED_RES,
    EE_BUCKET_TIFS,
    EE_DRIVE_FOLDER_ID,
    EE_FOLDER_H5PYS,
    EE_FOLDER_TIFS,
    MODALITIES,
    MODIS_FILL_VALUE,
    NDSI_VALID_DATA_BOUNDS,
    NDVI_VALID_DATA_BOUNDS,
    NO_DATA_VALUE,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_TIMESTEPS,
    TIFS_FOLDER,
)
from src.data.earthengine.eo import (
    CLOUD_BANDS,
    DEM_BANDS,
    EE_SPACE_BANDS,
    EE_WC_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    EO_SPACE_TIME_LOW_RES_BANDS,
    ESA_WORLDCOVER_BAND_INDEX,
    SPACE_BANDS,
    SPACE_DIV_VALUES_NP,
    SPACE_SHIFT_VALUES_NP,
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_HIGH_RES_DIV_VALUES_NP,
    SPACE_TIME_HIGH_RES_SHIFT_VALUES_NP,
    SPACE_TIME_LOW_RES_BANDS,
    SPACE_TIME_LOW_RES_DIV_VALUES_NP,
    SPACE_TIME_LOW_RES_SHIFT_VALUES_NP,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_MED_RES_DIV_VALUES_NP,
    SPACE_TIME_MED_RES_SHIFT_VALUES_NP,
    STATIC_BANDS,
    STATIC_DIV_VALUES_NP,
    STATIC_SHIFT_VALUES_NP,
    TIME_BANDS,
    TIME_DIV_VALUES_NP,
    TIME_SHIFT_VALUES_NP,
)
from src.data.earthengine.esa_worldcover import NUM_WC_CLASSES, WC_CLASS_VALUES
from src.data.utils import RunningStats

logger = logging.getLogger("__main__")


class Normalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    # using the pre-training population statistics
    # for NDVI, NDSI, ESA Worldcover, and Location bands, we use pre-defined values
    std_bands: Dict[str, list] = {
        "space_time_high_res": SPACE_TIME_HIGH_RES_BANDS,
        "space_time_med_res": SPACE_TIME_MED_RES_BANDS,
        "space_time_low_res": [b for b in SPACE_TIME_LOW_RES_BANDS if b != "NDVI" and b != "NDSI"],
        "space": DEM_BANDS,
        "time": TIME_BANDS,
        "static": [],
    }

    def __init__(self, std: bool = True, normalizing_dicts: Optional[Dict] = None):
        self.shift_div_dict: Dict[str, Dict[str, np.ndarray]] = {
            "space_time_high_res": {
                "shift": deepcopy(SPACE_TIME_HIGH_RES_SHIFT_VALUES_NP),
                "div": deepcopy(SPACE_TIME_HIGH_RES_DIV_VALUES_NP),
            },
            "space_time_med_res": {
                "shift": deepcopy(SPACE_TIME_MED_RES_SHIFT_VALUES_NP),
                "div": deepcopy(SPACE_TIME_MED_RES_DIV_VALUES_NP),
            },
            "space_time_low_res": {
                "shift": deepcopy(SPACE_TIME_LOW_RES_SHIFT_VALUES_NP),
                "div": deepcopy(SPACE_TIME_LOW_RES_DIV_VALUES_NP),
            },
            "space": {
                "shift": deepcopy(SPACE_SHIFT_VALUES_NP),
                "div": deepcopy(SPACE_DIV_VALUES_NP),
            },
            "time": {
                "shift": deepcopy(TIME_SHIFT_VALUES_NP),
                "div": deepcopy(TIME_DIV_VALUES_NP),
            },
            "static": {
                "shift": deepcopy(STATIC_SHIFT_VALUES_NP),
                "div": deepcopy(STATIC_DIV_VALUES_NP),
            },
        }

        self.normalizing_dicts = normalizing_dicts
        if std:
            name_to_bands = {
                "space_time_high_res": SPACE_TIME_HIGH_RES_BANDS,
                "space_time_med_res": SPACE_TIME_MED_RES_BANDS,
                "space_time_low_res": SPACE_TIME_LOW_RES_BANDS,
                "space": DEM_BANDS,
                "time": TIME_BANDS,
                "static": STATIC_BANDS,
            }
            assert normalizing_dicts is not None
            for key, val in normalizing_dicts.items():
                if key == "total_n" or key == "sampled_n":
                    continue
                bands_to_replace = self.std_bands[key]
                for band in bands_to_replace:
                    band_idx = name_to_bands[key].index(band)
                    mean = val["mean"][band_idx]
                    std = val["std"][band_idx]
                    min_value = mean - (2 * std)
                    max_value = mean + (2 * std)
                    div = max_value - min_value
                    if div == 0:
                        raise ValueError(f"{band} has div value of 0")
                    self.shift_div_dict[key]["shift"][band_idx] = min_value
                    self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(
        x: np.ndarray,
        valid_data_mask: np.ndarray,
        shift_values: np.ndarray,
        div_values: np.ndarray,
    ) -> np.ndarray:
        # we don't want to normalize the no data values to be able to identify them later
        assert np.all(x[valid_data_mask] != NO_DATA_VALUE)
        x_normalized = np.where(valid_data_mask, (x - shift_values) / div_values, NO_DATA_VALUE)

        return x_normalized

    def __call__(self, x: np.ndarray, array_type: str, valid_data_mask: np.ndarray):
        if self.normalizing_dicts is not None:
            if array_type not in self.normalizing_dicts:
                raise ValueError(f"Unknown array type: {array_type}")
            return self._normalize(
                x,
                valid_data_mask,
                self.shift_div_dict[array_type]["shift"],
                self.shift_div_dict[array_type]["div"],
            )
        else:
            raise NotImplementedError(
                "Only normalization with precomputed mean/std is implemented."
            )


class StackedDatasetOutput(NamedTuple):
    space_time_high_res_x: np.ndarray
    space_time_med_res_x: np.ndarray
    space_time_low_res_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    months: np.ndarray


class DatasetOutput(NamedTuple):
    space_time_high_res_x: np.ndarray
    space_time_med_res_x: np.ndarray
    space_time_low_res_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    months: np.ndarray
    valid_data_mask_space_time_high_res: np.ndarray
    valid_data_mask_space_time_med_res: np.ndarray
    valid_data_mask_space_time_low_res: np.ndarray
    valid_data_mask_space: np.ndarray
    valid_data_mask_time: np.ndarray
    valid_data_mask_static: np.ndarray

    def normalize(self, normalizer: Optional[Normalizer]) -> "DatasetOutput":
        if normalizer is None:
            return self
        return DatasetOutput(
            normalizer(
                self.space_time_high_res_x,
                "space_time_high_res",
                self.valid_data_mask_space_time_high_res,
            ).astype(np.half),
            normalizer(
                self.space_time_med_res_x,
                "space_time_med_res",
                self.valid_data_mask_space_time_med_res,
            ).astype(np.half),
            normalizer(
                self.space_time_low_res_x,
                "space_time_low_res",
                self.valid_data_mask_space_time_low_res,
            ).astype(np.half),
            normalizer(self.space_x, "space", self.valid_data_mask_space).astype(np.half),
            normalizer(self.time_x, "time", self.valid_data_mask_time).astype(np.half),
            normalizer(self.static_x, "static", self.valid_data_mask_static).astype(np.half),
            self.months,
            self.valid_data_mask_space_time_high_res,
            self.valid_data_mask_space_time_med_res,
            self.valid_data_mask_space_time_low_res,
            self.valid_data_mask_space,
            self.valid_data_mask_time,
            self.valid_data_mask_static,
        )


class ListOfDatasetOutputs(NamedTuple):
    space_time_high_res_x: List[np.ndarray]
    space_time_med_res_x: List[np.ndarray]
    space_time_low_res_x: List[np.ndarray]
    space_x: List[np.ndarray]
    time_x: List[np.ndarray]
    static_x: List[np.ndarray]
    months: List[np.ndarray]

    def to_datasetoutput(self) -> StackedDatasetOutput:
        return StackedDatasetOutput(
            np.stack(self.space_time_high_res_x, axis=0),
            np.stack(self.space_time_med_res_x, axis=0),
            np.stack(self.space_time_low_res_x, axis=0),
            np.stack(self.space_x, axis=0),
            np.stack(self.time_x, axis=0),
            np.stack(self.static_x, axis=0),
            np.stack(self.months, axis=0),
        )


def to_cartesian(
    lat: Union[float, np.ndarray], lon: Union[float, np.ndarray]
) -> Union[np.ndarray]:
    if isinstance(lat, float):
        assert -90 <= lat <= 90, f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert -180 <= lon <= 180, f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")


class Dataset(PyTorchDataset):
    def __init__(
        self,
        data_folder: Path,
        download: bool = True,
        h5py_folder: Optional[Path] = None,
        h5pys_only: bool = False,
        output_hw_high_res: int = DATASET_OUTPUT_HW_HIGH_RES,
        output_hw_med_res: int = DATASET_OUTPUT_HW_MED_RES,
        output_hw_low_res: int = DATASET_OUTPUT_HW_LOW_RES,
        output_timesteps: int = NUM_TIMESTEPS,
        normalizer: Optional[Normalizer] = None,
    ):
        self.data_folder = data_folder
        self.h5pys_only = h5pys_only
        self.h5py_folder = h5py_folder
        self.cache = False
        self.normalizer = normalizer

        if h5py_folder is not None:
            self.cache = True
        if h5pys_only:
            assert h5py_folder is not None, "Can't use h5pys only if there is no cache folder"
            self.tifs: List[Path] = []
            if download:
                self.download_h5pys(h5py_folder)
            self.h5pys = list(h5py_folder.glob("*.h5"))
        else:
            if download:
                self.download_tifs_from_drive_folder()
            self.tifs = []
            tifs = list(data_folder.glob("*.tif")) + list(data_folder.glob("*.tiff"))
            for tif in tifs:
                try:
                    _ = self.start_month_from_file(tif)
                    self.tifs.append(tif)
                except IndexError:
                    warnings.warn(f"IndexError for {tif}")
            self.h5pys = []

        self.output_hw_high_res = output_hw_high_res
        self.output_hw_med_res = output_hw_med_res
        self.output_hw_low_res = output_hw_low_res
        self.output_timesteps = output_timesteps

    def __len__(self) -> int:
        if self.h5pys_only:
            return len(self.h5pys)
        return len(self.tifs)

    @staticmethod
    def download_tifs_from_drive_folder():
        """
        Downloads all filed from a folder in Google Drive. Drive folder ID and destination folder are defined in the config file.
        Modified from: https://developers.google.com/drive/api/guides/manage-downloads
        """

        SERVICE_ACCOUNT_FILE = Path(__file__).parents[2] / "ee-marlena-credentials.json"
        SCOPES = ["https://www.googleapis.com/auth/drive"]

        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)

        os.makedirs(TIFS_FOLDER, exist_ok=True)

        # create drive api client
        service = build("drive", "v3", credentials=creds)

        # page token to get all files
        page_token = None
        items = []
        while True:
            # List files in the folder
            results = (
                service.files()
                .list(
                    q=f"'{EE_DRIVE_FOLDER_ID}' in parents",
                    fields="nextPageToken, files(id, name)",
                    pageToken=page_token,
                )
                .execute()
            )
            items.extend(results.get("files", []))
            page_token = results.get("nextPageToken", None)
            if page_token is None:
                break

        if not items:
            print("No files found in the folder.")
        else:
            try:
                for item in items:
                    print(f"Downloading {item['name']}...")
                    request = service.files().get_media(fileId=item["id"])
                    filename = item["name"]

                    # Define the full path for the local file
                    local_file_path = os.path.join(TIFS_FOLDER, filename)

                    # Save the file
                    with open(local_file_path, "wb") as fh:
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                            print(f"Download {int(status.progress() * 100)}% complete.")
            except HttpError as e:
                print(f"HttpError: {e}")
                print("Response content:", e.content)

    @staticmethod
    def download_tifs_from_cloud(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS} {data_folder}")

    @staticmethod
    def download_h5pys(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_H5PYS} {data_folder}")

    @staticmethod
    def return_subset_indices(
        total_h,
        total_w,
        total_t,
        size: int,
        num_timesteps: int,
    ) -> Tuple[int, int, int]:
        """
        space_time_high_res_x: array of shape [H, W, T, D]
        space_time_med_res_x: array of shape [H, W, T, D]
        space_time_low_res_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        possible_h = total_h - size
        possible_w = total_w - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = total_t - num_timesteps
        assert possible_t >= 0

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        if possible_t > 0:
            start_t = np.random.choice(possible_t)
        else:
            start_t = possible_t

        return start_h, start_w, start_t

    @staticmethod
    def subset_image(
        space_time_high_res_x: np.ndarray,
        space_time_med_res_x: np.ndarray,
        space_time_low_res_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Crops the exported image into a size that can be processed by the model.
        Exported image size: can be larger that exported area / scale
        Size that can be processed by the model: max(patch_size * number_of_patches_per_dim)

        space_time_high_res_x: array of shape [H, W, T, D]
        space_time_med_res_x: array of shape [H, W, T, D]
        space_time_low_res_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        assert (space_time_high_res_x.shape[0] == space_x.shape[0]) & (
            space_time_high_res_x.shape[1] == space_x.shape[1]
        )
        assert space_time_high_res_x.shape[2] == time_x.shape[0]
        possible_h = space_time_high_res_x.shape[0] - size
        possible_w = space_time_high_res_x.shape[1] - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = space_time_high_res_x.shape[2] - num_timesteps
        assert possible_t >= 0

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        if possible_t > 0:
            start_t = np.random.choice(possible_t)
        else:
            start_t = possible_t

        return (
            space_time_high_res_x[
                start_h : start_h + size,
                start_w : start_w + size,
                start_t : start_t + num_timesteps,
            ],
            space_time_med_res_x[
                start_h : start_h + size,
                start_w : start_w + size,
                start_t : start_t + num_timesteps,
            ],
            space_time_low_res_x[
                start_h : start_h + size,
                start_w : start_w + size,
                start_t : start_t + num_timesteps,
            ],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x[start_t : start_t + num_timesteps],
            static_x,
            months[start_t : start_t + num_timesteps],
        )
    
    @staticmethod
    def one_hot_encode_esa_worldcover(data: np.ndarray) -> np.ndarray:
        """One-hot encode the ESA Worldcover band, setting all channels to NO_DATA_VALUE where class=0."""
        
        assert np.all(np.isin(data, WC_CLASS_VALUES + [0])), (
            "ESA Worldcover data contains unexpected class values."
        )
        nodata_mask = (data == 0)

        # Map class values to indices 0..NUM_WC_CLASSES-1
        mapped = np.zeros_like(data)
        for idx, class_value in enumerate(WC_CLASS_VALUES):
            mapped[data == class_value] = idx

        h, w = data.shape
        one_hot = np.zeros((h, w, NUM_WC_CLASSES), dtype=np.int16)

        # Standard one-hot encoding
        one_hot[np.arange(h)[:, None], np.arange(w), mapped] = 1

        # Set all channels to NO_DATA_VALUE where original class was 0
        one_hot[nodata_mask] = NO_DATA_VALUE
        return one_hot


    @staticmethod
    def _check_and_fillna(data: np.ndarray, bands_np: np.ndarray) -> np.ndarray:
        """Fill in the missing values in the data array"""
        if data.shape[-1] != len(bands_np):
            raise ValueError(f"Expected data to have {len(bands_np)} bands - got {data.shape[-1]}")
        is_nan_inf = np.isnan(data) | np.isinf(data)

        if not is_nan_inf.any():
            return data

        if len(data.shape) <= 2:
            return np.nan_to_num(data, nan=0)
        if len(data.shape) == 3:
            has_time = False
        elif len(data.shape) == 4:
            has_time = True
        else:
            raise ValueError(
                f"Expected data to be 3D or 4D (x, y, (time), band) - got {data.shape}"
            )

        # treat infinities as NaNs
        data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # if any of the bands has only nan values, array should be markes as invalid
        # assert np.isnan(data).all(axis=tuple(range(data.ndim - 1))).any()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_time_band = np.nanmean(data, axis=(0, 1))  # t, b or b

        mean_per_time_band = np.nan_to_num(mean_per_time_band, nan=0, posinf=0, neginf=0)
        assert not (np.isnan(mean_per_time_band).any() | np.isinf(mean_per_time_band).any())

        if is_nan_inf.any():
            if has_time:
                means_to_fill = (
                    repeat(
                        np.nanmean(mean_per_time_band, axis=0),
                        "b -> h w t b",
                        h=data.shape[0],
                        w=data.shape[1],
                        t=data.shape[2],
                    )
                    * is_nan_inf
                )
            else:
                means_to_fill = (
                    repeat(mean_per_time_band, "b -> h w b", h=data.shape[0], w=data.shape[1])
                    * is_nan_inf
                )
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0) + means_to_fill
        return data

    def tif_to_h5py_path(self, tif_path: Path) -> Path:
        assert self.h5py_folder is not None
        tif_name = tif_path.stem
        return self.h5py_folder / f"{tif_name}.h5"

    @staticmethod
    def downsample_dynamic_in_time_with_mean(data, mask, target_shape=(2, 2)):
        H, W, T, C = data.shape
        new_H, new_W = target_shape

        # make sure that we are processing dynamic-in-time array
        assert data.ndim == 4
        assert H % new_H == 0 and W % new_W == 0, "H and W must be divisible by target dimensions"

        # Compute block sizes
        h_block = H // new_H
        w_block = W // new_W

        # reshape
        # for data, take the mean over blocks, for the mask take the min (we want the block mask to be invalid where at least one value is invalid)
        return data.reshape(new_H, h_block, new_W, w_block, T, C).mean(axis=(1, 3)), mask.reshape(
            new_H, h_block, new_W, w_block, T, C
        ).min(axis=(1, 3))

    @classmethod
    def start_month_from_file(cls, tif_path: Path) -> int:
        start_date = tif_path.name.partition("dates=")[2][:10]
        start_month = int(start_date.split("-")[1])
        return start_month

    @classmethod
    def month_array_from_file(cls, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        start_month = cls.start_month_from_file(tif_path)
        # - 1 because we want to index from 0
        # TODO: account for the possibility that different timesteps can be in different months
        return np.full(num_timesteps, start_month - 1)

    @staticmethod
    def create_valid_mask(
        s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create masks that identify valid data to be used during normalization and modeling.

        0: invalid data
        1: valid data
        """
        assert s_t_h_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x"])
        assert s_t_m_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_m_x"])
        assert s_t_l_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_l_x"])
        assert sp_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"])
        assert t_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"])
        assert st_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"])

        # start by unmasking invalid data that is characterized by universal no data value
        valid_mask_s_t_h = s_t_h_x != NO_DATA_VALUE
        valid_mask_s_t_m = s_t_m_x != NO_DATA_VALUE
        valid_mask_s_t_l = s_t_l_x != NO_DATA_VALUE
        valid_mask_sp = sp_x != NO_DATA_VALUE
        valid_mask_t = t_x != NO_DATA_VALUE
        valid_mask_st = st_x != NO_DATA_VALUE

        # apply the channel-specific no-data bounds
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x"].items():
            valid_mask_s_t_h[..., ch] &= s_t_h_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_m_x"].items():
            valid_mask_s_t_m[..., ch] &= s_t_m_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_l_x"].items():
            valid_mask_s_t_l[..., ch] &= s_t_l_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"].items():
            valid_mask_sp[..., ch] &= sp_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"].items():
            valid_mask_t[..., ch] &= t_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"].items():
            valid_mask_st[..., ch] &= st_x[..., ch] >= lower_bound

        return (
            valid_mask_s_t_h,
            valid_mask_s_t_m,
            valid_mask_s_t_l,
            valid_mask_sp,
            valid_mask_t,
            valid_mask_st,
        )

    @classmethod
    def _tif_to_array(cls, tif_path: Path) -> DatasetOutput:
        """
        Loads a spatiotemporal tif file, divides it into different array groups, and creates valid data masks.

        The different array types are:
        space_time_high_res_x: (H, W, T, C_STH)
        space_time_med_res_x: (3, 3, T, C_STM)
        space_time_low_res_x: (2, 2, T, C_STL)
        space_x: (H, W, C_SP)
        time_x: (T, C_T)
        static_x: (C_ST)

        space_time_med_res_x and space_time_low_res_x are created by taking the block mean of their high res version.
        valid data masks are created by masking out values below a channel-specific threshold (0: invalid, 1: valid).
        """
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

            # extract lat, lon in EPSG:4326 from tif_path
            lat_pattern = r"lat=(.*?)_"
            lon_pattern = r"lon=(.*?)_"
            lat = float(
                np.mean([float(value) for value in re.findall(lat_pattern, str(tif_path))])
            )
            lon = float(
                np.mean([float(value) for value in re.findall(lon_pattern, str(tif_path))])
            )

        num_timesteps = (values.shape[0] - len(EE_SPACE_BANDS)) / len(EO_ALL_DYNAMIC_IN_TIME_BANDS)
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        assert num_timesteps == NUM_TIMESTEPS, f"{tif_path} has incorrect number of timesteps"
        dynamic_in_time_x = rearrange(
            values[: -(len(EE_SPACE_BANDS))],
            "(t c) h w -> h w t c",
            c=len(EO_ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = cls._check_and_fillna(
            dynamic_in_time_x, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP
        )
        space_time_high_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            : -(
                len(SPACE_TIME_MED_RES_BANDS)
                + len(EO_SPACE_TIME_LOW_RES_BANDS)
                + len(TIME_BANDS)
                + len(CLOUD_BANDS)
            ),
        ]
        space_time_med_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(
                len(SPACE_TIME_MED_RES_BANDS)
                + len(EO_SPACE_TIME_LOW_RES_BANDS)
                + len(TIME_BANDS)
                + len(CLOUD_BANDS)
            ) : -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)),
        ]
        space_time_low_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)) : -(
                len(TIME_BANDS) + len(CLOUD_BANDS)
            ),
        ]
        time_x = dynamic_in_time_x[
            :, :, :, -(len(TIME_BANDS) + len(CLOUD_BANDS)) : -len(CLOUD_BANDS)
        ]
        time_x = np.nanmean(time_x, axis=(0, 1))

        # NDSI = (Green - SWIR) / (Green + SWIR)
        if MODALITIES["ndsi"].get("active"):
            ndsi = cls.calculate_ndi(
                space_time_low_res_x, band_1="sur_refl_b04", band_2="sur_refl_b06"
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndsi), axis=-1)
            assert (ndsi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDSI for {tif_path}"
            )
            assert ((ndsi >= -1) & (ndsi <= 1) | (ndsi == NO_DATA_VALUE)).all(), (
                f"NDSI values out of bounds [-1, 1] for {tif_path}"
            )

        # NDVI = (NIR - Red) / (NIR + Red)
        if MODALITIES["ndvi"].get("active"):
            ndvi = cls.calculate_ndi(
                space_time_low_res_x, band_1="sur_refl_b02", band_2="sur_refl_b01"
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndvi), axis=-1)
            assert (ndvi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDVI for {tif_path}"
            )
            assert ((ndvi >= -1) & (ndvi <= 1) | (ndvi == NO_DATA_VALUE)).all(), (
                f"NDVI values out of bounds [-1, 1] for {tif_path}"
            )

        space_x = rearrange(
            values[-len(EE_SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = cls._check_and_fillna(space_x, np.array(EE_SPACE_BANDS))

        # one-hot encode ESA Worldcover band
        esa_wc = cls.one_hot_encode_esa_worldcover(space_x[:, :, ESA_WORLDCOVER_BAND_INDEX])
        space_x = np.concatenate((space_x[:, :, : (-len(EE_WC_BANDS))], esa_wc), axis=-1)

        static_x = to_cartesian(lat, lon)
        static_x = cls._check_and_fillna(static_x, np.array(STATIC_BANDS))

        months = cls.month_array_from_file(tif_path, int(num_timesteps))
        (
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
            months,
        ) = cls.subset_image(
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
            months,
            size=DATASET_OUTPUT_HW_HIGH_RES,
            num_timesteps=NUM_TIMESTEPS,
        )
        (
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = cls.create_valid_mask(
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
        )

        # for downsampling, the arrays need to be in divisible shape so we do it after cropping
        space_time_med_res_x, valid_data_mask_s_t_m = cls.downsample_dynamic_in_time_with_mean(
            space_time_med_res_x,
            valid_data_mask_s_t_m,
            target_shape=(NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM),
        )
        space_time_low_res_x, valid_data_mask_s_t_l = cls.downsample_dynamic_in_time_with_mean(
            space_time_low_res_x,
            valid_data_mask_s_t_l,
            target_shape=(NUM_LOW_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM),
        )

        try:
            assert not np.isnan(space_time_high_res_x).any(), f"NaNs in s_t_h_x for {tif_path}"
            assert not np.isnan(space_time_med_res_x).any(), f"NaNs in s_t_m_x for {tif_path}"
            assert not np.isnan(space_time_low_res_x).any(), f"NaNs in s_t_l_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_high_res_x).any(), f"Infs in s_t_h_x for {tif_path}"
            assert not np.isinf(space_time_med_res_x).any(), f"Infs in s_t_m_x for {tif_path}"
            assert not np.isinf(space_time_low_res_x).any(), f"Infs in s_t_l_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_high_res_x.astype(np.half),
                space_time_med_res_x.astype(np.half),
                space_time_low_res_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            )
        except AssertionError as e:
            raise e

    def _tif_to_array_with_checks(self, idx):
        tif_path = self.tifs[idx]
        try:
            dataset = self._tif_to_array(tif_path)
            return dataset
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.tifs[idx] = self.tifs[new_idx]
            tif_path = self.tifs[idx]
        dataset = self._tif_to_array(tif_path)
        return dataset

    def load_tif(self, idx: int) -> DatasetOutput:
        if self.h5py_folder is None:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = self._tif_to_array_with_checks(idx)
            return DatasetOutput(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            )
        else:
            h5py_path = self.tif_to_h5py_path(self.tifs[idx])
            if h5py_path.exists():
                try:
                    return self.read_and_slice_h5py_file(h5py_path)
                except Exception as e:
                    logger.warn(f"Exception {e} for {self.tifs[idx]}")
                    h5py_path.unlink()
                    (
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        months,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                    ) = self._tif_to_array_with_checks(idx)
                    self.save_h5py(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                        self.tifs[idx].stem,
                    )
                    return DatasetOutput(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        months,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                    )
            else:
                (
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                ) = self._tif_to_array_with_checks(idx)
                self.save_h5py(
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                    self.tifs[idx].stem,
                )
                return DatasetOutput(
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                )

    def save_h5py(
        self,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        valid_data_mask_s_t_h,
        valid_data_mask_s_t_m,
        valid_data_mask_s_t_l,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
        tif_stem,
    ):
        assert self.h5py_folder is not None
        with h5py.File(self.h5py_folder / f"{tif_stem}.h5", "w") as hf:
            hf.create_dataset("s_t_h_x", data=s_t_h_x)
            hf.create_dataset("s_t_m_x", data=s_t_m_x)
            hf.create_dataset("s_t_l_x", data=s_t_l_x)
            hf.create_dataset("sp_x", data=sp_x)
            hf.create_dataset("t_x", data=t_x)
            hf.create_dataset("st_x", data=st_x)
            hf.create_dataset("valid_data_mask_s_t_h", data=valid_data_mask_s_t_h)
            hf.create_dataset("valid_data_mask_s_t_m", data=valid_data_mask_s_t_m)
            hf.create_dataset("valid_data_mask_s_t_l", data=valid_data_mask_s_t_l)
            hf.create_dataset("valid_data_mask_sp", data=valid_data_mask_sp)
            hf.create_dataset("valid_data_mask_t", data=valid_data_mask_t)
            hf.create_dataset("valid_data_mask_st", data=valid_data_mask_st)

    @staticmethod
    def calculate_ndi(input_array: np.ndarray, band_1: str, band_2: str) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """

        for b in [band_1, band_2]:
            assert b in SPACE_TIME_LOW_RES_BANDS

        band_1_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_2)]

        invalid = (
            (band_1_np == NO_DATA_VALUE)
            | (band_1_np == MODIS_FILL_VALUE)
            | (band_2_np == NO_DATA_VALUE)
            | (band_2_np == MODIS_FILL_VALUE)
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndi = np.expand_dims(
                np.where(
                    ((band_1_np + band_2_np) > 0) & (~invalid),
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    NO_DATA_VALUE,
                ),
                -1,
            )
        # when the input bands have different signs, NDI can be outside [-1, 1]
        # set values outside valid range to NO_DATA_VALUE (will be masked out later)
        ndi[(ndi < -1) | (ndi > 1)] = NO_DATA_VALUE
        return ndi

    def read_and_slice_h5py_file(self, h5py_path: Path):
        with h5py.File(h5py_path, "r") as hf:
            assert hf["s_t_h_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(SPACE_TIME_HIGH_RES_BANDS),
            )
            assert hf["s_t_m_x"].shape == (
                NUM_MED_RES_PIXELS_PER_DIM,
                NUM_MED_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_MED_RES_BANDS),
            )
            assert hf["s_t_l_x"].shape == (
                NUM_LOW_RES_PIXELS_PER_DIM,
                NUM_LOW_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_LOW_RES_BANDS),
            )
            assert hf["sp_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(SPACE_BANDS),
            )
            assert hf["t_x"].shape == (self.output_timesteps, len(TIME_BANDS))
            assert hf["st_x"].shape == (len(STATIC_BANDS),)
            assert hf["valid_data_mask_s_t_h"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(SPACE_TIME_HIGH_RES_BANDS),
            )
            assert hf["valid_data_mask_s_t_m"].shape == (
                NUM_MED_RES_PIXELS_PER_DIM,
                NUM_MED_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_MED_RES_BANDS),
            )
            assert hf["valid_data_mask_s_t_l"].shape == (
                NUM_LOW_RES_PIXELS_PER_DIM,
                NUM_LOW_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_LOW_RES_BANDS),
            )
            assert hf["valid_data_mask_sp"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(SPACE_BANDS),
            )
            assert hf["valid_data_mask_t"].shape == (self.output_timesteps, len(TIME_BANDS))
            assert hf["valid_data_mask_st"].shape == (len(STATIC_BANDS),)

            months = self.month_array_from_file(h5py_path, self.output_timesteps)
            output = DatasetOutput(
                hf["s_t_h_x"][:],
                hf["s_t_m_x"][:],
                hf["s_t_l_x"][:],
                hf["sp_x"][:],
                hf["t_x"][:],
                hf["st_x"][:],
                months,
                hf["valid_data_mask_s_t_h"][:],
                hf["valid_data_mask_s_t_m"][:],
                hf["valid_data_mask_s_t_l"][:],
                hf["valid_data_mask_sp"][:],
                hf["valid_data_mask_t"][:],
                hf["valid_data_mask_st"][:],
            )
        return output

    def __getitem__(self, idx):
        if self.h5pys_only:
            return self.read_and_slice_h5py_file(self.h5pys[idx]).normalize(self.normalizer)
        else:
            h5py = self.load_tif(idx)
            return h5py.normalize(self.normalizer)

    def process_h5pys(self):
        # iterate through the dataset and save it all as h5pys
        assert self.h5py_folder is not None
        assert not self.h5pys_only
        assert self.cache

        for i in tqdm(range(len(self))):
            # loading the tifs also saves them
            # if they don't exist
            _ = self[i]

    @staticmethod
    def load_normalization_values(path: Path):
        if not path.exists():
            raise ValueError(f"No file found at path {path}")
        with path.open("r") as f:
            norm_dict = json.load(f)
        # we computed the normalizing dict using the same datset
        output_dict = {}
        for key, val in norm_dict.items():
            output_dict[key] = val
        return output_dict

    @staticmethod
    def plot_distribution(dataset, idx, channel_idx, assets_folder_name):
        import os

        import matplotlib.pyplot as plt

        os.makedirs(assets_folder_name, exist_ok=True)

        plt.figure()
        plt.hist(dataset.flatten(), bins=20)
        plt.savefig(f"{assets_folder_name}/{idx}_{channel_idx}.png")
        plt.close()

    def compute_normalization_values(
        self,
        output_hw: int = DATASET_OUTPUT_HW_HIGH_RES,
        output_timesteps: int = NUM_TIMESTEPS,
        estimate_from: Optional[int] = 10000,
        plot_distribution: bool = True,
        assets_folder_name: str = "assets",
    ):
        if estimate_from is not None:
            indices_to_sample = sample(list(range(len(self))), k=estimate_from)
        else:
            indices_to_sample = list(range(len(self)))

        output = ListOfDatasetOutputs([], [], [], [], [], [], [])
        for i in tqdm(indices_to_sample):
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = self[i]
            output.space_time_high_res_x.append(np.where(valid_data_mask_s_t_h, s_t_h_x, np.nan))
            output.space_time_med_res_x.append(np.where(valid_data_mask_s_t_m, s_t_m_x, np.nan))
            output.space_time_low_res_x.append(np.where(valid_data_mask_s_t_l, s_t_l_x, np.nan))
            output.space_x.append(np.where(valid_data_mask_sp, sp_x, np.nan))
            output.time_x.append(np.where(valid_data_mask_t, t_x, np.nan))
            output.static_x.append(np.where(valid_data_mask_st, st_x, np.nan))
            output.months.append(months)
        d_o = output.to_datasetoutput()

        if plot_distribution:
            for idx, ds in enumerate(d_o):
                for channel_idx in range(ds.shape[-1]):
                    self.plot_distribution(
                        ds[..., channel_idx], idx, channel_idx, assets_folder_name
                    )

        norm_dict = {
            "total_n": len(self),
            "sampled_n": len(indices_to_sample),
            "space_time_high_res": {
                "mean": np.nanmean(
                    d_o.space_time_high_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
                "std": np.nanstd(
                    d_o.space_time_high_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
            },
            "space_time_med_res": {
                "mean": np.nanmean(
                    d_o.space_time_med_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
                "std": np.nanstd(
                    d_o.space_time_med_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
            },
            "space_time_low_res": {
                "mean": np.nanmean(
                    d_o.space_time_low_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
                "std": np.nanstd(
                    d_o.space_time_low_res_x, axis=(0, 1, 2, 3), dtype=np.float64
                ).tolist(),
            },
            "space": {
                "mean": np.nanmean(d_o.space_x, axis=(0, 1, 2), dtype=np.float64).tolist(),
                "std": np.nanstd(d_o.space_x, axis=(0, 1, 2), dtype=np.float64).tolist(),
            },
            "time": {
                "mean": np.nanmean(d_o.time_x, axis=(0, 1), dtype=np.float64).tolist(),
                "std": np.nanstd(d_o.time_x, axis=(0, 1), dtype=np.float64).tolist(),
            },
            "static": {
                "mean": np.nanmean(d_o.static_x, axis=0, dtype=np.float64).tolist(),
                "std": np.nanstd(d_o.static_x, axis=0, dtype=np.float64).tolist(),
            },
        }

        with open(self.data_folder.parents[1] / "normalizing_dict.json", "w") as f:
            json.dump(norm_dict, f)

        return norm_dict

    def compute_running_stats(self, sampled_n=50000):
        """
        Compute running statistics for the entire dataset.
        """
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self[0]

        stats_high_res = RunningStats(shape=(s_t_h_x.shape[-1],))
        stats_med_res = RunningStats(shape=(s_t_m_x.shape[-1],))
        stats_low_res = RunningStats(shape=(s_t_l_x.shape[-1],))
        stats_space = RunningStats(shape=(sp_x.shape[-1],))
        stats_time = RunningStats(shape=(t_x.shape[-1],))
        stats_static = RunningStats(shape=(st_x.shape[-1],))

        for i in tqdm(range(len(self))):
            if i >= sampled_n:
                logger.info(f"Reached {sampled_n} samples, stopping computation.")
                break
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                _,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = self[i]
            s_t_h_x = np.where(valid_data_mask_s_t_h, s_t_h_x, np.nan)
            s_t_m_x = np.where(valid_data_mask_s_t_m, s_t_m_x, np.nan)
            s_t_l_x = np.where(valid_data_mask_s_t_l, s_t_l_x, np.nan)
            sp_x = np.where(valid_data_mask_sp, sp_x, np.nan)
            t_x = np.where(valid_data_mask_t, t_x, np.nan)
            st_x = np.where(valid_data_mask_st, st_x, np.nan)

            # Collapse dimensions if needed, e.g., s_t_h_x.shape = (H, W, T, C) --> reshape to (-1, C)
            stats_high_res.update(s_t_h_x.reshape(-1, s_t_h_x.shape[-1]))
            stats_med_res.update(s_t_m_x.reshape(-1, s_t_m_x.shape[-1]))
            stats_low_res.update(s_t_l_x.reshape(-1, s_t_l_x.shape[-1]))
            stats_space.update(sp_x.reshape(-1, sp_x.shape[-1]))
            stats_time.update(t_x.reshape(-1, t_x.shape[-1]))
            stats_static.update(st_x.reshape(-1, st_x.shape[-1]))

        s_t_h_x_mean, s_t_h_x_std = stats_high_res.finalize()
        s_t_m_x_mean, s_t_m_x_std = stats_med_res.finalize()
        s_t_l_x_mean, s_t_l_x_std = stats_low_res.finalize()
        sp_x_mean, sp_x_std = stats_space.finalize()
        t_x_mean, t_x_std = stats_time.finalize()
        st_x_mean, st_x_std = stats_static.finalize()

        norm_dict = {
            "total_n": len(self),
            "sampled_n": sampled_n,
            "space_time_high_res": {
                "mean": s_t_h_x_mean.tolist(),
                "std": s_t_h_x_std.tolist(),
            },
            "space_time_med_res": {
                "mean": s_t_m_x_mean.tolist(),
                "std": s_t_m_x_std.tolist(),
            },
            "space_time_low_res": {
                "mean": s_t_l_x_mean.tolist(),
                "std": s_t_l_x_std.tolist(),
            },
            "space": {
                "mean": sp_x_mean.tolist(),
                "std": sp_x_std.tolist(),
            },
            "time": {
                "mean": t_x_mean.tolist(),
                "std": t_x_std.tolist(),
            },
            "static": {
                "mean": st_x_mean.tolist(),
                "std": st_x_std.tolist(),
            },
        }

        with open(self.data_folder.parents[1] / "normalizing_dict_updated.json", "w") as f:
            json.dump(norm_dict, f)

        return norm_dict
