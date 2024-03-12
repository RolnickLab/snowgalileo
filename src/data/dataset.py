import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, cast
from typing import OrderedDict as OrderedDictType

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange, repeat
from torch.utils.data import Dataset as PyTorchDataset

from .config import EE_BUCKET_TIFS
from .earthengine.eo import (
    DW_BANDS,
    DYNAMIC_DIV_VALUES,
    DYNAMIC_SHIFT_VALUES,
    ERA5_BANDS,
    S1_BANDS,
    SRTM_BANDS,
    STATIC_BANDS,
    STATIC_DIV_VALUES,
    STATIC_SHIFT_VALUES,
)
from .earthengine.eo import DYNAMIC_BANDS as EO_DYNAMIC_BANDS

DYNAMIC_BANDS = EO_DYNAMIC_BANDS + ["NDVI"]

DYNAMIC_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [DYNAMIC_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [DYNAMIC_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [DYNAMIC_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [DYNAMIC_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [DYNAMIC_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [DYNAMIC_BANDS.index(b) for b in ["B11", "B12"]],
        "ERA5": [DYNAMIC_BANDS.index(b) for b in ERA5_BANDS],
        "DW": [DYNAMIC_BANDS.index(b) for b in DW_BANDS],
        "NDVI": [DYNAMIC_BANDS.index("NDVI")],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {"SRTM": [STATIC_BANDS.index(b) for b in SRTM_BANDS]}
)

NUM_DYNAMIC_BAND_GROUPS = len(DYNAMIC_BANDS_GROUPS_IDX)
NUM_STATIC_BAND_GROUPS = len(STATIC_BAND_GROUPS_IDX)

NUM_DYNAMIC_BANDS = len(DYNAMIC_BANDS)
NUM_STATIC_BANDS = len(STATIC_BANDS)


class Dataset(PyTorchDataset):
    def __init__(self, data_folder: Path, download: bool = True):
        self.data_folder = data_folder
        if download:
            self.download(data_folder)
        self.tifs = list(data_folder.glob("*.tif"))

    def __len__(self) -> int:
        return len(self.tifs)

    @staticmethod
    def download(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage cp -n -r gs://{EE_BUCKET_TIFS}/tifs/* {data_folder}")

    @staticmethod
    def _fillna(data: np.ndarray, bands_np: np.ndarray):
        """Fill in the missing values in the data array"""
        if len(data.shape) == 3:
            has_time = False
        elif len(data.shape) == 4:
            has_time = True
        else:
            raise ValueError(
                f"Expected data to be 3 or 4D (x, y, (time), band) - got {data.shape}"
            )
        if data.shape[-1] != len(bands_np):
            raise ValueError(f"Expected data to have {len(bands_np)} bands - got {data.shape[-1]}")

        is_nan = np.isnan(data)
        if not is_nan.any():
            return data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_time_band = np.nanmean(data, axis=(0, 1))  # t, b or b

        if np.isnan(mean_per_time_band).any():
            # If a band has all nan values, fill with default: 0
            mean_per_time_band = np.nan_to_num(mean_per_time_band, nan=0)
        if is_nan.any():
            if has_time:
                means_to_fill = (
                    repeat(
                        np.nanmean(mean_per_time_band, axis=0),
                        "b -> h w t b",
                        h=data.shape[0],
                        w=data.shape[1],
                        t=data.shape[2],
                    )
                    * is_nan
                )
            else:
                means_to_fill = (
                    repeat(mean_per_time_band, "b -> h w b", h=data.shape[0], w=data.shape[1])
                    * is_nan
                )
            data = np.nan_to_num(data, nan=0) + means_to_fill
        return data

    @staticmethod
    def month_to_array(start_month: int, num_timesteps: int):
        """
        Given a start_month and num_timesteps, returns an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # >>> np.fmod(np.array([9., 10, 11, 12, 13, 14]), 12)
        # array([ 9., 10., 11.,  0.,  1.,  2.])
        # - 1 because we want to index from 0
        return np.fmod(np.arange(start_month - 1, start_month - 1 + num_timesteps), 12)

    @classmethod
    def tif_to_array(cls, tif_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            values = cast(np.ndarray, data.values)

        static_data = rearrange(values[-len(STATIC_BANDS) :], "b h w -> h w b")
        static_data = cls._fillna(static_data, np.array(STATIC_BANDS))
        num_timesteps = (values.shape[0] - len(STATIC_BANDS)) / len(EO_DYNAMIC_BANDS)
        assert num_timesteps % 1 == 0
        dynamic_data = rearrange(
            values[: -len(STATIC_BANDS)],
            "(t b) h w -> h w t b",
            b=len(EO_DYNAMIC_BANDS),
            t=int(num_timesteps),
        )
        dynamic_data = cls._fillna(dynamic_data, np.array(EO_DYNAMIC_BANDS))

        dynamic_data = cls.normalize(dynamic_data, DYNAMIC_SHIFT_VALUES, DYNAMIC_DIV_VALUES)
        dynamic_data = np.concatenate((dynamic_data, cls.calculate_ndvi(dynamic_data)), axis=-1)

        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        start_date = tif_path.name.partition("dates=")[2][:10]
        start_month = int(start_date.split("-")[1])
        months = cls.month_to_array(start_month, int(num_timesteps))
        return (
            dynamic_data,
            cls.normalize(static_data, STATIC_SHIFT_VALUES, STATIC_DIV_VALUES),
            months,
        )

    @staticmethod
    def normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray):
        return (x - shift_values) / div_values

    @staticmethod
    def calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_BANDS), returns an array of shape
        [h, w, t, 1] representing NDVI,
        (b08 - b04) / (b08 + b04)
        """
        band_1, band_2 = "B8", "B4"
        band_1_np = input_array[:, :, :, EO_DYNAMIC_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, EO_DYNAMIC_BANDS.index(band_2)]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            return np.expand_dims(
                np.where(
                    (band_1_np + band_2_np) > 0,
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    0,
                ),
                -1,
            )

    def __getitem__(self, idx):
        raise NotImplementedError
