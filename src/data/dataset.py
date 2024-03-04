import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, cast
from typing import OrderedDict as OrderedDictType

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange

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


class Dataset:
    def __init__(self, data_folder: Path, download: bool = True):
        self.data_folder = data_folder
        if download:
            self.download(data_folder)
        self.tifs = list(data_folder.glob("*.tiff"))

    def __len__(self) -> int:
        return len(self.tifs)

    @staticmethod
    def download(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage cp -n -r gs://{EE_BUCKET_TIFS}/tifs/ {data_folder}")

    @classmethod
    def tif_to_array(cls, tif_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        data = cast(xr.Dataset, rioxarray.open_rasterio(tif_path))
        values = cast(np.ndarray, data.values)
        static_data = rearrange(values[-len(STATIC_BANDS) :], "b h w -> h w b")
        num_timesteps = (values.shape[0] - len(STATIC_BANDS)) / len(EO_DYNAMIC_BANDS)
        assert num_timesteps % 1 == 0
        dynamic_data = rearrange(
            values[: -len(STATIC_BANDS)],
            "(t b) h w -> h w t b",
            b=len(EO_DYNAMIC_BANDS),
            t=int(num_timesteps),
        )

        dynamic_data = cls.normalize(dynamic_data, DYNAMIC_SHIFT_VALUES, DYNAMIC_DIV_VALUES)
        dynamic_data = np.concatenate((dynamic_data, cls.calculate_ndvi(dynamic_data)), axis=-1)
        return (
            dynamic_data,
            cls.normalize(static_data, STATIC_SHIFT_VALUES, STATIC_DIV_VALUES),
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
