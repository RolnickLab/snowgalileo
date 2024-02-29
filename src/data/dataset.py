import os
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange

from .config import EE_BUCKET_TIFS
from .earthengine.eo import (
    DYNAMIC_BANDS as EO_DYNAMIC_BANDS,
)
from .earthengine.eo import (
    DYNAMIC_DIV_VALUES,
    DYNAMIC_SHIFT_VALUES,
    STATIC_BANDS,
    STATIC_DIV_VALUES,
    STATIC_SHIFT_VALUES,
)

DYNAMIC_BANDS = EO_DYNAMIC_BANDS + ["NDVI"]


class Dataset:
    def __init__(self, data_folder: Path, download: bool = True):
        self.data_folder = data_folder
        if download:
            self.download(data_folder)

    @staticmethod
    def download(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage cp -n -r gs://{EE_BUCKET_TIFS}/tifs/ {data_folder}")

    @classmethod
    def tif_to_array(cls, tif_path: Path):
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
