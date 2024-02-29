import os
from pathlib import Path
from typing import cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange

from .config import EE_BUCKET_TIFS
from .earthengine.eo import (
    DYNAMIC_BANDS,
    DYNAMIC_DIV_VALUES,
    DYNAMIC_SHIFT_VALUES,
    STATIC_BANDS,
    STATIC_DIV_VALUES,
    STATIC_SHIFT_VALUES,
)


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
        num_timesteps = (values.shape[0] - len(STATIC_BANDS)) / len(DYNAMIC_BANDS)
        assert num_timesteps % 1 == 0
        dynamic_data = rearrange(
            values[: -len(STATIC_BANDS)],
            "(t b) h w -> h w t b",
            b=len(DYNAMIC_BANDS),
            t=int(num_timesteps),
        )
        return (
            cls.normalize(dynamic_data, DYNAMIC_SHIFT_VALUES, DYNAMIC_DIV_VALUES),
            cls.normalize(static_data, STATIC_SHIFT_VALUES, STATIC_DIV_VALUES),
        )

    @staticmethod
    def normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray):
        return (x - shift_values) / div_values
