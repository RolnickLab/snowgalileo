import os
from pathlib import Path
from typing import cast

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange

from .config import EE_BUCKET_TIFS
from .earthengine.eo import DYNAMIC_BANDS, STATIC_BANDS


class Dataset:
    def __init__(self, data_folder: Path, download: bool = True):
        self.data_folder = data_folder
        if download:
            self.download(data_folder)

    @staticmethod
    def download(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage cp -n -r gs://{EE_BUCKET_TIFS}/tifs/ {data_folder}")

    @staticmethod
    def tif_to_array(tif_path: Path):
        data = cast(xr.Dataset, rioxarray.open_rasterio(tif_path))
        values = cast(np.ndarray, data.values)
        static_data = values[len(STATIC_BANDS) :]  # [2, H, W]
        num_timesteps = (values.shape[0] - len(STATIC_BANDS)) / len(DYNAMIC_BANDS)
        assert num_timesteps % 1 == 0
        dynamic_data = rearrange(
            values[: -len(STATIC_BANDS)],
            "(t b) h w -> b t h w",
            b=len(DYNAMIC_BANDS),
            t=int(num_timesteps),
        )
        return dynamic_data, static_data
