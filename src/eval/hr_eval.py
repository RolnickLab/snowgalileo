from pathlib import Path
from typing import Union, cast

import numpy as np
import psutil
import rioxarray
import xarray as xr
from einops import rearrange

from src.config import DEFAULT_SEED
from src.data.config import NO_DATA_VALUE
from src.data.dataset import Dataset as BaseDataset
from src.data.earthengine.eo_eval import (
    CLOUD_BANDS,
    EE_SPACE_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    EO_SPACE_TIME_LOW_RES_BANDS,
    NUM_TIMESTEPS,
    SPACE_TIME_MED_RES_BANDS,
    TIME_BANDS,
)
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class HRMetaDataset(BaseDataset):
    """Dataset class for retrieving forest metadata from ESA WorldCover data"""

    def __init__(self, data_folder, download=False, h5pys_only=False, *args, **kwargs):
        super().__init__(
            data_folder=data_folder, download=download, h5pys_only=h5pys_only, *args, **kwargs
        )

    @classmethod
    def _get_hr(cls, tif_path: Path):
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

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

        num_hr_days = 0
        num_s1_days = 0
        num_s2_days = 0
        num_landsat_days = 0
        last_s1_day = -1
        last_s2_day = -1
        last_landsat_day = -1
        last_hr_day = -1

        hr_dict: dict[str, Union[int, float]] = {}

        # assumption: the first band of each sensor determines if the sensor data is present
        for t in range(NUM_TIMESTEPS - 1):
            s1_present = not np.any(
                space_time_high_res_x[:, :, t, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP.index("VV")]
                == NO_DATA_VALUE
            )
            s2_present = not np.any(
                space_time_high_res_x[:, :, t, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP.index("B2")]
                == NO_DATA_VALUE
            )
            landsat_present = not np.any(
                space_time_high_res_x[:, :, t, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP.index("B2_landsat")]
                == NO_DATA_VALUE
            )

            if s1_present or s2_present or landsat_present:
                num_hr_days += 1
                last_hr_day = t
            if s1_present:
                num_s1_days += 1
                last_s1_day = t
            if s2_present:
                num_s2_days += 1
                last_s2_day = t
            if landsat_present:
                num_landsat_days += 1
                last_landsat_day = t

        hr_dict.update(
            {
                "last_hr_day": last_hr_day,
                "num_hr_days": num_hr_days,
                "last_s1_day": last_s1_day,
                "num_s1_days": num_s1_days,
                "last_s2_day": last_s2_day,
                "num_s2_days": num_s2_days,
                "last_landsat_day": last_landsat_day,
                "num_landsat_days": num_landsat_days,
            }
        )
        return hr_dict

    def return_hr_from_filename(self, filename: str):
        hr_dict: dict[str, Union[int, str, float]] = {"filename": filename}

        tif_path = Path(self.data_folder / filename)
        assert tif_path.exists(), f"File {tif_path} does not exist"
        if tif_path.suffix == ".tif":
            try:
                hr_dict = self._get_hr(tif_path)
                print(f"Processed {tif_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")
                hr_dict.update(
                    {
                        "last_hr_day": -1,
                        "num_hr_days": -1,
                        "last_s1_day": -1,
                        "num_s1_days": -1,
                        "last_s2_day": -1,
                        "num_s2_days": -1,
                        "last_landsat_day": -1,
                        "num_landsat_days": -1,
                    }
                )
            return hr_dict, filename
        else:
            raise ValueError(f"File {tif_path} is not a .tif file")
