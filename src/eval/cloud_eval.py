import os
from pathlib import Path
from typing import Dict, Union, cast

import numpy as np
import psutil
import rioxarray
import xarray as xr
from einops import rearrange

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.data.dataset import Dataset as BaseDataset
from src.data.earthengine.eo_eval import (
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    NUM_TIMESTEPS,
    SPACE_BANDS,
)
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class CloudMetaDataset(BaseDataset):
    """Dataset class for retrieving cloud metadata from MODIS data"""

    def __init__(self, data_folder, download=False, h5pys_only=False, *args, **kwargs):
        super().__init__(
            data_folder=data_folder, download=download, h5pys_only=h5pys_only, *args, **kwargs
        )

    @staticmethod
    def bitwise_extract(value: int, from_bit: int, to_bit: int = None) -> int:
        # python equivalent of https://gis.stackexchange.com/questions/349371/creating-cloud-free-images-out-of-a-mod09a1-modis-image-in-gee/349401#349401
        if to_bit is None:
            to_bit = from_bit
        mask_size = (to_bit - from_bit) + 1
        mask = (1 << mask_size) - 1
        return (value >> from_bit) & mask

    @staticmethod
    def map_int_to_cloud_states(state: int):
        """Retrieve cloud state from MODIS QA state integer
        QA state translation from Table 13 in https://lpdaac.usgs.gov/documents/306/MOD09_User_Guide_V6.pdf
        16-bit unsigned integer, bit 0 is LSB
        Returns if there is cloud, cloud shadow, cirrus detected
        """
        # fill value by MODIS is 0
        if state == 0:
            assert False, "Fill value encountered in MODIS QA state"

        # mapping 0: clear, 1: cloudy, 2: mixed
        # 00 clear, 01 cloudy, 10 mixed, 11 clear
        cloud_state: Union[bool, str] = CloudMetaDataset.bitwise_extract(
            state, 0, 1
        )  # first two bits
        if cloud_state == 0:
            cloud_state = False
        elif cloud_state == 1:
            cloud_state = True
        elif cloud_state == 2:
            cloud_state = True
        elif cloud_state == 3:
            cloud_state = False

        # 0: no cloud shadow, 1: cloud shadow
        cloud_shadow: Union[bool, str] = CloudMetaDataset.bitwise_extract(state, 2)
        if cloud_shadow == 0:
            cloud_shadow = False
        else:
            cloud_shadow = True

        # 00: none, 01: small, 10: average, 11: high
        cirrus_detected: Union[bool, str] = CloudMetaDataset.bitwise_extract(state, 8, 9)
        if cirrus_detected == 0:
            cirrus = False
        if cirrus_detected == 1:
            cirrus = True
        if cirrus_detected == 2:
            cirrus = True
        if cirrus_detected == 3:
            cirrus = True

        internal_cloud_flag: Union[bool, str] = CloudMetaDataset.bitwise_extract(state, 10)
        if internal_cloud_flag == 0:
            internal_cloud_flag = False
        else:
            internal_cloud_flag = True

        return (cloud_state or internal_cloud_flag), cloud_shadow, cirrus

    @classmethod
    def _get_cloud_band_and_location(cls, tif_path: Path):
        """Extract the MODIS cloud band from a tif file"""
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

            # extract lat, lon in EPSG:4326 from tif_path
            parts = tif_path.stem.split("_")
            lat = float(parts[3])
            lon = float(parts[4])

        # NOTE: hacky assert that will get triggered once the baselines branch is merged
        assert len(SPACE_BANDS) == 4, "Expected 4 space bands for space bands"

        num_timesteps = (values.shape[0] - len(SPACE_BANDS)) / len(EO_ALL_DYNAMIC_IN_TIME_BANDS)
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        assert num_timesteps == NUM_TIMESTEPS, f"{tif_path} has incorrect number of timesteps"
        dynamic_in_time_x = rearrange(
            values[: -(len(SPACE_BANDS))],
            "(t c) h w -> h w t c",
            c=len(EO_ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = cls._check_and_fillna(
            dynamic_in_time_x, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP
        )
        # resolution: 1000m
        modis_cloud_x = dynamic_in_time_x[
            :,
            :,
            :,
            -4:-3,
        ]

        try:
            assert not np.isnan(modis_cloud_x).any(), f"NaNs in modis cloud for {tif_path}"
            assert not np.isinf(modis_cloud_x).any(), f"Infs in modis cloud for {tif_path}"
            return modis_cloud_x, lat, lon
        except AssertionError as e:
            raise e

    def _get_cloud_states(
        self, modis_cloud_x: np.ndarray, lat: float, lon: float, cloud_state_dict: dict
    ) -> Dict[str, Union[int, str, float]]:
        """Get the last day with cloud and total number of cloudy days from modis cloud data"""
        last_clear_day = -1
        total_clear_days = 0
        total_cloudy_days = 0
        total_cloud_shadow_days = 0
        total_cirrus_days = 0
        total_days = 0

        # check if any fill values (0) are present
        if (modis_cloud_x == 0).any():
            cloud_state_dict.update(
                {
                    "last_clear_day": -1,
                    "total_clear_days": -1,
                    "total_cloudy_days": -1,
                    "total_cloud_shadow_days": -1,
                    "total_cirrus_days": -1,
                    "lat": "nan",
                    "lon": "nan",
                    "total_days": -1,
                }
            )
            return cloud_state_dict

        # loops through time series, so last_clear_day is the last occurrence
        # we exclude the last timestep from the analysis as in the case of Landsat, it will always be clear
        for t in range(NUM_TIMESTEPS - 1):
            states = [self.map_int_to_cloud_states(int(v)) for v in np.unique(modis_cloud_x[t])]

            # aggregate states for this day
            is_cloud = any(s[0] for s in states)
            is_cloud_shadow = any(s[1] for s in states)
            is_cirrus = any(s[2] for s in states)

            if not (is_cloud or is_cloud_shadow or is_cirrus):
                last_clear_day = t
                total_clear_days += 1

            if is_cloud:
                total_cloudy_days += 1
            if is_cloud_shadow:
                total_cloud_shadow_days += 1
            if is_cirrus:
                total_cirrus_days += 1

            total_days += 1

        cloud_state_dict.update(
            {
                "last_clear_day": last_clear_day,
                "total_clear_days": total_clear_days,
                "total_cloudy_days": total_cloudy_days,
                "total_cloud_shadow_days": total_cloud_shadow_days,
                "total_cirrus_days": total_cirrus_days,
                "lat": lat,
                "lon": lon,
                "total_days": total_days,
            }
        )
        return cloud_state_dict

    def return_cloud_state_from_filename(self, filename: str):
        cloud_state_dict: dict[str, Union[int, str, float]] = {"filename": filename}

        tif_path = Path(self.data_folder / filename)
        assert tif_path.exists(), f"File {tif_path} does not exist"
        if tif_path.suffix == ".tif":
            try:
                modis_cloud_x, lat, lon = self._get_cloud_band_and_location(tif_path)
                cloud_state_dict = self._get_cloud_states(
                    modis_cloud_x, lat, lon, cloud_state_dict
                )
                print(f"Processed {tif_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")
                cloud_state_dict.update(
                    {
                        "last_clear_day": -1,
                        "total_clear_days": -1,
                        "total_cloudy_days": -1,
                        "total_cloud_shadow_days": -1,
                        "total_cirrus_days": -1,
                        "lat": "nan",
                        "lon": "nan",
                        "total_days": -1,
                    }
                )
            return cloud_state_dict
        else:
            raise ValueError(f"File {tif_path} is not a .tif file")


if __name__ == "__main__":
    # NOTE: for testing purposes, remove later

    # test by getting cloud states for 1000 samples in tif folder
    tifs_folder = DATA_FOLDER / "landsat_eval_tifs/patches_UTM_5_95_cropped/test"
    cloud_dataset = CloudMetaDataset(data_folder=tifs_folder)
    num_samples = 1000

    all_files = [f for f in os.listdir(tifs_folder) if f.endswith(".tif")]
    random_subset = np.random.choice(all_files, num_samples, replace=False)

    for i in random_subset:
        tif_path = Path(tifs_folder / i)
        cloud_state = cloud_dataset.return_cloud_state_from_filename(i)
        print(cloud_state)
