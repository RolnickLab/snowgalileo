import os
import re
from pathlib import Path
from typing import Dict, cast

import numpy as np
import psutil
import rioxarray
import xarray as xr
from einops import rearrange
from scipy import stats

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.data.dataset import Dataset as BaseDataset
from src.data.earthengine.eo import (
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
    def map_int_to_cloud_states(state: int):
        """Retrieve cloud state from MODIS QA state integer
        Returns if there is cloud, cloud shadow, cirrus detected
        """
        # fill value by MODIS is 0
        if state == 0:
            assert False, "Fill value encountered in MODIS QA state"

        # qa state translation from Table 13 in https://lpdaac.usgs.gov/documents/306/MOD09_User_Guide_V6.pdf
        # 16-bit unsigned integer, bit 0 is LSB
        qa_bin = format(state, ">016b")

        # mapping 0: clear, 1: cloudy, 2: mixed
        # 00 clear, 01 cloudy, 10 mixed, 11 clear
        cloud_state = qa_bin[:2]  # first two bits
        if cloud_state == "00":
            cloud_state = False
        elif cloud_state == "01":
            cloud_state = True
        elif cloud_state == "10":
            cloud_state = True
        elif cloud_state == "11":
            cloud_state = False

        # 0: no cloud shadow, 1: cloud shadow
        cloud_shadow = qa_bin[2]
        if cloud_shadow == "0":
            cloud_shadow = False
        else:
            cloud_shadow = True

        # 00: none, 01: small, 10: average, 11: high
        cirrus_detected = qa_bin[8:10]
        if cirrus_detected == "00":
            cirrus = False
        if cirrus_detected == "01":
            cirrus = True
        if cirrus_detected == "10":
            cirrus = True
        if cirrus_detected == "11":
            cirrus = True

        internal_cloud_flag = qa_bin[10]
        if internal_cloud_flag == "0":
            internal_cloud_flag = False
        else:
            internal_cloud_flag = True

        return qa_bin, (cloud_state or internal_cloud_flag), cloud_shadow, cirrus

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
        # get the mode cloud value for each image
        modis_cloud_x = stats.mode(modis_cloud_x, axis=(0, 1))[0]

        try:
            assert not np.isnan(modis_cloud_x).any(), f"NaNs in modis cloud for {tif_path}"
            assert not np.isinf(modis_cloud_x).any(), f"Infs in modis cloud for {tif_path}"
            return modis_cloud_x, lat, lon
        except AssertionError as e:
            raise e

    def _get_cloud_states(
        self, modis_cloud_x: np.ndarray, lat: float, lon: float, cloud_state_dict: dict
    ) -> Dict[str, int]:
        """Get the last day with cloud and total number of cloudy days from modis cloud data"""
        last_clear_day = -1
        total_clear_days = 0
        total_cloudy_days = 0
        total_cloud_shadow_days = 0
        total_cirrus_days = 0
        total_days = 0

        # check if any fill values (0) are present
        if (modis_cloud_x == 0).any():
            return cloud_state_dict.update(
                {
                    "last_clear_day": -1,
                    "total_clear_days": -1,
                    "total_cloudy_days": -1,
                    "total_cloud_shadow_days": -1,
                    "total_cirrus_days": -1,
                    "lat": np.nan,
                    "lon": np.nan,
                    "total_days": -1,
                }
            )

        # loops from beginning to end of time series, so last_clear_day is the last occurrence
        for t in range(NUM_TIMESTEPS):
            _, cloud, cloud_shadow, cirrus = self.map_int_to_cloud_states(
                modis_cloud_x[t].astype(int).item(0)
            )
            if (
                not cloud
                and not cloud_shadow
                and not cirrus
            ):
                last_clear_day = t
                total_clear_days += 1
            if cloud:
                total_cloudy_days += 1
            if cloud_shadow:
                total_cloud_shadow_days += 1
            if cirrus:
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
        cloud_state_dict = {"filename": filename}

        tif_path = Path(self.data_folder / filename)
        assert tif_path.exists(), f"File {tif_path} does not exist"
        if tif_path.suffix == ".tif":
            try:
                cloud_state_dict = self._get_cloud_states(
                    *self._get_cloud_band_and_location(tif_path), cloud_state_dict
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
                        "lat": np.nan,
                        "lon": np.nan,
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
