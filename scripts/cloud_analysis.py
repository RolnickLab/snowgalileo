import argparse
import os
import re
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import psutil
import rioxarray
import xarray as xr
from einops import rearrange, repeat
from scipy import stats

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.data.dataset import to_cartesian
from src.data.earthengine.eo import (
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    NUM_TIMESTEPS,
    SPACE_BANDS,
    STATIC_BANDS,
)
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

argparser = argparse.ArgumentParser()
argparser.add_argument("--tifs_folder", type=str, default="")
args = argparser.parse_args().__dict__

tifs_folder = DATA_FOLDER / args["tifs_folder"]
assert tifs_folder.exists(), f"{tifs_folder} does not exist!"


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
        raise ValueError(f"Expected data to be 3D or 4D (x, y, (time), band) - got {data.shape}")

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


def get_cloud_state_modis(state: int):
    # qa state translation from: https://lpdaac.usgs.gov/documents/306/MOD09_User_Guide_V6.pdf
    qa_bin = format(state, ">016b")

    # mapping 0: clear, 1: cloudy, 2: mixed
    # 00 clear, 01 cloudy, 10 mixed, 11 clear
    cloud_state = qa_bin[:2]  # first two bits
    if cloud_state == "00":
        cloud_state = 0
    elif cloud_state == "01":
        cloud_state = 1
    elif cloud_state == "10":
        cloud_state = 1
    elif cloud_state == "11":
        cloud_state = 1

    # 0: no cloud shadow, 1: cloud shadow
    cloud_shadow = qa_bin[2]
    if cloud_shadow == "0":
        cloud_shadow = 0
    else:
        cloud_shadow = 1

    # 00: none, 01: small, 10: average, 11: high
    cirrus_detected = qa_bin[8:10]
    if cirrus_detected == "00":
        cirrus = 0
    if cirrus_detected == "01":
        cirrus = 0
    if cirrus_detected == "10":
        cirrus = 0
    if cirrus_detected == "11":
        cirrus = 1

    internal_cloud_flag = qa_bin[10]
    if internal_cloud_flag == "0":
        internal_cloud_flag = 0
    else:
        internal_cloud_flag = 1

    return qa_bin, (cloud_state or internal_cloud_flag), cloud_shadow, cirrus


def _get_cloud_bands(tif_path: Path):
    with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
        # [all_combined_bands, H, W]
        # all_combined_bands includes all dynamic-in-time bands
        # interleaved for all timesteps
        # followed by the static-in-time bands
        values = cast(np.ndarray, data.values)

        # extract lat, lon in EPSG:4326 from tif_path
        lat_pattern = r"lat=(.*?)_"
        lon_pattern = r"lon=(.*?)_"
        lat = float(np.mean([float(value) for value in re.findall(lat_pattern, str(tif_path))]))
        lon = float(np.mean([float(value) for value in re.findall(lon_pattern, str(tif_path))]))

    num_timesteps = (values.shape[0] - len(SPACE_BANDS)) / len(EO_ALL_DYNAMIC_IN_TIME_BANDS)
    assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
    assert num_timesteps == NUM_TIMESTEPS, f"{tif_path} has incorrect number of timesteps"
    dynamic_in_time_x = rearrange(
        values[: -(len(SPACE_BANDS))],
        "(t c) h w -> h w t c",
        c=len(EO_ALL_DYNAMIC_IN_TIME_BANDS),
        t=int(num_timesteps),
    )
    dynamic_in_time_x = _check_and_fillna(dynamic_in_time_x, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP)
    # resolution: 1000m
    modis_cloud_x = dynamic_in_time_x[
        :,
        :,
        :,
        -3:-2,
    ]
    # get the mode cloud value for each image
    modis_cloud_x = stats.mode(modis_cloud_x, axis=(0, 1))[0]

    static_x = to_cartesian(lat, lon)
    static_x = _check_and_fillna(static_x, np.array(STATIC_BANDS))

    try:
        assert not np.isnan(modis_cloud_x).any(), f"NaNs in modis cloud for {tif_path}"
        assert not np.isinf(modis_cloud_x).any(), f"Infs in modis cloud for {tif_path}"
        return modis_cloud_x, static_x
    except AssertionError as e:
        raise e


def main():
    modis_cloud_counts = {"num_samples": 0, "cloud": 0, "cloud_shadow": 0, "cirrus": 0}
    num_samples = 3000

    all_files = [f for f in os.listdir(tifs_folder) if f.endswith(".tif")]
    random_subset = np.random.choice(all_files, num_samples, replace=False)

    for i in random_subset:
        tif_path = Path(tifs_folder / i)
        # print the number of files in the folder
        print(f"Processing {tif_path} - {len(os.listdir(tifs_folder))} files in folder")
        if tif_path.suffix == ".tif":
            try:
                modis_state, _ = _get_cloud_bands(tif_path)

                for timestep in range(NUM_TIMESTEPS):
                    cloud, cloud_shadow, cirrus = get_cloud_state_modis(
                        modis_state[timestep].astype(int).item(0)
                    )
                    modis_cloud_counts["num_samples"] += 1
                    modis_cloud_counts["cloud"] += cloud
                    modis_cloud_counts["cloud_shadow"] += cloud_shadow
                    modis_cloud_counts["cirrus"] += cirrus

                print(f"Processed {tif_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")
                continue

    assert modis_cloud_counts["num_samples"] == num_samples * NUM_TIMESTEPS, (
        f"Expected {num_samples * NUM_TIMESTEPS} samples, got {modis_cloud_counts['num_samples']}"
    )

    print(f"Modis cloud counts: {modis_cloud_counts}")


if __name__ == "__main__":
    main()
