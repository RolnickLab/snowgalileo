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
from src.data.config import DATA_FOLDER, NO_DATA_VALUE
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
argparser.add_argument("--satellite", type=str, default="modis", choices=["modis", "landsat"])
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
        cirrus_detected = 0
    if cirrus_detected == "01":
        cirrus_detected = 0
    if cirrus_detected == "10":
        cirrus_detected = 0
    if cirrus_detected == "11":
        cirrus_detected = 1

    internal_cloud_flag = qa_bin[10]
    if internal_cloud_flag == "0":
        internal_cloud_flag = 0
    else:
        internal_cloud_flag = 1

    return qa_bin, (cloud_state or cloud_shadow or cirrus_detected or internal_cloud_flag)

def get_cloud_state_landsat_bit(state: int, cloud_dict: dict) -> dict:
    qa_bin = format(state, ">016b")

    if qa_bin[0] == "0":
        cloud_dict["fill"]["image"] += 1
    else:
        cloud_dict["fill"]["fill"] += 1

    if qa_bin[1] == "0":
        cloud_dict["dilated cloud"]["not dilated or no cloud"] += 1
    else:
        cloud_dict["dilated cloud"]["cloud dilation"] += 1
    if qa_bin[2] == "0":
        cloud_dict["cirrus"]["not dilated or no cloud"] += 1
    else:
        cloud_dict["cirrus"]["cloud dilation"] += 1
    if qa_bin[3] == "0":
        cloud_dict["cloud"]["not high confidence cloud"] += 1
    else:
        cloud_dict["cloud"]["high confidence cloud"] += 1
    if qa_bin[4] == "0":
        cloud_dict["cloud shadow"]["not high confidence cloud shadow"] += 1
    else:
        cloud_dict["cloud shadow"]["high confidence cloud shadow"] += 1
    if qa_bin[5] == "0":
        cloud_dict["snow"]["not high confidence snow"] += 1
    else:
        cloud_dict["snow"]["high confidence snow"] += 1
    if qa_bin[6] == "0":
        cloud_dict["clear"]["dilated cloud or cloud are set"] += 1
    else:
        cloud_dict["clear"]["dilated cloud or cloud are not set"] += 1
    if qa_bin[7] == "0":
        cloud_dict["water"]["land or cloud"] += 1
    else:
        cloud_dict["water"]["water"] += 1
    if qa_bin[8:10] == "00":
        cloud_dict["cloud confidence"]["no confidence set"] += 1
    elif qa_bin[8:10] == "01":
        cloud_dict["cloud confidence"]["low confidence"] += 1
    elif qa_bin[8:10] == "10":
        cloud_dict["cloud confidence"]["medium confidence"] += 1
    elif qa_bin[8:10] == "11":
        cloud_dict["cloud confidence"]["high confidence"] += 1
    if qa_bin[10:12] == "00":
        cloud_dict["cloud shadow confidence"]["no confidence set"] += 1
    elif qa_bin[10:12] == "01":
        cloud_dict["cloud shadow confidence"]["low confidence"] += 1
    elif qa_bin[10:12] == "10":
        cloud_dict["cloud shadow confidence"]["reserved"] += 1
    elif qa_bin[10:12] == "11":
        cloud_dict["cloud shadow confidence"]["high confidence"] += 1
    if qa_bin[12:14] == "00":
        cloud_dict["snow/ice confidence"]["no confidence set"] += 1
    elif qa_bin[12:14] == "01":
        cloud_dict["snow/ice confidence"]["low confidence"] += 1
    elif qa_bin[12:14] == "10":
        cloud_dict["snow/ice confidence"]["reserved"] += 1
    elif qa_bin[12:14] == "11":
        cloud_dict["snow/ice confidence"]["high confidence"] += 1
    if qa_bin[14:16] == "00":
        cloud_dict["cirrus confidence"]["no confidence set"] += 1
    elif qa_bin[14:16] == "01":
        cloud_dict["cirrus confidence"]["low confidence"] += 1
    elif qa_bin[14:16] == "10":
        cloud_dict["cirrus confidence"]["reserved"] += 1
    elif qa_bin[14:16] == "11":
        cloud_dict["cloud shadow confidence"]["high confidence"] += 1

    return cloud_dict


def get_cloud_state_landsat(state: int) -> str:
    if state == 21824:
        return "clear with lows set"
    elif state == 21826:
        return "dilated cloud over land"
    elif state == 21888:
        return "water with lows set"
    elif state == 21890:
        return "dilated cloud over water"
    elif state == 22080:
        return "mid conf cloud"
    elif state == 22144:
        return "mid conf cloud over water"
    elif state == 22280:
        return "high conf cloud"
    elif state == 23888:
        return "high conf cloud shadow"
    elif state == 23952:
        return "water with cloud shadow"
    elif state == 24088:
        return "mid conf cloud w shadow"
    elif state == 24216:
        return "mid conf cloud w shadow over water"
    elif state == 24344:
        return "high conf cloud w shadow"
    elif state == 24472:
        return "high conf cloud w shadow over water"
    elif state == 30048:
        return "high conf snow/ice"
    elif state == 54596:
        return "high conf cirrus"
    elif state == 54852:
        return "cirrus, mid cloud"
    elif state == 55052:
        return "cirrus, high cloud"
    else:
        return "unknown"


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
    # resolution: 60m
    s2_cloud_x = dynamic_in_time_x[
        :,
        :,
        :,
        -2:-1,
    ]
    # resolution: 30m
    landsat_cloud_x = dynamic_in_time_x[
        :,
        :,
        :,
        -1,
    ]
    # get the mode cloud value for each image
    modis_cloud_x = stats.mode(modis_cloud_x, axis=(0, 1))[0]

    static_x = to_cartesian(lat, lon)
    static_x = _check_and_fillna(static_x, np.array(STATIC_BANDS))

    try:
        assert not np.isnan(modis_cloud_x).any(), f"NaNs in modis cloud for {tif_path}"
        assert not np.isnan(s2_cloud_x).any(), f"NaNs in s2 cloud for {tif_path}"
        assert not np.isnan(landsat_cloud_x).any(), f"NaNs in landsat cloud for {tif_path}"
        assert not np.isinf(modis_cloud_x).any(), f"Infs in modis cloud for {tif_path}"
        assert not np.isinf(s2_cloud_x).any(), f"Infs in s2 cloud for {tif_path}"
        assert not np.isinf(landsat_cloud_x).any(), f"Infs in landsat cloud for {tif_path}"
        return modis_cloud_x, s2_cloud_x, landsat_cloud_x, static_x
    except AssertionError as e:
        raise e


def main():
    modis_cloud_counts = {"clear": 0, "cloudy": 0, "mixed": 0, "assumed_clear": 0}
    landsat_cloud_counts = {
        "clear with lows set": 0,
        "dilated cloud over land": 0,
        "water with lows set": 0,
        "dilated cloud over water": 0,
        "mid conf cloud": 0,
        "mid conf cloud over water": 0,
        "high conf cloud": 0,
        "high conf cloud shadow": 0,
        "water with cloud shadow": 0,
        "mid conf cloud w shadow": 0,
        "mid conf cloud w shadow over water": 0,
        "high conf cloud w shadow": 0,
        "high conf cloud w shadow over water": 0,
        "high conf snow/ice": 0,
        "high conf cirrus": 0,
        "cirrus, mid cloud": 0,
        "cirrus, high cloud": 0,
        "unknown": 0,
    }

    landsat_cloud_dict = {
        "fill": {"image": 0, "fill": 0},
        "dilated cloud": {"not dilated or no cloud": 0, "cloud dilation": 0},
        "cirrus": {"not dilated or no cloud": 0, "cloud dilation": 0},
        "cloud": {"not high confidence cloud": 0, "high confidence cloud": 0},
        "cloud shadow": {"not high confidence cloud shadow": 0, "high confidence cloud shadow": 0},
        "snow": {"not high confidence snow": 0, "high confidence snow": 0},
        "clear": {"dilated cloud or cloud are set": 0, "dilated cloud or cloud are not set": 0},
        "water": {"land or cloud": 0, "water": 0},
        "cloud confidence": {
            "no confidence set": 0,
            "low confidence": 0,
            "medium confidence": 0,
            "high confidence": 0,
        },
        "cloud shadow confidence": {
            "no confidence set": 0,
            "low confidence": 0,
            "reserved": 0,
            "high confidence": 0,
        },
        "snow/ice confidence": {
            "no confidence set": 0,
            "low confidence": 0,
            "reserved": 0,
            "high confidence": 0,
        },
        "cirrus confidence": {
            "no confidence set": 0,
            "low confidence": 0,
            "reserved": 0,
            "high confidence": 0,
        },
    }

    num_samples = 3000

    all_files = [f for f in os.listdir(tifs_folder) if f.endswith(".tif")]
    random_subset = np.random.choice(all_files, num_samples, replace=False)

    for i in random_subset:
        tif_path = Path(tifs_folder / i)
        # print the number of files in the folder
        print(f"Processing {tif_path} - {len(os.listdir(tifs_folder))} files in folder")
        if tif_path.suffix == ".tif":
            try:
                modis_state, s2_state, landsat_state, latlon = _get_cloud_bands(tif_path)

                if args["satellite"] == "modis":
                    for timestep in range(NUM_TIMESTEPS):
                        modis_cloud_map = get_cloud_state_modis(
                            modis_state[timestep].astype(int).item(0)
                        )
                        if modis_cloud_map == 0:
                            modis_cloud_counts["clear"] += 1
                        elif modis_cloud_map == 1:
                            modis_cloud_counts["cloudy"] += 1
                        elif modis_cloud_map == 2:
                            modis_cloud_counts["mixed"] += 1
                        elif modis_cloud_map == 3:
                            modis_cloud_counts["assumed_clear"] += 1
                elif args["satellite"] == "landsat":
                    for timestep in range(NUM_TIMESTEPS):
                        landsat_qa_state = landsat_state[timestep].astype(int).item(0)
                        if landsat_qa_state == NO_DATA_VALUE:
                            continue
                        landsat_cloud_map = get_cloud_state_landsat(landsat_qa_state)
                        if landsat_cloud_map in landsat_cloud_counts:
                            landsat_cloud_counts[landsat_cloud_map] += 1
                        landsat_cloud_dict = get_cloud_state_landsat_bit(
                            landsat_qa_state, landsat_cloud_dict
                        )

                print(f"Processed {tif_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")
                continue

    if args["satellite"] == "landsat":
        print(f"Landsat cloud counts: {landsat_cloud_counts}")
        landsat_cloud_map = np.array([count for count in landsat_cloud_counts.values()])
        print(f"Landsat cloud map: {landsat_cloud_map}")
        print(f"Landsat cloud dict: {landsat_cloud_dict}")
    elif args["satellite"] == "modis":
        print(f"Modis cloud counts: {modis_cloud_counts}")
        modis_cloud_map = np.array(
            [
                modis_cloud_counts["clear"],
                modis_cloud_counts["cloudy"],
                modis_cloud_counts["mixed"],
                modis_cloud_counts["assumed_clear"],
            ]
        )
        print(f"Modis cloud map: {modis_cloud_map}")


if __name__ == "__main__":
    main()
