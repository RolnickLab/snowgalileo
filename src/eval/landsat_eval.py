import json
import re
import warnings
from pathlib import Path
from typing import cast, Optional, Union, Tuple

import numpy as np
import rioxarray
import xarray as xr

from src.data.config import DATA_FOLDER, NO_DATA_VALUE, CHANNEL_WISE_INVALID_DATA_THRESHOLDS, DATASET_OUTPUT_HW_HIGH_RES, DATASET_OUTPUT_HW_MED_RES, DATASET_OUTPUT_HW_LOW_RES, DATASET_OUTPUT_HW_LOW_RES, NUM_TIMESTEPS
from src.data.dataset import Dataset as TifDataset
from src.utils import masked_output_np_to_tensor

from torch.utils.data import Dataset as PyTorchDataset


with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval.json")).open("r") as f:
    config = json.load(f)


class LandsatEvalDataset(PyTorchDataset):
    def __init__(self, split: str = "train", exclude_prediction_date: bool = False):
        self.split = split
        # whether to exclude the prediction date from the input timesteps
        # if True, the prediction date will be masked out in the input
        self.exclude_prediction_date = exclude_prediction_date

        assert self.split in ["train", "valid", "test"]

        self.input_h5py_folder = DATA_FOLDER / config["input_h5py_folder"]
        self.label_folder = DATA_FOLDER / config["label_folder"]
        self.input_tif_folder = DATA_FOLDER / config["input_tif_folder"]

        # print the number of label tifs
        print(f"Number of label tifs: {len(list(self.label_folder.glob('*.tif')) + list(self.label_folder.glob('*.tiff')))}")

        # print the number of input h5pys
        print(f"Number of input h5pys: {len(list(self.input_h5py_folder.glob('*.h5py')))}")

        # print the number of input tifs
        print(f"Number of input tifs: {len(list(self.input_tif_folder.glob('*.tif')) + list(self.input_tif_folder.glob('*.tiff')))}")

        ### TODO: replace this by parent class init
        self.cache = True
        self.tifs = []
        tifs = list(self.data_folder.glob("*.tif")) + list(self.data_folder.glob("*.tiff"))
        for tif in tifs:
            try:
                _ = self.start_month_from_file(tif)
                self.tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for {tif}")
        self.h5pys = []

        self.output_hw_high_res = DATASET_OUTPUT_HW_HIGH_RES
        self.output_hw_med_res = DATASET_OUTPUT_HW_MED_RES
        self.output_hw_low_res = DATASET_OUTPUT_HW_LOW_RES
        self.output_timesteps = NUM_TIMESTEPS

        assert self.output_hw_high_res == 100
        assert self.output_hw_med_res == 100
        assert self.output_hw_low_res == 100
        ###

        self.label_tifs = []
        label_tifs = list(self.label_folder.glob("*.tif")) + list(self.label_folder.glob("*.tiff"))
        for tif in label_tifs:
            try:
                _ = self.prediction_month_from_file(tif)
                self.label_tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for {tif}")

    # NOTE: overwritten from TifDataset since the eval tif files have different naming conventions
    @classmethod
    def prediction_month_from_file(cls, tif_path: Path) -> int:
        # assumes the tif file name is in the format "LC09_YYYYMMDD_[FSC]_[lat]_[lon].tif"
        prediction_month = int(tif_path.name.split("_")[1][4:6])
        print(f"Start month: {prediction_month}", flush=True)
        return prediction_month

    def mask_prediction_timestep(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # NOTE: 0 = valid, 1 = masked
        assert self.exclude_prediction_date
        s_t_h_m[:, :, -1, :] = 1
        s_t_m_m[:, :, -1, :] = 1
        s_t_l_m[:, :, -1, :] = 1
        t_m[-1, :] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m
    
    @staticmethod
    def create_valid_mask(
        s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        We need to adjust the mask function to account for no data values that occur due to the evaluation-specific export.

        This function will mask out 0 values, and NO_DATA_VALUES that are based on missing sensors.

        0: invalid data
        1: valid data
        """
        print("Creating valid mask for LandsatEvalDataset", flush=True)
        assert s_t_h_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x"])
        assert s_t_m_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_m_x"])
        assert s_t_l_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_l_x"])
        assert sp_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"])
        assert t_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"])
        assert st_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"])

        # TODO: assert the amount of 0 values in the input data, to check if they are int or float
        print("Amount of 0 values in s_t_h_x:", np.sum(s_t_h_x == 0), flush=True)
        print("Amount of 0.0 values in s_t_h_x:", np.sum(s_t_h_x == 0.0), flush=True)
        print("Amount of NO_DATA values in s_t_h_x:", np.sum(s_t_h_x == NO_DATA_VALUE), flush=True)

        # start by unmasking invalid data that is characterized by universal no data value
        valid_mask_s_t_h = (s_t_h_x != NO_DATA_VALUE) and (s_t_h_x != 0)
        valid_mask_s_t_m = (s_t_m_x != NO_DATA_VALUE) and (s_t_m_x != 0)
        valid_mask_s_t_l = (s_t_l_x != NO_DATA_VALUE) and (s_t_l_x != 0)
        valid_mask_sp = (sp_x != NO_DATA_VALUE) and (sp_x != 0)
        valid_mask_t = (t_x != NO_DATA_VALUE) and (t_x != 0)
        valid_mask_st = (st_x != NO_DATA_VALUE) and (st_x != 0)

        print("Amount of invalid data in s_t_h_x:", np.sum(~valid_mask_s_t_h), flush=True)

        # apply the channel-specific no-data bounds
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x"].items():
            valid_mask_s_t_h[..., ch] &= s_t_h_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_m_x"].items():
            valid_mask_s_t_m[..., ch] &= s_t_m_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_l_x"].items():
            valid_mask_s_t_l[..., ch] &= s_t_l_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"].items():
            valid_mask_sp[..., ch] &= sp_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"].items():
            valid_mask_t[..., ch] &= t_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"].items():
            valid_mask_st[..., ch] &= st_x[..., ch] >= lower_bound

        return (
            valid_mask_s_t_h,
            valid_mask_s_t_m,
            valid_mask_s_t_l,
            valid_mask_sp,
            valid_mask_t,
            valid_mask_st,
        )

    def __getitem__(self, idx):
        # NOTE: input will be a DatasetOutput object
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            month,
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = super().__getitem__(idx)

        s_t_h_m = np.logical_not(valid_data_mask_s_t_h)
        s_t_m_m = np.logical_not(valid_data_mask_s_t_m)
        s_t_l_m = np.logical_not(valid_data_mask_s_t_l)
        sp_m = np.logical_not(valid_data_mask_sp)
        t_m = np.logical_not(valid_data_mask_t)
        st_m = np.logical_not(valid_data_mask_st)

        if self.exclude_prediction_date:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_timestep(
                s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m
            )

        label_path = self.label_tifs[idx].name
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label_path)) as data:
            label = cast(np.ndarray, data.values)

        assert self.input_tifs.name[idx] == self.label_tifs.name[idx], "Input and label tif paths do not match."

        return (
            masked_output_np_to_tensor(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                month,
            ),
            label,
        )


    def __len__(self) -> int:
        return len(self.label_tifs)
    

if __name__ == "__main__":
    dataset = LandsatEvalDataset(split="test", exclude_prediction_date=True)
    print(f"Number of samples in dataset: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample[0].shape}, Label shape: {sample[1].shape}")
    print(f"Prediction month: {dataset.prediction_month_from_file(dataset.label_tifs[0])}")