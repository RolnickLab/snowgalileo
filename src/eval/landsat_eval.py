import json
import re
import warnings
from pathlib import Path
from typing import cast, Optional, Union

import numpy as np
import rioxarray
import xarray as xr

from src.data.config import DATA_FOLDER
from src.data.dataset import Dataset as TifDataset
from src.utils import masked_output_np_to_tensor


with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval.json")).open("r") as f:
    config = json.load(f)


class LandsatEvalDataset(TifDataset):
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

        super().__init__(
            data_folder=self.input_tif_folder,
            download=False,
            h5py_folder=self.input_h5py_folder,
        )

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

        # TODO: exclude inputs where only half of the image is given

        if self.exclude_prediction_date:
            s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m = self.mask_prediction_timestep(
                s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m
            )

        label_path = self.label_tifs[idx]
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label_path)) as data:
            label = cast(np.ndarray, data.values)

            # TODO: add assertion that input and label have the same lat and lon
            lat_pattern = r"lat=(.*?)_"
            lon_pattern = r"lon=(.*?)_"
            lat = float(
                np.mean([float(value) for value in re.findall(lat_pattern, str(label_path))])
            )
            lon = float(
                np.mean([float(value) for value in re.findall(lon_pattern, str(label_path))])
            )

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