import json
import re
import warnings
from pathlib import Path
from typing import cast, Optional, Union, Tuple
import logging
import torch
from typing import Dict, List, Sequence
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, r2_score, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score

import numpy as np
import rioxarray
import xarray as xr
import h5py
from einops import rearrange, repeat
from tqdm import tqdm

from src.data.config import (
    DATA_FOLDER,
    NO_DATA_VALUE,
    CHANNEL_WISE_INVALID_DATA_THRESHOLDS,
    DATASET_OUTPUT_HW_HIGH_RES,
    DATASET_OUTPUT_HW_MED_RES,
    DATASET_OUTPUT_HW_LOW_RES,
    DATASET_OUTPUT_HW_LOW_RES,
    NUM_TIMESTEPS,
    NUM_HIGH_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NORMALIZATION_DICT_FILENAME,
    MODALITIES,
)
from src.masking import _aggregate_mask_per_channel_group
from src.data.dataset import DatasetOutput, Normalizer, to_cartesian
from src.data.earthengine.eo_eval import (
    EO_SPACE_TIME_LOW_RES_BANDS,
    SPACE_BANDS,
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_LOW_RES_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    CLOUD_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX
)
from src.utils import masked_output_np_to_tensor, config_dir, device, DEFAULT_SEED
from src.eval.eval import EvalTask, model_class_name
from src.eval.patch_predict import get_finetune_results
from sklearn.base import BaseEstimator
from src.flexipresto import Encoder
from src.eval.patch_predict import EncoderWithHead, GalileoEncoderWithHead
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset as PyTorchDataset

logger = logging.getLogger("__main__")

class LandsatEvalDataset(PyTorchDataset):
    def __init__(
        self,
        split: str = "train",
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        normalizer: Optional[Normalizer] = None,
        data_config: Dict = {},
    ):
        self.split = split
        # whether to exclude the prediction date from the input timesteps
        # if True, the prediction date will be masked out in the input
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.normalizer = normalizer

        assert self.split in ["train", "test", "visualize"]

        self.label_folder = DATA_FOLDER / data_config["label_folder"] / self.split
        self.input_tif_folder = DATA_FOLDER / data_config["input_tif_folder"] / self.split

        if self.split != "visualize" and self.split != "":
            self.h5py_folder = DATA_FOLDER / data_config["input_h5py_folder"] / self.split
        else:
            self.h5py_folder = None

        # print the number of label tifs
        print(
            f"Number of label tifs: {len(list(self.label_folder.glob('*.tif')) + list(self.label_folder.glob('*.tiff')))}"
        )

        # print the number of input tifs
        print(
            f"Number of input tifs: {len(list(self.input_tif_folder.glob('*.tif')) + list(self.input_tif_folder.glob('*.tiff')))}"
        )

        ### TODO: replace this by parent class init
        self.cache = True
        self.input_tifs = []
        input_tifs = list(self.input_tif_folder.glob("*.tif")) + list(
            self.input_tif_folder.glob("*.tiff")
        )
        for tif in input_tifs:
            try:
                _ = self.prediction_month_from_file(tif)
                self.input_tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for input {tif}")
        self.h5pys = []

        self.output_hw_high_res = DATASET_OUTPUT_HW_HIGH_RES
        self.output_hw_med_res = DATASET_OUTPUT_HW_MED_RES
        self.output_hw_low_res = DATASET_OUTPUT_HW_LOW_RES
        self.output_timesteps = NUM_TIMESTEPS

        assert self.output_hw_high_res == 100
        assert self.output_hw_med_res == 100
        assert self.output_hw_low_res == 100
        ###

        self.label_height_width = data_config["input_height_width"]

        self.label_tifs = []
        label_tifs = list(self.label_folder.glob("*.tif")) + list(self.label_folder.glob("*.tiff"))
        for tif in label_tifs:
            try:
                _ = self.prediction_month_from_file(tif)
                self.label_tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for label {tif}")

        assert len(self.input_tifs) == len(self.label_tifs), (
            "Number of input tifs and label tifs do not match."
        )
        print(f"Number of input tifs: {len(self.input_tifs)}")
        print(f"Number of label tifs: {len(self.label_tifs)}")

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
    
    def mask_prediction_high_res(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # masks the high resolution, optical data in the prediction timestep
        # high resolution channel groups are: s1, s2, landsat, so we retain the first channel
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_high_res
        assert s_t_h_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)
        s_t_h_m[:, :, -1, 1:] = 1
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
        valid_mask_s_t_h = (s_t_h_x != NO_DATA_VALUE) & (s_t_h_x != 0)
        valid_mask_s_t_m = (s_t_m_x != NO_DATA_VALUE) & (s_t_m_x != 0)
        valid_mask_s_t_l = (s_t_l_x != NO_DATA_VALUE) & (s_t_l_x != 0)
        valid_mask_sp = (sp_x != NO_DATA_VALUE) & (sp_x != 0)
        valid_mask_t = (t_x != NO_DATA_VALUE) & (t_x != 0)
        valid_mask_st = (st_x != NO_DATA_VALUE) & (st_x != 0)

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

    @staticmethod
    def _check_and_fillna(data: np.ndarray, bands_np: np.ndarray) -> np.ndarray:
        """Fill in the missing values in the data array"""
        from einops import repeat

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
            raise ValueError(
                f"Expected data to be 3D or 4D (x, y, (time), band) - got {data.shape}"
            )

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

    def tif_to_h5py_path(self, tif_path: Path) -> Path:
        assert self.h5py_folder is not None
        tif_name = tif_path.stem
        return self.h5py_folder / f"{tif_name}.h5"

    @staticmethod
    def downsample_dynamic_in_time_with_mean(data, mask, target_shape=(2, 2)):
        H, W, T, C = data.shape
        new_H, new_W = target_shape

        # make sure that we are processing dynamic-in-time array
        assert data.ndim == 4
        assert H % new_H == 0 and W % new_W == 0, "H and W must be divisible by target dimensions"

        # Compute block sizes
        h_block = H // new_H
        w_block = W // new_W

        # reshape
        # for data, take the mean over blocks, for the mask take the min (we want the block mask to be invalid where at least one value is invalid)
        return data.reshape(new_H, h_block, new_W, w_block, T, C).mean(axis=(1, 3)), mask.reshape(
            new_H, h_block, new_W, w_block, T, C
        ).min(axis=(1, 3))

    @classmethod
    def month_array_from_file(cls, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        prediction_month = cls.prediction_month_from_file(tif_path)
        # - 1 because we want to index from 0
        # TODO: account for the possibility that different timesteps can be in different months
        return np.full(num_timesteps, prediction_month - 1)

    @classmethod
    def _tif_to_array(cls, tif_path: Path) -> DatasetOutput:
        """
        Loads a spatiotemporal tif file, divides it into different array groups, and creates valid data masks.

        The different array types are:
        space_time_high_res_x: (H, W, T, C_STH)
        space_time_med_res_x: (3, 3, T, C_STM)
        space_time_low_res_x: (2, 2, T, C_STL)
        space_x: (H, W, C_SP)
        time_x: (T, C_T)
        static_x: (C_ST)

        space_time_med_res_x and space_time_low_res_x are created by taking the block mean of their high res version.
        valid data masks are created by masking out values below a channel-specific threshold (0: invalid, 1: valid).
        """
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

            # extract lat, lon in EPSG:4326 from tif_path
            # TODO: make this dynamic in case the tif_path has a different naming convention
            parts = tif_path.stem.split("_")
            lat = float(parts[3])
            lon = float(parts[4])

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
        space_time_med_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(
                len(SPACE_TIME_MED_RES_BANDS)
                + len(EO_SPACE_TIME_LOW_RES_BANDS)
                + len(TIME_BANDS)
                + len(CLOUD_BANDS)
            ) : -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)),
        ]
        space_time_low_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)) : -(
                len(TIME_BANDS) + len(CLOUD_BANDS)
            ),
        ]
        time_x = dynamic_in_time_x[
            :, :, :, -(len(TIME_BANDS) + len(CLOUD_BANDS)) : -len(CLOUD_BANDS)
        ]
        time_x = np.nanmean(time_x, axis=(0, 1))

        # NDSI = (Green - SWIR) / (Green + SWIR)
        if MODALITIES["ndsi"].get("active"):
            ndsi = cls.calculate_ndi(
                space_time_low_res_x, band_1="sur_refl_b04", band_2="sur_refl_b06"
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndsi), axis=-1)

        # NDVI = (NIR - Red) / (NIR + Red)
        if MODALITIES["ndvi"].get("active"):
            ndvi = cls.calculate_ndi(
                space_time_low_res_x, band_1="sur_refl_b02", band_2="sur_refl_b01"
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndvi), axis=-1)

        space_x = rearrange(
            values[-len(SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = cls._check_and_fillna(space_x, np.array(SPACE_BANDS))

        static_x = to_cartesian(lat, lon)
        static_x = cls._check_and_fillna(static_x, np.array(STATIC_BANDS))

        months = cls.month_array_from_file(tif_path, int(num_timesteps))

        (
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = cls.create_valid_mask(
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
        )

        # for downsampling, the arrays need to be in divisible shape so we do it after cropping
        space_time_med_res_x, valid_data_mask_s_t_m = cls.downsample_dynamic_in_time_with_mean(
            space_time_med_res_x,
            valid_data_mask_s_t_m,
            target_shape=(NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM),
        )
        space_time_low_res_x, valid_data_mask_s_t_l = cls.downsample_dynamic_in_time_with_mean(
            space_time_low_res_x,
            valid_data_mask_s_t_l,
            target_shape=(NUM_LOW_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM),
        )

        try:
            assert not np.isnan(space_time_high_res_x).any(), f"NaNs in s_t_h_x for {tif_path}"
            assert not np.isnan(space_time_med_res_x).any(), f"NaNs in s_t_m_x for {tif_path}"
            assert not np.isnan(space_time_low_res_x).any(), f"NaNs in s_t_l_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_high_res_x).any(), f"Infs in s_t_h_x for {tif_path}"
            assert not np.isinf(space_time_med_res_x).any(), f"Infs in s_t_m_x for {tif_path}"
            assert not np.isinf(space_time_low_res_x).any(), f"Infs in s_t_l_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_high_res_x.astype(np.half),
                space_time_med_res_x.astype(np.half),
                space_time_low_res_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            )
        except AssertionError as e:
            raise e

    def _tif_to_array_with_checks(self, idx):
        tif_path = self.input_tifs[idx]
        try:
            dataset = self._tif_to_array(tif_path)
            return dataset
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.input_tifs[idx] = self.input_tifs[new_idx]
            tif_path = self.input_tifs[idx]
        dataset = self._tif_to_array(tif_path)
        return dataset

    def load_tif(self, idx: int) -> DatasetOutput:
        if self.h5py_folder is None:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = self._tif_to_array_with_checks(idx)
            return DatasetOutput(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            )
        else:
            h5py_path = self.tif_to_h5py_path(self.input_tifs[idx])
            if h5py_path.exists():
                try:
                    return self.read_and_slice_h5py_file(h5py_path)
                except Exception as e:
                    logger.warn(f"Exception {e} for {self.input_tifs[idx]}")
                    h5py_path.unlink()
                    (
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        months,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                    ) = self._tif_to_array_with_checks(idx)
                    self.save_h5py(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                        self.input_tifs[idx].stem,
                    )
                    return DatasetOutput(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        months,
                        valid_data_mask_s_t_h,
                        valid_data_mask_s_t_m,
                        valid_data_mask_s_t_l,
                        valid_data_mask_sp,
                        valid_data_mask_t,
                        valid_data_mask_st,
                    )
            else:
                (
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                ) = self._tif_to_array_with_checks(idx)
                self.save_h5py(
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                    self.input_tifs[idx].stem,
                )
                return DatasetOutput(
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    valid_data_mask_s_t_h,
                    valid_data_mask_s_t_m,
                    valid_data_mask_s_t_l,
                    valid_data_mask_sp,
                    valid_data_mask_t,
                    valid_data_mask_st,
                )

    def save_h5py(
        self,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        valid_data_mask_s_t_h,
        valid_data_mask_s_t_m,
        valid_data_mask_s_t_l,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
        tif_stem,
    ):
        assert self.h5py_folder is not None
        with h5py.File(self.h5py_folder / f"{tif_stem}.h5", "w") as hf:
            hf.create_dataset("s_t_h_x", data=s_t_h_x)
            hf.create_dataset("s_t_m_x", data=s_t_m_x)
            hf.create_dataset("s_t_l_x", data=s_t_l_x)
            hf.create_dataset("sp_x", data=sp_x)
            hf.create_dataset("t_x", data=t_x)
            hf.create_dataset("st_x", data=st_x)
            hf.create_dataset("valid_data_mask_s_t_h", data=valid_data_mask_s_t_h)
            hf.create_dataset("valid_data_mask_s_t_m", data=valid_data_mask_s_t_m)
            hf.create_dataset("valid_data_mask_s_t_l", data=valid_data_mask_s_t_l)
            hf.create_dataset("valid_data_mask_sp", data=valid_data_mask_sp)
            hf.create_dataset("valid_data_mask_t", data=valid_data_mask_t)
            hf.create_dataset("valid_data_mask_st", data=valid_data_mask_st)

    @staticmethod
    def calculate_ndi(input_array: np.ndarray, band_1: str, band_2: str) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """

        # TODO: make this dynamic instead
        assert band_1 in SPACE_TIME_LOW_RES_BANDS
        assert band_2 in SPACE_TIME_LOW_RES_BANDS

        band_1_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_2)]

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

    def read_and_slice_h5py_file(self, h5py_path: Path):
        with h5py.File(h5py_path, "r") as hf:
            print(f"Reading h5py file {h5py_path}", flush=True)
            assert hf["s_t_h_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(SPACE_TIME_HIGH_RES_BANDS),
            )
            assert hf["s_t_m_x"].shape == (
                NUM_MED_RES_PIXELS_PER_DIM,
                NUM_MED_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_MED_RES_BANDS),
            )
            assert hf["s_t_l_x"].shape == (
                NUM_LOW_RES_PIXELS_PER_DIM,
                NUM_LOW_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_LOW_RES_BANDS),
            )
            assert hf["sp_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(SPACE_BANDS),
            )
            assert hf["t_x"].shape == (self.output_timesteps, len(TIME_BANDS))
            assert hf["st_x"].shape == (len(STATIC_BANDS),)
            assert hf["valid_data_mask_s_t_h"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(SPACE_TIME_HIGH_RES_BANDS),
            )
            assert hf["valid_data_mask_s_t_m"].shape == (
                NUM_MED_RES_PIXELS_PER_DIM,
                NUM_MED_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_MED_RES_BANDS),
            )
            assert hf["valid_data_mask_s_t_l"].shape == (
                NUM_LOW_RES_PIXELS_PER_DIM,
                NUM_LOW_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_LOW_RES_BANDS),
            )
            assert hf["valid_data_mask_sp"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(SPACE_BANDS),
            )
            assert hf["valid_data_mask_t"].shape == (self.output_timesteps, len(TIME_BANDS))
            assert hf["valid_data_mask_st"].shape == (len(STATIC_BANDS),)

            months = self.month_array_from_file(h5py_path, self.output_timesteps)
            output = DatasetOutput(
                hf["s_t_h_x"][:],
                hf["s_t_m_x"][:],
                hf["s_t_l_x"][:],
                hf["sp_x"][:],
                hf["t_x"][:],
                hf["st_x"][:],
                months,
                hf["valid_data_mask_s_t_h"][:],
                hf["valid_data_mask_s_t_m"][:],
                hf["valid_data_mask_s_t_l"][:],
                hf["valid_data_mask_sp"][:],
                hf["valid_data_mask_t"][:],
                hf["valid_data_mask_st"][:],
            )
        return output

    @staticmethod
    def load_normalization_values(path: Path):
        if not path.exists():
            raise ValueError(f"No file found at path {path}")
        with path.open("r") as f:
            norm_dict = json.load(f)
        # we computed the normalizing dict using the same datset
        output_dict = {}
        for key, val in norm_dict.items():
            output_dict[key] = val
        return output_dict

    def __getitem__(self, idx):
        # NOTE: input will be a DatasetOutput object
        h5py = self.load_tif(idx)
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
        ) = h5py.normalize(self.normalizer)

        # unmask everything per default, then mask invalid data
        s_t_h_m = torch.zeros((self.output_hw_high_res, self.output_hw_high_res, self.output_timesteps, len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)))
        s_t_m_m = torch.zeros((NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM, self.output_timesteps, len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX)))
        s_t_l_m = torch.zeros((NUM_LOW_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM, self.output_timesteps, len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX)))
        sp_m = torch.zeros((self.output_hw_high_res, self.output_hw_high_res, len(SPACE_BAND_GROUPS_IDX)))
        t_m = torch.zeros((self.output_timesteps, len(TIME_BANDS_GROUPS_IDX)))
        st_m = torch.zeros((len(STATIC_BAND_GROUPS_IDX),))

        invalid_data_mask_s_t_h = torch.as_tensor(np.logical_not(valid_data_mask_s_t_h))
        invalid_data_mask_s_t_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_m))
        invalid_data_mask_s_t_l = torch.as_tensor(np.logical_not(valid_data_mask_s_t_l))
        invalid_data_mask_sp = torch.as_tensor(np.logical_not(valid_data_mask_sp))
        invalid_data_mask_t = torch.as_tensor(np.logical_not(valid_data_mask_t))
        invalid_data_mask_st = torch.as_tensor(np.logical_not(valid_data_mask_st))

        cg_mask_s_t_h, cg_mask_s_t_m, cg_mask_s_t_l, cg_mask_sp, cg_mask_t, cg_mask_st = (
            _aggregate_mask_per_channel_group(
                invalid_data_mask_s_t_h,
                invalid_data_mask_s_t_m,
                invalid_data_mask_s_t_l,
                invalid_data_mask_sp,
                invalid_data_mask_t,
                invalid_data_mask_st,
            )
        )

        # since we mask out the same values within each channel we can assume that the mask is the same for each channel group
        s_t_h_m[cg_mask_s_t_h.bool()] = 1
        s_t_m_m[cg_mask_s_t_m.bool()] = 1
        s_t_l_m[cg_mask_s_t_l.bool()] = 1
        sp_m[cg_mask_sp.bool()] = 1
        t_m[cg_mask_t.bool()] = 1
        st_m[cg_mask_st.bool()] = 1

        if self.exclude_prediction_date:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_timestep(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        if self.exclude_prediction_high_res:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_high_res(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        label = self.label_tifs[idx]
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label)) as data:
            label = cast(np.ndarray, data.values)
            # remove first dimension
            label = np.squeeze(label, axis=0)
            print(f"Label shape: {label.shape}", flush=True)


        # if assertion is triggered, go to the next tif file
        try:
            assert self.input_tifs[idx].name == self.label_tifs[idx].name, (f"Input path {self.input_tifs[idx].name} and label path {self.label_tifs[idx].name} do not match.")
        except AssertionError:
            print(
                f"Label shape {label.shape} does not match expected shape ({self.label_height_width}, {self.label_height_width}) for {self.label_tifs[idx].name}"
            )
            self.label_tifs[idx] = self.label_tifs[idx + 1] if idx < len(self.label_tifs) - 1 else self.label_tifs[idx - 1]
            return self.__getitem__(idx)

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
            self.input_tifs[idx].name,  # for logging purposes
        )

    def __len__(self) -> int:
        return len(self.label_tifs)


class LandsatEval(EvalTask):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        patch_size_high_res: int = 10,
        seed=DEFAULT_SEED,
        resample: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = None,
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.patch_size_high_res = patch_size_high_res
        self.resample = resample
        self.num_finetune_epochs = num_finetune_epochs
        self.decoder_mode = decoder_mode

        super().__init__(self.patch_size_high_res, seed)
        self.name = (
            f"{'attn' if self.decoder_mode == 'attention_probe' else 'linear' if self.decoder_mode == 'linear_probe' else 'finetune' if self.decoder_mode == 'finetune' else 'sklearn'}_{'_exclude_prediction_date_' if self.exclude_prediction_date else ''}{'_no_high_res_in_pred_date' if self.exclude_prediction_high_res else ''}"
        )
        self.eval_config = eval_config
        self.data_config = self.eval_config["data"]

    def compute_regression_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        return {
            f"{self.name}_{model_name}_rmse": root_mean_squared_error(target, preds),
            f"{self.name}_{model_name}_r2": r2_score(target, preds),
        }
    
    def compute_baseline_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        return {
            f"baseline_{self.name}_{model_name}_rmse": root_mean_squared_error(target, preds),
            f"baseline_{self.name}_{model_name}_r2": r2_score(target, preds),
        }
    
    def compute_classification_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray, baseline=False) -> Dict[str, float]:
        if baseline:
            bs = "baseline_"
        else:
            bs = ""

        return {
            f"{bs}{self.name}_{model_name}_overall_accuracy": accuracy_score(target, preds),
            f"{bs}{self.name}_{model_name}_balanced_accuracy": balanced_accuracy_score(target, preds),
            f"{bs}{self.name}_{model_name}_recall": recall_score(target, preds, average='weighted'),
            f"{bs}{self.name}_{model_name}_precision": precision_score(target, preds, average='weighted'),
            f"{bs}{self.name}_{model_name}_f1": f1_score(target, preds, average='weighted'),
        }

    def get_test_dl(self, baseline_galileo=False, hyperparams_config=None) -> DataLoader:
        if baseline_galileo:
            from src.eval.landsat_baselines import LandsatEvalDatasetGalileo, GalileoNormalizer, galileo_config_dir, GALILEO_NORMALIZATION_DICT_FILENAME
            test_ds = LandsatEvalDatasetGalileo(
                exclude_prediction_date=self.exclude_prediction_date,
                exclude_prediction_high_res=self.exclude_prediction_high_res,
                split="test",
            )
            if self.normalization == "std":
                normalizer=GalileoNormalizer(std=True,
                                             normalizing_dicts=test_ds.load_normalization_values(galileo_config_dir / GALILEO_NORMALIZATION_DICT_FILENAME), 
                                             std_multiplier=2)
                print(test_ds.load_normalization_values(galileo_config_dir / GALILEO_NORMALIZATION_DICT_FILENAME), flush=True)
            else:
                normalizer = GalileoNormalizer(std=False)
            test_ds.normalizer = normalizer
        else:
            test_ds = LandsatEvalDataset(
                exclude_prediction_date=self.exclude_prediction_date,
                exclude_prediction_high_res=self.exclude_prediction_high_res,
                split="test",
                data_config=self.data_config
            )

            if self.normalization == "std":
                normalizing_dict = test_ds.load_normalization_values(
                    path=config_dir / NORMALIZATION_DICT_FILENAME
                )
                print(normalizing_dict, flush=True)
                normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
            else:
                normalizer = Normalizer(std=False)
            test_ds.normalizer = normalizer

        test_dl = DataLoader(
            test_ds,
            batch_size=hyperparams_config["batch_size"],
            shuffle=False,
            num_workers=hyperparams_config["num_workers"],
        )
        return test_dl

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator], baseline_galileo=False, hyperparams_config=None
    ) -> Dict:
        
        prediction_folder = DATA_FOLDER / "predictions"
        if not prediction_folder.exists():
            prediction_folder.mkdir(parents=True, exist_ok=True)

        test_dl = self.get_test_dl(baseline_galileo=baseline_galileo, hyperparams_config=hyperparams_config)

        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }
        results_dict: Dict[str, float] = {}
        pred_list = []

        encodings_list = []
        labels_list = []

        if baseline_galileo:
            for masked_output, label,_ in tqdm(test_dl, desc="Computing test predictions"):
                (
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                ) = [t.to(device) for t in masked_output]

                labels_list.append(self.rearrange_targets_into_token_sequence(label))

                pretrained_model.eval()
                with torch.no_grad():
                    (
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        _,
                    ) = pretrained_model(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months,
                        patch_size=self.patch_size_high_res,
                    )

                encodings = self.group_encodings_per_token_galileo_baseline(
                    pretrained_model,
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                )
                encodings_list.append(encodings.cpu().numpy())

        else:
            for masked_output, label,_ in tqdm(test_dl, desc="Computing test predictions"):
                (
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
                    months,
                ) = [t.to(device) for t in masked_output]

                labels_list.append(self.rearrange_targets_into_token_sequence(label))

                pretrained_model.eval()
                with torch.no_grad():
                    (
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
                        _,
                    ) = pretrained_model(
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
                        months,
                        patch_size_high_res=self.patch_size_high_res,
                        patch_size_med_res=1,
                        patch_size_low_res=1,
                    )

                encodings = self.group_encodings_per_token(
                    pretrained_model,
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
                )
                encodings_list.append(encodings.cpu().numpy())

        encodings_np, targets_np = np.concatenate(encodings_list), np.concatenate(labels_list)

        print(f"Shape of encodings: {encodings_np.shape}", flush=True)
        print(f"Shape of targets: {targets_np.shape}", flush=True)

        for model in sklearn_models:
            preds = model.predict(encodings_np)
            pred_dict[model_class_name(model)].append(preds)
        
        # TODO: careful, this only works if we only use one model
        pred_list.append(preds)

        preds_np = np.concatenate(pred_list)
        baseline_np = np.zeros_like(preds_np)

        # TODO: Binning

        # create 10 bins for multi-class classification
        multi_class_bins = np.linspace(0.1, 1, 9)
        binned_preds_np = np.digitize(preds_np, bins=multi_class_bins)
        binned_targets_np = np.digitize(targets_np, bins=multi_class_bins)

        for model_name_str, pred_list in pred_dict.items():
            results_dict.update(
                self.compute_regression_metrics(
                    model_name_str,
                    preds_np,
                    targets_np,
                )
            )
            results_dict.update(
                self.compute_baseline_metrics(
                    model_name_str,
                    baseline_np,
                    targets_np,
                )
            )
            results_dict.update(
                self.compute_classification_metrics(
                    model_name_str,
                    binned_preds_np,
                    binned_targets_np,
                    baseline=False,
                )
            )
            results_dict.update(
                self.compute_classification_metrics(
                    model_name_str,
                    baseline_np,
                    binned_targets_np,
                    baseline=True,
                )
            )
        np.save(
            prediction_folder / f"predictions_final.npy",
            preds_np,
        )

        return results_dict

    @torch.no_grad()
    def _visualize_best_worst(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator], num_images: int = 50, sort_for: str = "overall_accuracy", baseline_galileo=False, hyperparams_config=None
    ) -> Dict:
        
        prediction_folder = DATA_FOLDER / "ascending_accuracy_predictions"
        if not prediction_folder.exists():
            prediction_folder.mkdir(parents=True, exist_ok=True)

        test_dl = self.get_test_dl(baseline_galileo=baseline_galileo, hyperparams_config=hyperparams_config)

        predictions = []
        targets = []
        results_per_image = []

        for masked_output, label, filename in tqdm(test_dl, desc="Computing test predictions"):
            (
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
                months,
            ) = [t.to(device) for t in masked_output]

            label = label.squeeze(0).numpy()

            pretrained_model.eval()
            with torch.no_grad():
                (
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
                    _,
                ) = pretrained_model(
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
                    months,
                    patch_size_high_res=self.patch_size_high_res,
                    patch_size_med_res=1,
                    patch_size_low_res=1,
                )

            encodings = self.group_encodings_per_token(
                pretrained_model,
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
            )
            encodings = encodings.cpu().numpy()

            for model in sklearn_models:

                preds = model.predict(encodings)
                # reshape the predictions to match the label shape
                pred_reshaped = preds.reshape(label.shape)

                # create 10 bins for multi-class classification
                multi_class_bins = np.linspace(0.1, 1, 9)
                binned_preds_np = np.digitize(preds, bins=multi_class_bins)
                binned_targets_np = np.digitize(label.flatten(), bins=multi_class_bins)

                results_per_image.append({
                    f"overall_accuracy": accuracy_score(binned_targets_np, binned_preds_np),
                    f"balanced_accuracy": balanced_accuracy_score(binned_targets_np, binned_preds_np),
                    f"recall": recall_score(binned_targets_np, binned_preds_np, average='weighted'),
                    f"precision": precision_score(binned_targets_np, binned_preds_np, average='weighted'),
                    f"f1": f1_score(binned_targets_np, binned_preds_np, average='weighted'),
                }
                    )
                predictions.append(pred_reshaped)
                targets.append(label)

        preds = np.array(predictions)
        target = np.array(targets)
                
        if sort_for == "overall_accuracy":
            sorted_indices = np.argsort([res["overall_accuracy"] for res in results_per_image])
        elif sort_for == "balanced_accuracy":
            sorted_indices = np.argsort([res["balanced_accuracy"] for res in results_per_image])
        else:
            raise ValueError(f"Unknown sort_for value: {sort_for}")

        # reorders predictions and targets based on increasing accuracy (worst to best)
        preds = preds[sorted_indices]
        target = target[sorted_indices]

        for i in range(num_images):
            # save predictions and targets with three lowest accuracies
            filename = test_ds.input_tifs[sorted_indices[i]].name
            print(f"Processing {filename} with index {sorted_indices[i]}", flush=True)
            
            pred_to_save = preds[i]
            target_to_save = target[i]

            acc = results_per_image[sorted_indices[i]]["overall_accuracy"]

            # save the predictions as numpy
            np.save(
                prediction_folder / f"{filename}_{acc}_prediction.npy",
                pred_to_save,
            )
            np.save(
                prediction_folder / f"{filename}_{acc}_target.npy",
                target_to_save,
            )
            print(f"Saved predictions for {filename} with overall accuracy: {acc}", flush=True)

        for i in range(num_images):
            # now save predictions and targets with highest accuracy
            filename = test_ds.input_tifs[sorted_indices[-(i + 1)]].name
            print(f"Processing {filename} with index {sorted_indices[-(i + 1)]}", flush=True)

            pred_to_save = preds[-(i + 1)]
            target_to_save = target[-(i + 1)]

            acc = results_per_image[sorted_indices[-(i + 1)]]["overall_accuracy"]

            # save the predictions as numpy
            np.save(
                prediction_folder / f"{filename}_{acc}_prediction.npy",
                pred_to_save,
            )
            np.save(
                prediction_folder / f"{filename}_{acc}_target.npy",
                target_to_save,
            )
            print(f"Saved predictions for {filename} with overall accuracy: {acc}", flush=True)
        

    @torch.no_grad()
    def _visualize_predictions(
        self, 
        model: EncoderWithHead,
        log_wandb: bool = False,
    ):
        vis_ds = LandsatEvalDataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            split="visualize",
            data_config=self.data_config
        )

        visualization_folder = DATA_FOLDER / "visualizations"
        if not visualization_folder.exists():
            visualization_folder.mkdir(parents=True, exist_ok=True)

        if self.normalization == "std":
            normalizing_dict = vis_ds.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
            print(normalizing_dict, flush=True)
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
            vis_ds.normalizer = normalizer
        else:
            normalizer = Normalizer(std=False)
            vis_ds.normalizer = normalizer

        vis_dl = DataLoader(
            vis_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for masked_output, labels, filename in tqdm(vis_dl, desc="Predicting visualization images"):
                (
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
                    months,
                ) = [t.to(device) for t in masked_output]

                #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(
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
                    months,
                    patch_size_high_res=10,
                    patch_size_med_res=1,
                    patch_size_low_res=1,
                )

                # check that all predictions are between 0 and 1
                assert logits.min() >= 0 and logits.max() <= 1

                spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                preds_2D = rearrange(
                    torch.squeeze(logits),
                    "(h w) -> h w",
                    h=spatial_patches_per_dim,
                    w=spatial_patches_per_dim,
                ).float().cpu().numpy()
                labels = labels.float().cpu().numpy()
                # squeeze labels if needed
                if len(labels.shape) == 3:
                    labels = np.squeeze(labels, axis=0)

                r2 = r2_score(labels.flatten(), preds_2D.flatten())
                rmse = root_mean_squared_error(labels.flatten(), preds_2D.flatten())

                if log_wandb:
                    import wandb
                    import matplotlib.pyplot as plt

                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(preds_2D, cmap="gray", vmin=0, vmax=1)
                    axs[0].set_title("Predictions")
                    axs[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(np.abs(preds_2D - labels), cmap="coolwarm", vmin=0, vmax=1)
                    axs[2].set_title("Absolute Error")
                    fig.colorbar(axs[2].images[0], ax=axs[2], orientation='vertical')
                    #plt.savefig(f"visualizations/{filename}_r2_{r2}_rmse_{rmse}.png")

                    filename = filename[0].split(".tif")[0]

                    wandb.init(entity="sea-ice", project="ai4snow-finetune")
                    wandb.log(
                        {
                            f"{self.name}_visualization_{filename}_r2_{r2}_rmse_{rmse}": wandb.Image(
                                fig,
                                caption=f"R2: {r2:.4f}, RMSE: {rmse:.4f}, Lat: {filename.split('_')[3]}, Lon: {filename.split('_')[4]}, Date: {filename.split('_')[1]}",
                            )
                        }
                    )
                    plt.close(fig)

                # save the predictions as numpy
                np.save(
                    visualization_folder / f"{filename}_r2_{r2}_rmse_{rmse}.npy",
                    preds_2D,
                )
                print(f"Saved predictions for {filename} with R2: {r2} and RMSE: {rmse}", flush=True)


    def make_weights_for_balanced_classes(self, train_ds, nclasses):
        """
        Computes a weight for each sample based on the frquency of its mean class per image, binned into nclasses classes.
        """
        n_images = len(train_ds)
        print(f"Number of images: {n_images}")
        count_per_class = [0] * nclasses
        for _, target, _ in train_ds:
            mean_per_image = np.mean(target)
            # bin the mean value into one of nclasses classes
            # 0.0 will be in class 0, 1.0 in class nclasses-1, 0.99 in class nclasses-2
            multi_class_bins = np.linspace(0.1, 1, nclasses-1)
            binned_targets_np = np.digitize(mean_per_image, bins=multi_class_bins)
            count_per_class[binned_targets_np] += 1
        weight_per_class = [0.] * nclasses
        for i in range(nclasses):
            weight_per_class[i] = float(n_images) / float(count_per_class[i])
        weights = [0] * n_images
        for idx, (_, target, _) in enumerate(train_ds):
            mean_per_image = np.mean(target)
            # bin the mean value into one of nclasses classes
            multi_class_bins = np.linspace(0.1, 1, nclasses-1)
            binned_targets_np = np.digitize(mean_per_image, bins=multi_class_bins)
            weights[idx] = weight_per_class[binned_targets_np]
        return weights


    def evaluate_model_on_task(
        self, 
        pretrained_model: Encoder, 
        model_modes: Optional[List[str]] = None, 
        baseline_galileo: bool = False, 
        log_wandb: bool = False, 
        hyperparams_config: Optional[Dict] = None, 
        initialization_id: Optional[str] = None, 
        sweep_run = None,
        save_final_checkpoint: bool = False,
    ) -> Dict:
        
        # TODO: refine naming of eval config
        assert self.decoder_mode in ["finetune", "linear_probe", "attention_probe", "sklearn"], f"Unknown evaluation mode: {self.decoder_mode}"
        if self.decoder_mode == "finetune":
            eval_config = self.eval_config["finetune"]
        elif self.decoder_mode == "linear_probe":
            eval_config = self.eval_config["linear_probe"]
        elif self.decoder_mode == "attention_probe":
            eval_config = self.eval_config["attention_probe"]
        elif self.decoder_mode == "sklearn":
            eval_config = None

        if hyperparams_config is None:
            hyperparams_config = self.eval_config["hyperparams"]

        if initialization_id is not None:
            hyperparams_config["initialization_id"] = initialization_id

        BATCH_SIZE = hyperparams_config.get("batch_size", 16)
        NUM_WORKERS = hyperparams_config.get("num_workers", 4)

        if baseline_galileo:
            from src.eval.landsat_baselines import LandsatEvalDatasetGalileo, GalileoNormalizer, galileo_config_dir, GALILEO_NORMALIZATION_DICT_FILENAME
            train_ds = LandsatEvalDatasetGalileo(
                exclude_prediction_date=self.exclude_prediction_date,
                exclude_prediction_high_res=self.exclude_prediction_high_res,
                split="train",
            )
            if self.normalization == "std":
                normalizer=GalileoNormalizer(std=True,
                                             normalizing_dicts=train_ds.load_normalization_values(galileo_config_dir / GALILEO_NORMALIZATION_DICT_FILENAME), 
                                             std_multiplier=2)
                print(train_ds.load_normalization_values(galileo_config_dir / GALILEO_NORMALIZATION_DICT_FILENAME), flush=True)
            else:
                normalizer = GalileoNormalizer(std=False)
            train_ds.normalizer = normalizer
        else:
            train_ds = LandsatEvalDataset(
                exclude_prediction_date=self.exclude_prediction_date,
                exclude_prediction_high_res=self.exclude_prediction_high_res,
                split="train",
                data_config=self.data_config
            )

            if self.normalization == "std":
                normalizing_dict = train_ds.load_normalization_values(
                    path=config_dir / NORMALIZATION_DICT_FILENAME
                )
                print(normalizing_dict, flush=True)
                normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
            else:
                normalizer = Normalizer(std=False)
            train_ds.normalizer = normalizer
        
        if self.resample:
            from torch.utils.data import WeightedRandomSampler
            # oversample the dataset to have a uniform distribution of mean class values per image
            weights = self.make_weights_for_balanced_classes(train_ds, nclasses=10)
            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_dl = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
            )
        else:
            train_dl = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
            )

        if self.decoder_mode == "sklearn":
            if model_modes is None:
                model_modes = self.all_regression_sklearn_models
            for model_mode in model_modes:
                assert model_mode in self.all_regression_sklearn_models
            assert save_final_checkpoint == False, "Cannot save final checkpoint when using sklearn evaluation mode."
            trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes, baseline_galileo=baseline_galileo)
            results = self._evaluate_model(pretrained_model, trained_sklearn_models, baseline_galileo=baseline_galileo, hyperparams_config=hyperparams_config) 

        elif self.decoder_mode in ["finetune", "linear_probe", "attention_probe"]:
            test_dl = self.get_test_dl(hyperparams_config=hyperparams_config, baseline_galileo=baseline_galileo)
            loaders_dict = {"train": train_dl, "test": test_dl}
            results = get_finetune_results(
                loaders_dict, 
                pretrained_model, 
                num_runs=1, 
                device=device, 
                identifier=self.name, 
                eval_config=eval_config, 
                hyperparams_config=hyperparams_config, 
                num_finetune_epochs=self.num_finetune_epochs, 
                baseline_galileo=baseline_galileo, 
                log_wandb=log_wandb, 
                sweep_run=sweep_run,
                save_final_checkpoint=save_final_checkpoint
            )
        else:
            raise ValueError(f"Unknown evaluation mode: {self.decoder_mode}")

        return results
    
    def visualize_sample_predictions(
        self,
        model: EncoderWithHead, 
        log_wandb: bool = False, 
    ):
        self._visualize_predictions(model, log_wandb=log_wandb)
