import json
import re
import warnings
from pathlib import Path
from typing import cast, Optional, Union, Tuple
import logging

import numpy as np
import rioxarray
import xarray as xr
import h5py
from einops import rearrange, repeat

from src.data.config import DATA_FOLDER, NO_DATA_VALUE, CHANNEL_WISE_INVALID_DATA_THRESHOLDS, DATASET_OUTPUT_HW_HIGH_RES, DATASET_OUTPUT_HW_MED_RES, DATASET_OUTPUT_HW_LOW_RES, DATASET_OUTPUT_HW_LOW_RES, NUM_TIMESTEPS, NUM_HIGH_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM, NORMALIZATION_DICT_FILENAME
from src.data.dataset import DatasetOutput, Normalizer, SPACE_BANDS, SPACE_TIME_HIGH_RES_BANDS, SPACE_TIME_MED_RES_BANDS, SPACE_TIME_LOW_RES_BANDS, STATIC_BANDS, TIME_BANDS, EO_ALL_DYNAMIC_IN_TIME_BANDS, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP, MODALITIES, CLOUD_BANDS
from src.data.dataset import EO_SPACE_TIME_LOW_RES_BANDS, to_cartesian
from src.utils import masked_output_np_to_tensor, config_dir

from torch.utils.data import Dataset as PyTorchDataset

logger = logging.getLogger("__main__")

with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval.json")).open("r") as f:
    config = json.load(f)


class LandsatEvalDataset(PyTorchDataset):
    def __init__(self, split: str = "train", exclude_prediction_date: bool = False, normalizer: Optional[Normalizer] = None):
        self.split = split
        # whether to exclude the prediction date from the input timesteps
        # if True, the prediction date will be masked out in the input
        self.exclude_prediction_date = exclude_prediction_date
        self.normalizer = normalizer

        assert self.split in ["train", "valid", "test"]

        self.h5py_folder = DATA_FOLDER / config["input_h5py_folder"]
        self.label_folder = DATA_FOLDER / config["label_folder"]
        self.input_tif_folder = DATA_FOLDER / config["input_tif_folder"]

        # print the number of label tifs
        print(f"Number of label tifs: {len(list(self.label_folder.glob('*.tif')) + list(self.label_folder.glob('*.tiff')))}")

        # print the number of input h5pys
        print(f"Number of input h5pys: {len(list(self.h5py_folder.glob('*.h5py')))}")

        # print the number of input tifs
        print(f"Number of input tifs: {len(list(self.input_tif_folder.glob('*.tif')) + list(self.input_tif_folder.glob('*.tiff')))}")

        ### TODO: replace this by parent class init
        self.cache = True
        self.input_tifs = []
        input_tifs = list(self.input_tif_folder.glob("*.tif")) + list(self.input_tif_folder.glob("*.tiff"))
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

        self.label_tifs = []
        label_tifs = list(self.label_folder.glob("*.tif")) + list(self.label_folder.glob("*.tiff"))
        for tif in label_tifs:
            try:
                _ = self.prediction_month_from_file(tif)
                self.label_tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for label {tif}")

        assert len(self.input_tifs) == len(self.label_tifs), "Number of input tifs and label tifs do not match."
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
                space_time_high_res_x,
                space_time_med_res_x,
                space_time_low_res_x,
                space_x,
                time_x,
                static_x,
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
            self.tifs[idx] = self.input_tifs[new_idx]
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
    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer
    sample = dataset[0]
    print(f"Sample shape: {sample[0].shape}, Label shape: {sample[1].shape}")
    print(f"Prediction month: {dataset.prediction_month_from_file(dataset.label_tifs[0])}")