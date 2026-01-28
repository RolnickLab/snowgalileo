import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, cast

import joblib
import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import rearrange
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.config import (
    DATA_FOLDER,
    MODALITIES,
    MODIS_FILL_VALUE,
    NDI_VALID_DATA_BOUNDS,
    NO_DATA_VALUE,
    NORMALIZATION_DICT_FILENAME,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_TIMESTEPS,
    RESULTS_FOLDER,
)
from src.data.dataset import Dataset as BaseDataset
from src.data.dataset import DatasetOutput, Normalizer, to_cartesian
from src.data.earthengine.eo_eval import (
    CLOUD_BANDS,
    EE_SPACE_BANDS,
    EE_WC_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    EO_SPACE_TIME_LOW_RES_BANDS,
    ESA_WORLDCOVER_BAND_INDEX,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BANDS,
    TIME_BANDS_GROUPS_IDX,
)
from src.eval.downstream_augmentation import DownstreamAugmentation
from src.eval.eval import EvalTask, model_class_name
from src.eval.metrics import compute_classification_metrics, compute_regression_metrics
from src.eval.patch_predict import EncoderWithHead, evaluate_seg, get_finetune_results_on_val_set
from src.masking import _aggregate_mask_per_channel_group
from src.snowgalileo import Encoder
from src.utils import DEFAULT_SEED, config_dir, device, masked_output_np_to_tensor

logger = logging.getLogger("__main__")


class LandsatEvalDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        h5pys_only: bool = False,
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        normalizer: Optional[Normalizer] = None,
        augmentation=DownstreamAugmentation(False),
        data_config: Dict = {},
    ):
        super().__init__(
            data_folder=DATA_FOLDER / data_config["input_tif_folder"] / split,
            download=False,
            h5py_folder=DATA_FOLDER / data_config["input_h5py_folder"] / split,
            h5pys_only=h5pys_only,
            normalizer=normalizer,
        )

        self.split = split
        assert self.split in ["train", "test", "visualize", "inference"]

        self.augmentation = augmentation

        # whether to exclude the prediction date from the input timesteps
        # if True, the prediction date will be masked out in the input
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.exclude_prediction_sensors = exclude_prediction_sensors
        self.label_height_width = data_config["input_height_width"]

        self.input_tif_folder = DATA_FOLDER / data_config["input_tif_folder"] / self.split

        self.label_folder = DATA_FOLDER / data_config["label_folder"] / self.split

        if self.split not in ["visualize", "inference"]:
            self.h5py_folder = DATA_FOLDER / data_config["input_h5py_folder"] / self.split
            if self.h5py_folder is not None:  # for mypy to pass
                self.h5py_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.h5py_folder = None

        input_tifs = list(self.input_tif_folder.glob("*.tif")) + list(
            self.input_tif_folder.glob("*.tiff")
        )
        self.tifs = self._sanity_check(input_tifs)
        self.tifs.sort(key=lambda p: p.name)

        self.h5pys: list = []

        label_tifs = list(self.label_folder.glob("*.tif")) + list(self.label_folder.glob("*.tiff"))
        self.label_tifs: List[Path] = self._sanity_check(label_tifs)
        self.label_tifs.sort(key=lambda p: p.name)

        self.pairs = []

        if h5pys_only:
            assert self.h5py_folder is not None, "Can't use h5pys only if there is no cache folder"
            self.tifs: List[Path] = []
            self.h5pys = list(self.h5py_folder.glob("*.h5"))
            self.h5pys.sort(key=lambda p: p.name)

            for img, lbl in zip(self.h5pys, self.label_tifs):
                if img.name.split(".")[0] == lbl.name.split(".")[0]:
                    self.pairs.append((img, lbl))
                else:
                    print(f"Skipping mismatched pair: {img.name}, {lbl.name}")
        else:
            for img, lbl in zip(self.tifs, self.label_tifs):
                if img.name == lbl.name:
                    self.pairs.append((img, lbl))
                else:
                    print(f"Skipping mismatched pair: {img.name}, {lbl.name}")

    @classmethod
    def _sanity_check(cls, tifs):
        checked_tifs: List[Path] = []
        for tif in tifs:
            try:
                _ = cls.prediction_month_from_file(tif)
                checked_tifs.append(tif)
            except IndexError:
                warnings.warn(f"IndexError for {tif}")
        return checked_tifs

    # NOTE: overwritten from TifDataset since the eval tif files have different naming conventions
    @classmethod
    def prediction_month_from_file(cls, tif_path: Path) -> int:
        # assumes the tif file name is in the format "LC09_YYYYMMDD_[FSC]_[lat]_[lon].tif"
        prediction_month = int(tif_path.name.split("_")[1][4:6])
        return prediction_month

    def mask_prediction_timestep(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # NOTE: space-only and static data are kept as is
        # 0 = valid, 1 = masked
        assert self.exclude_prediction_date
        s_t_h_m[:, :, -1, :] = 1
        s_t_m_m[:, :, -1, :] = 1
        s_t_l_m[:, :, -1, :] = 1
        t_m[-1, :] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m
    
    def mask_prediction_sensor_data(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # Masks all sensor channel groups in the prediction timestep
        # This includes all Sentinel-1, Sentinel-2, Landsat, Sentinel-3, MODIS, VIIRS data, as well as the MODIS-derived indeces
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_sensors
        assert t_m.shape[-1] == len(TIME_BANDS_GROUPS_IDX)
        s_t_h_m[:, :, -1, :] = 1
        s_t_m_m[:, :, -1, :] = 1
        s_t_l_m[:, :, -1, :] = 1
        t_m[-1, :-1] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    def mask_prediction_high_res(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # Masks the high resolution, optical channel groups in the prediction timestep
        # This includes all Sentinel-2 and Landsat bands
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_high_res
        assert s_t_h_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)
        # Keep the first channel group (Sentinel-1)
        s_t_h_m[:, :, -1, 1:] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

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
            assert (ndsi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDSI for {tif_path}"
            )
            assert (
                (ndsi >= NDI_VALID_DATA_BOUNDS[0]) & (ndsi <= NDI_VALID_DATA_BOUNDS[1])
                | (ndsi == NO_DATA_VALUE)
            ).all(), f"NDI values out of bounds {NDI_VALID_DATA_BOUNDS} for {tif_path}"

        # NDVI = (NIR - Red) / (NIR + Red)
        if MODALITIES["ndvi"].get("active"):
            ndvi = cls.calculate_ndi(
                space_time_low_res_x, band_1="sur_refl_b02", band_2="sur_refl_b01"
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndvi), axis=-1)
            assert (ndvi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDVI for {tif_path}"
            )
            assert (
                (ndvi >= NDI_VALID_DATA_BOUNDS[0]) & (ndvi <= NDI_VALID_DATA_BOUNDS[1])
                | (ndvi == NO_DATA_VALUE)
            ).all(), f"NDI values out of bounds {NDI_VALID_DATA_BOUNDS} for {tif_path}"

        space_x = rearrange(
            values[-len(EE_SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = cls._check_and_fillna(space_x, np.array(EE_SPACE_BANDS))

        # one-hot encode ESA Worldcover band
        esa_wc = cls.one_hot_encode_esa_worldcover(space_x[:, :, ESA_WORLDCOVER_BAND_INDEX])
        assert esa_wc.all() in [0, 1, NO_DATA_VALUE], (
            f"Unexpected values in ESA Worldcover for {tif_path}"
        )
        space_x = np.concatenate((space_x[:, :, : (-len(EE_WC_BANDS))], esa_wc), axis=-1)

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

    def _tif_to_array_with_checks(self, idx) -> DatasetOutput:
        tif_path, _ = self.pairs[idx]
        try:
            dataset = self._tif_to_array(tif_path)
            return dataset
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.pairs[idx] = self.pairs[new_idx]
            tif_path = self.pairs[idx][0]
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
            image_path, _ = self.pairs[idx]
            h5py_path = self.tif_to_h5py_path(image_path)
            if h5py_path.exists():
                try:
                    return self.read_and_slice_h5py_file(h5py_path)
                except Exception as e:
                    logger.warn(f"Exception {e} for {image_path}")
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
                        image_path.stem,
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
                    image_path.stem,
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

    def __getitem__(self, idx):
        if self.h5pys_only:
            h5py_path, _ = self.pairs[idx]
            image = self.read_and_slice_h5py_file(h5py_path)
        else:
            image = self.load_tif(idx)

        if self.normalizer is None:
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
            ) = image

        else:
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
            ) = image.normalize(self.normalizer)

        # unmask everything per default, then mask invalid data
        # 0 = valid data, 1 = masked
        s_t_h_m = torch.zeros(
            (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX),
            )
        )
        s_t_m_m = torch.zeros(
            (
                NUM_MED_RES_PIXELS_PER_DIM,
                NUM_MED_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX),
            )
        )
        s_t_l_m = torch.zeros(
            (
                NUM_LOW_RES_PIXELS_PER_DIM,
                NUM_LOW_RES_PIXELS_PER_DIM,
                self.output_timesteps,
                len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX),
            )
        )
        sp_m = torch.zeros(
            (self.output_hw_high_res, self.output_hw_high_res, len(SPACE_BAND_GROUPS_IDX))
        )
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

        # Apply the invalid data masks
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

        if self.exclude_prediction_sensors:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_sensor_data(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        """
        if self.split == "inference":
            # return input tif instead of label in inference mode
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
                self.tifs[idx],
            )
        """

        image_path, label_path = self.pairs[idx]
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label_path)) as data:
            label = cast(np.ndarray, data.values)
            # remove first dimension (for shape consistency)
            label = np.squeeze(label, axis=0)

        assert image_path.name.split(".")[0] == label_path.name.split(".")[0], (
            f"Input path {image_path.name} and label path {label_path.name} do not match."
        )

        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            month,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            label,
        ) = self.augmentation.apply(
            torch.as_tensor(s_t_h_x),
            torch.as_tensor(s_t_m_x),
            torch.as_tensor(s_t_l_x),
            torch.as_tensor(sp_x),
            torch.as_tensor(t_x),
            torch.as_tensor(st_x),
            torch.as_tensor(month),
            torch.as_tensor(s_t_h_m),
            torch.as_tensor(s_t_m_m),
            torch.as_tensor(s_t_l_m),
            torch.as_tensor(sp_m),
            torch.as_tensor(t_m),
            torch.as_tensor(st_m),
            torch.as_tensor(label),
        )

        # for inference mode, return full filepath to image not only the filename
        if self.split == "inference":
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
                str(image_path),
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
            image_path.name,  # for logging purposes
        )

    def __len__(self) -> int:
        return len(self.pairs)


class LandsatEval(EvalTask):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        patch_size_high_res: int = 10,
        h5pys_only: bool = False,
        seed=DEFAULT_SEED,
        resample: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = {},
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.exclude_prediction_sensors = exclude_prediction_sensors
        self.patch_size_high_res = patch_size_high_res
        self.resample = resample
        self.num_finetune_epochs = num_finetune_epochs
        self.decoder_mode = decoder_mode
        self.h5pys_only = h5pys_only

        super().__init__(self.patch_size_high_res, seed)
        name_id = f"{eval_config['name']}" if eval_config and "name" in eval_config else ""
        self.name = f"{'attn' if self.decoder_mode == 'attention_probe' else 'linear' if self.decoder_mode == 'linear_probe' else 'finetune' if self.decoder_mode == 'finetune' else 'sklearn'}{'_exclude_prediction_date_' if self.exclude_prediction_date else ''}{'_exclude_prediction_sensors_' if self.exclude_prediction_sensors else ''}{'_no_high_res_in_pred_date' if self.exclude_prediction_high_res else ''}{name_id}"
        self.eval_config = eval_config
        self.data_config = self.eval_config["data"]

    @staticmethod
    def _get_dataset(
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        exclude_prediction_sensors: bool,
        split: str,
        augmentation,
        h5pys_only: bool = False,
        data_config: Dict = {},
        normalization: Union[str, Normalizer] = "std"
    ) -> LandsatEvalDataset:
        
        ds = LandsatEvalDataset(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            split=split,
            h5pys_only=h5pys_only,
            augmentation=augmentation,
            data_config=data_config,
        )

        if normalization == "std":
            normalizing_dict = ds.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        else:
            normalizer = Normalizer(std=False)
        ds.normalizer = normalizer

        return ds

    def get_test_dl(self, hyperparameter_config=None, return_ds=False):
        test_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="test",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(False),
            normalization=self.normalization
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        if return_ds:
            return test_ds, test_dl
        return test_dl

    @torch.no_grad()
    def _evaluate_trained_sklearn_model(
        self,
        pretrained_model: Encoder,
        sklearn_model: BaseEstimator,
        hyperparameter_config=None,
    ) -> Dict:
        prediction_folder = DATA_FOLDER / "predictions"
        if not prediction_folder.exists():
            prediction_folder.mkdir(parents=True, exist_ok=True)

        test_dl = self.get_test_dl(hyperparameter_config=hyperparameter_config)

        pred_dict: Dict[str, BaseEstimator] = {model_class_name(sklearn_model): []}
        pred_list = []

        encodings_list = []
        labels_list = []

        for masked_output, label, _ in tqdm(test_dl, desc="Computing test predictions"):
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

        preds = sklearn_model.predict(encodings_np)
        pred_dict[model_class_name(sklearn_model)].append(preds)
        pred_list.append(preds)

        preds_np = np.concatenate(pred_list)
        majority_baseline_np = np.zeros_like(preds_np)

        # mask for computing metrics without boundary values
        mask = (targets_np > 0) & (targets_np < 1)
        all_labels_1D_f = targets_np[mask]
        all_preds_1D_f = preds_np[mask]

        # create 10 bins for multi-class classification
        multi_class_bins = np.linspace(0.1, 1, 9)
        binned_preds_np = np.digitize(preds_np, bins=multi_class_bins)
        binned_targets_np = np.digitize(targets_np, bins=multi_class_bins)

        binned_preds_np_f = np.digitize(all_preds_1D_f, bins=multi_class_bins)
        binned_targets_np_f = np.digitize(all_labels_1D_f, bins=multi_class_bins)

        results: Dict = {
            "model": {},
            "baseline": {
                "majority": {},
                "balanced": {},
            },
        }

        for model_name_str, pred_list in pred_dict.items():
            results["model"]["regression"] = compute_regression_metrics(preds_np, targets_np)

            results["baseline"]["majority"]["regression"] = compute_regression_metrics(
                majority_baseline_np, targets_np
            )

            results["baseline"]["balanced"]["regression"] = compute_regression_metrics(
                all_preds_1D_f, all_labels_1D_f
            )

            results["model"]["classification"] = compute_classification_metrics(
                binned_preds_np, binned_targets_np
            )

            results["baseline"]["majority"]["classification"] = compute_classification_metrics(
                majority_baseline_np, binned_targets_np
            )

            results["baseline"]["balanced"]["classification"] = compute_classification_metrics(
                binned_preds_np_f, binned_targets_np_f
            )

        np.save(
            prediction_folder / "predictions_final.npy",
            preds_np,
        )

        return results

    # TODO: adjust to also work with attention probe and linear head mode
    @torch.no_grad()
    def _visualize_best_worst(
        self,
        pretrained_model: Encoder,
        sklearn_models: Sequence[BaseEstimator],
        num_images: int = 50,
        sort_for: str = "overall_accuracy",
        hyperparameter_config=None,
    ):
        prediction_folder = DATA_FOLDER / "ascending_accuracy_predictions"
        if not prediction_folder.exists():
            prediction_folder.mkdir(parents=True, exist_ok=True)

        test_ds, test_dl = self.get_test_dl(
            hyperparameter_config=hyperparameter_config,
            return_ds=True,
        )

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

            for model in sklearn_models:
                preds = model.predict(encodings.cpu().numpy())
                # reshape the predictions to match the label shape
                pred_reshaped = preds.reshape(label.shape)

                # create 10 bins for multi-class classification
                multi_class_bins = np.linspace(0.1, 1, 9)
                binned_preds_np = np.digitize(preds, bins=multi_class_bins)
                binned_targets_np = np.digitize(label.flatten(), bins=multi_class_bins)

                results_per_image.append(
                    {
                        "overall_accuracy": accuracy_score(binned_targets_np, binned_preds_np),
                        "balanced_accuracy": balanced_accuracy_score(
                            binned_targets_np, binned_preds_np
                        ),
                        "recall": recall_score(
                            binned_targets_np, binned_preds_np, average="weighted"
                        ),
                        "precision": precision_score(
                            binned_targets_np, binned_preds_np, average="weighted"
                        ),
                        "f1": f1_score(binned_targets_np, binned_preds_np, average="weighted"),
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
    def _predict_and_store_output(
        self, model: EncoderWithHead, id: str, log_wandb: bool = True, eval_config: str = ""
    ):
        inference_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="inference",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(False),
            normalization=self.normalization
        )

        output_tif_folder = DATA_FOLDER / "output_tifs" / eval_config
        if not output_tif_folder.exists():
            output_tif_folder.mkdir(parents=True, exist_ok=True)

        output_npy_folder = DATA_FOLDER / "output_png" / eval_config
        if not output_npy_folder.exists():
            output_npy_folder.mkdir(parents=True, exist_ok=True)

        inference_dl = DataLoader(
            inference_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for masked_output, labels, filepath in tqdm(inference_dl, desc="Predicting output"):
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

                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
                preds_2D = (
                    rearrange(
                        torch.squeeze(logits),
                        "(h w) -> h w",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                # upsample to resolution of input tif for storage in the same TIF
                preds_up = np.repeat(np.repeat(preds_2D, 10, axis=0), 10, axis=1)

                # unpack filepath from batch dimension
                filepath = Path(filepath[0])
                filename = filepath.stem

                with cast(xr.Dataset, rioxarray.open_rasterio(filepath)) as data:
                    stack = np.concatenate([data.values, preds_up[None, :, :]], axis=0)

                    new_band = np.arange(1, stack.shape[0] + 1)
                    new = xr.DataArray(
                        stack,
                        dims=data.dims,
                        coords={
                            "band": new_band,
                            "y": data.coords["y"],
                            "x": data.coords["x"],
                        },
                        attrs=data.attrs,
                    )

                    new = new.rio.write_crs(data.rio.crs)
                    new = new.rio.write_transform(data.rio.transform())

                new.rio.to_raster(output_tif_folder / f"{filename}_with_preds.tif")

                # also save the predictions as numpy
                np.save(
                    output_npy_folder / f"{filename}_output.npy",
                    preds_2D,
                )

                labels = labels.float().cpu().numpy()
                # squeeze labels if needed
                if len(labels.shape) == 3:
                    labels = np.squeeze(labels, axis=0)

                r2 = r2_score(labels.flatten(), preds_2D.flatten())
                rmse = root_mean_squared_error(labels.flatten(), preds_2D.flatten())

                if log_wandb:
                    import matplotlib.pyplot as plt
                    import wandb

                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(preds_2D, cmap="gray", vmin=0, vmax=1)
                    axs[0].set_title("Predictions")
                    axs[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(np.abs(preds_2D - labels), cmap="coolwarm", vmin=0, vmax=1)
                    axs[2].set_title("Absolute Error")
                    fig.colorbar(axs[2].images[0], ax=axs[2], orientation="vertical")
                    # plt.savefig(f"visualizations/{filename}_r2_{r2}_rmse_{rmse}.png")

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

                print(f"Saved predictions for {filename}", flush=True)

    @torch.no_grad()
    def _evaluate_model(self, model: EncoderWithHead, log_wandb: bool = True):
        test_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="test",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(False),
            normalization=self.normalization
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        results = evaluate_seg(data_loader=test_dl, finetuned_model=model, device=device)

        if log_wandb:
            import wandb

            wandb.init(entity="sea-ice", project="ai4snow-finetune")
            wandb.log(results)

            def flatten_for_summary(d, prefix=""):
                out = {}
                for k, v in d.items():
                    key = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, dict):
                        out.update(flatten_for_summary(v, key))
                    else:
                        out[key] = v
                return out

            for k, v in flatten_for_summary(results).items():
                wandb.summary[k] = v

    @torch.no_grad()
    def _evaluate_individual_samples(
        self,
        model: EncoderWithHead,
        id: str,
    ):
        test_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="test",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(False),
            normalization=self.normalization
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # create a csv to store results
        results_csv_path = RESULTS_FOLDER / f"evaluation_results_{id}.csv"
        results_csv_path.touch(exist_ok=True)

        # create header if file is empty
        if results_csv_path.stat().st_size == 0:
            with open(results_csv_path, "w") as f:
                f.write("filename,r2,rmse\n")

        all_preds_2D = []
        all_labels_2D = []

        all_preds_1D = []
        all_labels_1D = []

        with torch.no_grad():
            for masked_output, labels, filename in tqdm(
                test_dl, desc="Predicting visualization images"
            ):
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

                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

                all_preds_1D.append(torch.squeeze(logits).float().cpu().numpy())
                all_labels_1D.append(rearrange(labels, "b h w -> (b h w)").float().cpu().numpy())

                spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                preds_2D = (
                    rearrange(
                        torch.squeeze(logits),
                        "(h w) -> h w",
                        h=spatial_patches_per_dim,
                        w=spatial_patches_per_dim,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
                labels = labels.float().cpu().numpy()
                # squeeze labels if needed
                if len(labels.shape) == 3:
                    labels = np.squeeze(labels, axis=0)

                all_preds_2D.append(preds_2D.flatten())
                all_labels_2D.append(labels.flatten())

                r2 = r2_score(labels.flatten(), preds_2D.flatten())
                rmse = root_mean_squared_error(labels.flatten(), preds_2D.flatten())

                # append results to csv with filename, r2, rmse
                with open(results_csv_path, "a") as f:
                    f.write(f"{filename[0]},{r2},{rmse}\n")

            print(f"Saved predictions for {filename} with R2: {r2}", flush=True)

    @torch.no_grad()
    def _visualize_predictions(
        self,
        model: Union[EncoderWithHead, Encoder],
        log_wandb: bool = False,
        sklearn: bool = False,
        sklearn_models: Optional[List] = None,
    ):
        if sklearn and sklearn_models is None:
            raise ValueError("sklearn_models must be provided when sklearn=True")

        vis_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="visualize",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(False),
            normalization=self.normalization
        )

        visualization_folder = DATA_FOLDER / "visualizations" / str(self.eval_config["name"])
        if not visualization_folder.exists():
            visualization_folder.mkdir(parents=True, exist_ok=True)

        vis_dl = DataLoader(
            vis_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        with torch.no_grad():
            for masked_output, labels, filename in tqdm(
                vis_dl, desc="Predicting visualization images"
            ):
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

                # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

                labels = labels.float().cpu().numpy()
                # squeeze labels if needed
                if len(labels.shape) == 3:
                    labels = np.squeeze(labels, axis=0)

                if not sklearn:
                    # check that all predictions are between 0 and 1
                    assert logits.min() >= 0 and logits.max() <= 1

                    spatial_patches_per_dim = int(logits.shape[1] ** 0.5)
                    preds_2D = (
                        rearrange(
                            torch.squeeze(logits),
                            "(h w) -> h w",
                            h=spatial_patches_per_dim,
                            w=spatial_patches_per_dim,
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )
                else:
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
                    ) = logits
                    encodings = self.group_encodings_per_token(
                        model,
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

                    for sklearn_model in sklearn_models:
                        # TODO: change the type
                        sklearn_model = sklearn_model[0]
                        preds = sklearn_model.predict(encodings)
                        # reshape the predictions to match the label shape
                        preds_2D = preds.reshape(labels.shape)

                r2 = r2_score(labels.flatten(), preds_2D.flatten())
                rmse = root_mean_squared_error(labels.flatten(), preds_2D.flatten())

                if log_wandb:
                    import matplotlib.pyplot as plt
                    import wandb

                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(preds_2D, cmap="gray", vmin=0, vmax=1)
                    axs[0].set_title("Predictions")
                    axs[1].imshow(labels, cmap="gray", vmin=0, vmax=1)
                    axs[1].set_title("Ground Truth")
                    axs[2].imshow(np.abs(preds_2D - labels), cmap="coolwarm", vmin=0, vmax=1)
                    axs[2].set_title("Absolute Error")
                    fig.colorbar(axs[2].images[0], ax=axs[2], orientation="vertical")
                    # plt.savefig(f"visualizations/{filename}_r2_{r2}_rmse_{rmse}.png")

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
                print(
                    f"Saved predictions for {filename} with R2: {r2} and RMSE: {rmse}", flush=True
                )

    @staticmethod
    def make_weights_for_balanced_classes(train_ds, nclasses):
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
            multi_class_bins = np.linspace(0.1, 1, nclasses - 1)
            binned_targets_np = np.digitize(mean_per_image, bins=multi_class_bins)
            count_per_class[binned_targets_np] += 1
        weight_per_class = [0.0] * nclasses
        for i in range(nclasses):
            weight_per_class[i] = float(n_images) / float(count_per_class[i])
        weights = [0] * n_images
        for idx, (_, target, _) in enumerate(train_ds):
            mean_per_image = np.mean(target)
            # bin the mean value into one of nclasses classes
            multi_class_bins = np.linspace(0.1, 1, nclasses - 1)
            binned_targets_np = np.digitize(mean_per_image, bins=multi_class_bins)
            weights[idx] = weight_per_class[binned_targets_np]
        return weights

    def train_and_evaluate_model_on_task(
        self,
        pretrained_model: Encoder,
        model_modes: Optional[List[str]] = None,
        log_wandb: bool = False,
        hyperparameter_config: Optional[Dict] = None,
        initialization_id: Optional[str] = None,
        sweep_run=None,
        save_final_checkpoint: bool = False,
    ) -> Dict:
        assert self.decoder_mode in ["finetune", "linear_probe", "attention_probe", "sklearn"], (
            f"Unknown evaluation mode: {self.decoder_mode}"
        )
        if self.decoder_mode == "finetune":
            eval_config = self.eval_config["finetune"]
        elif self.decoder_mode == "linear_probe":
            eval_config = self.eval_config["linear_probe"]
        elif self.decoder_mode == "attention_probe":
            eval_config = self.eval_config["attention_probe"]
        elif self.decoder_mode == "sklearn":
            eval_config = None

        # optionally passing a hyperparameter config makes hyperparameter sweeping possible
        if hyperparameter_config is None:
            hyperparameter_config = self.eval_config["hyperparameters_snowgalileo"]

        if initialization_id is not None:
            hyperparameter_config["initialization_id"] = initialization_id
        BATCH_SIZE = hyperparameter_config.get("batch_size", 16)
        NUM_WORKERS = hyperparameter_config.get("num_workers", 4)

        train_ds = self._get_dataset(
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            split="train",
            h5pys_only=self.h5pys_only,
            data_config=self.data_config,
            augmentation=DownstreamAugmentation(hyperparameter_config.get("augmentation", False)),
            normalization=self.normalization
        )

        if self.resample:
            from torch.utils.data import WeightedRandomSampler

            # oversample the dataset to have a uniform distribution of mean class values per image
            weights = LandsatEval.make_weights_for_balanced_classes(train_ds, nclasses=10)
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
            trained_sklearn_models = self.train_sklearn_model(
                train_dl, pretrained_model, model_modes
            )

            for idx, sklearn_model in enumerate(trained_sklearn_models):
                if save_final_checkpoint:
                    try:
                        model_path = Path(f"./linear_probe_{idx}.joblib")
                        joblib.dump(trained_sklearn_models, model_path)
                        print(f"Saved sklearn model to {model_path}", flush=True)
                    except Exception as e:
                        print(f"Could not save sklearn model due to {e}", flush=True)
                results = self._evaluate_trained_sklearn_model(
                    pretrained_model,
                    sklearn_model,
                    hyperparameter_config=hyperparameter_config,
                )

        elif self.decoder_mode in ["finetune", "linear_probe", "attention_probe"]:
            test_dl = self.get_test_dl(hyperparameter_config=hyperparameter_config)
            loaders_dict = {"train": train_dl, "test": test_dl}
            results = get_finetune_results_on_val_set(
                loaders_dict,
                pretrained_model,
                num_runs=1,
                device=device,
                identifier=self.name,
                eval_config=eval_config,
                hyperparameter_config=hyperparameter_config,
                num_finetune_epochs=self.num_finetune_epochs,
                log_wandb=log_wandb,
                sweep_run=sweep_run,
                save_final_checkpoint=save_final_checkpoint,
            )
        else:
            raise ValueError(f"Unknown evaluation mode: {self.decoder_mode}")

        return results

    def visualize_sample_predictions(
        self,
        model: Union[EncoderWithHead, Encoder],
        log_wandb: bool = False,
        sklearn: bool = False,
        sklearn_models: Optional[List] = None,
    ):
        self._visualize_predictions(
            model, log_wandb=log_wandb, sklearn=sklearn, sklearn_models=sklearn_models
        )

    def evaluate_model_on_task(self, model: EncoderWithHead):
        self._evaluate_model(model)

    def evaluate_indidvidual_samples(self, model: EncoderWithHead, id: str):
        self._evaluate_individual_samples(model, id=id)

    def predict_and_store_output(self, model: EncoderWithHead, id: str, eval_config: str = ""):
        self._predict_and_store_output(model, id=id, eval_config=eval_config)
