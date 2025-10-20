import json
import re
import warnings
from pathlib import Path
from typing import cast, Optional, Union, Tuple
import logging
import torch
import numpy as np
import rioxarray
import xarray as xr
import h5py
from einops import rearrange, repeat, reduce
from tqdm import tqdm
from typing import NamedTuple, Dict
from copy import deepcopy
from torch.utils.data import DataLoader

from src.data.config import (
    DATA_FOLDER,
    NO_DATA_VALUE,
    CHANNEL_WISE_INVALID_DATA_THRESHOLDS,
    NUM_TIMESTEPS,
    MODALITIES,
    NORMALIZATION_DICT_FILENAME,
)
from src.utils import config_dir
from src.eval.landsat_eval import LandsatEvalDataset, masked_output_np_to_tensor, LandsatEval
from galileo.src.data.dataset import SPACE_BANDS as GALILEO_SPACE_BANDS
from galileo.src.data.dataset import STATIC_BANDS as GALILEO_STATIC_BANDS
from galileo.src.data.dataset import TIME_BANDS as GALILEO_TIME_BANDS
from galileo.src.data.dataset import SPACE_TIME_BANDS as GALILEO_SPACE_TIME_BANDS
from galileo.src.data.dataset import SPACE_BAND_GROUPS_IDX as GALILEO_SPACE_BANDS_GROUPS_IDX
from galileo.src.data.dataset import STATIC_BAND_GROUPS_IDX as GALILEO_STATIC_BANDS_GROUPS_IDX
from galileo.src.data.dataset import TIME_BAND_GROUPS_IDX as GALILEO_TIME_BANDS_GROUPS_IDX
from galileo.src.data.dataset import SPACE_TIME_BANDS_GROUPS_IDX as GALILEO_SPACE_TIME_BANDS_GROUPS_IDX
from galileo.src.data.dataset import NUM_TIMESTEPS as GALILEO_TIMESTEPS
from galileo.src.data.dataset import DATASET_OUTPUT_HW as GALILEO_HW
from galileo.src.data.dataset import SRTM_BANDS, LANDSCAN_BANDS
from galileo.src.data.dataset import (
    SPACE_SHIFT_VALUES as GALILEO_SPACE_SHIFT_VALUES,
    SPACE_DIV_VALUES as GALILEO_SPACE_DIV_VALUES,
    TIME_SHIFT_VALUES as GALILEO_TIME_SHIFT_VALUES,
    TIME_DIV_VALUES as GALILEO_TIME_DIV_VALUES,
    STATIC_SHIFT_VALUES as GALILEO_STATIC_SHIFT_VALUES,
    STATIC_DIV_VALUES as GALILEO_STATIC_DIV_VALUES,
    SPACE_TIME_SHIFT_VALUES as GALILEO_SPACE_TIME_SHIFT_VALUES,
    SPACE_TIME_DIV_VALUES as GALILEO_SPACE_TIME_DIV_VALUES,
)
from galileo.src.data.config import NORMALIZATION_DICT_FILENAME as GALILEO_NORMALIZATION_DICT_FILENAME
from galileo.src.utils import config_dir as galileo_config_dir

from src.eval.landsat_bands import LANDSAT_SPACE_TIME_BANDS, LANDSAT_STATIC_BANDS

from src.data.dataset import Normalizer, to_cartesian
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
)

from torch.utils.data import Dataset as PyTorchDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from typing import Any

# TODO: !!! Change this later because it doesn't match pre-training shape
GALILEO_HW = 100

logger = logging.getLogger("__main__")

with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval_1_99_test.json")).open("r") as f:
    config = json.load(f)
    data_config = config["data"]

LANDSAT_SPACE_TIME_HIGH_RES_BANDS_TO_GALILEO_SPACE_TIME_BANDS = [LANDSAT_SPACE_TIME_BANDS.index(s) for s in GALILEO_SPACE_TIME_BANDS if s in LANDSAT_SPACE_TIME_BANDS]
GALILEO_SPACE_TIME_BANDS_TO_LANDSAT_SPACE_TIME_HIGH_RES_BANDS = [idx for idx, s in enumerate(GALILEO_SPACE_TIME_BANDS) if s in LANDSAT_SPACE_TIME_BANDS]

LANDSAT_STATIC_BANDS_TO_GALILEO_STATIC_BANDS = [LANDSAT_STATIC_BANDS.index(s) for s in GALILEO_STATIC_BANDS if s in LANDSAT_STATIC_BANDS]
GALILEO_STATIC_BANDS_TO_LANDSAT_STATIC_BANDS = [idx for idx, s in enumerate(GALILEO_STATIC_BANDS) if s in LANDSAT_STATIC_BANDS]

# TODO:
# - subset image and label
# - normalizer in eval script
class MaskedOutputGalileo(NamedTuple):
    """
    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """
    space_time_x: torch.Tensor
    space_x: torch.Tensor
    time_x: torch.Tensor
    static_x: torch.Tensor
    space_time_mask: torch.Tensor
    space_mask: torch.Tensor
    time_mask: torch.Tensor
    static_mask: torch.Tensor
    months: torch.Tensor

def masked_output_np_to_tensor_galileo(
    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, month
) -> MaskedOutputGalileo:
    """converts eval task"""
    return MaskedOutputGalileo(
        torch.as_tensor(s_t_x, dtype=torch.float32),
        torch.as_tensor(sp_x, dtype=torch.float32),
        torch.as_tensor(t_x, dtype=torch.float32),
        torch.as_tensor(st_x, dtype=torch.float32),
        torch.as_tensor(s_t_m, dtype=torch.float32),
        torch.as_tensor(sp_m, dtype=torch.float32),
        torch.as_tensor(t_m, dtype=torch.float32),
        torch.as_tensor(st_m, dtype=torch.float32),
        torch.as_tensor(month, dtype=torch.long),
    )


class GalileoDatasetOutput(NamedTuple):
    galileo_space_time_x: np.ndarray
    galileo_space_x: np.ndarray
    galileo_time_x: np.ndarray
    galileo_static_x: np.ndarray
    months: np.ndarray
    galileo_valid_data_mask_s_t: np.ndarray
    galileo_valid_data_mask_sp: np.ndarray
    galileo_valid_data_mask_t: np.ndarray
    galileo_valid_data_mask_st: np.ndarray



class GalileoNormalizer:
    # these are the bands we will replace with the 2*std computation
    # if std = True
    std_bands: Dict[int, list] = {
        len(GALILEO_SPACE_TIME_BANDS): [b for b in GALILEO_SPACE_TIME_BANDS if b != "NDVI"],
        len(GALILEO_SPACE_BANDS): SRTM_BANDS,
        len(GALILEO_TIME_BANDS): GALILEO_TIME_BANDS,
        len(GALILEO_STATIC_BANDS): LANDSCAN_BANDS,
    }

    def __init__(
        self, std: bool = True, normalizing_dicts: Optional[Dict] = None, std_multiplier: float = 2
    ):
        self.shift_div_dict = {
            len(GALILEO_SPACE_TIME_BANDS): {
                "shift": deepcopy(GALILEO_SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(GALILEO_SPACE_TIME_DIV_VALUES),
            },
            len(GALILEO_SPACE_BANDS): {
                "shift": deepcopy(GALILEO_SPACE_SHIFT_VALUES),
                "div": deepcopy(GALILEO_SPACE_DIV_VALUES),
            },
            len(GALILEO_TIME_BANDS): {
                "shift": deepcopy(GALILEO_TIME_SHIFT_VALUES),
                "div": deepcopy(GALILEO_TIME_DIV_VALUES),
            },
            len(GALILEO_STATIC_BANDS): {
                "shift": deepcopy(GALILEO_STATIC_SHIFT_VALUES),
                "div": deepcopy(GALILEO_STATIC_DIV_VALUES),
            },
        }
        print(self.shift_div_dict.keys())
        self.normalizing_dicts = normalizing_dicts
        if std:
            name_to_bands = {
                len(GALILEO_SPACE_TIME_BANDS): GALILEO_SPACE_TIME_BANDS,
                len(GALILEO_SPACE_BANDS): GALILEO_SPACE_BANDS,
                len(GALILEO_TIME_BANDS): GALILEO_TIME_BANDS,
                len(GALILEO_STATIC_BANDS): GALILEO_STATIC_BANDS,
            }
            assert normalizing_dicts is not None
            for key, val in normalizing_dicts.items():
                if isinstance(key, str):
                    continue
                bands_to_replace = self.std_bands[key]
                for band in bands_to_replace:
                    band_idx = name_to_bands[key].index(band)
                    mean = val["mean"][band_idx]
                    std = val["std"][band_idx]
                    min_value = mean - (std_multiplier * std)
                    max_value = mean + (std_multiplier * std)
                    div = max_value - min_value
                    if div == 0:
                        raise ValueError(f"{band} has div value of 0")
                    self.shift_div_dict[key]["shift"][band_idx] = min_value
                    self.shift_div_dict[key]["div"][band_idx] = div

    @staticmethod
    def _normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray) -> np.ndarray:
        x = (x - shift_values) / div_values
        return x

    def __call__(self, x: np.ndarray):
        div_values = self.shift_div_dict[x.shape[-1]]["div"]
        return self._normalize(x, self.shift_div_dict[x.shape[-1]]["shift"], div_values)

class LandsatEvalDatasetGalileo(PyTorchDataset):
    def __init__(
        self,
        split: str = "train",
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        normalizer: Optional[Normalizer] = None,
    ):
        self.split = split
        # whether to exclude the prediction date from the input timesteps
        # if True, the prediction date will be masked out in the input
        self.exclude_prediction_date = exclude_prediction_date
        # if True, only the high resolution optical data (Sentinel-2 and Landsat) will be masked out in the prediction timestep
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.normalizer = normalizer

        assert self.split in ["train", "test", "visualize"]

        self.label_folder = DATA_FOLDER / data_config["label_folder"] / self.split
        self.input_tif_folder = DATA_FOLDER / data_config["input_tif_folder"] / self.split

        if (self.split != "visualize") and (self.split != "test") and (data_config.get("input_h5py_folder") != ""):
            self.h5py_folder = DATA_FOLDER / data_config["input_h5py_folder"] / self.split
        else:
            self.h5py_folder = None

        # NOTE: setting h5py folder to None here for Galileo
        self.h5py_folder = None

        # print the number of label tifs
        print(
            f"Number of label tifs: {len(list(self.label_folder.glob('*.tif')) + list(self.label_folder.glob('*.tiff')))}"
        )

        # print the number of input tifs
        print(
            f"Number of input tifs: {len(list(self.input_tif_folder.glob('*.tif')) + list(self.input_tif_folder.glob('*.tiff')))}"
        )
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

        self.output_hw_high_res = GALILEO_HW
        self.output_timesteps = GALILEO_TIMESTEPS

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

    def mask_prediction_timestep(self, s_t_m, sp_m, t_m, st_m):
        # NOTE: 0 = valid, 1 = masked
        assert self.exclude_prediction_date
        s_t_m[:, :, -1, :] = 1
        t_m[-1, :] = 1
        return s_t_m, sp_m, t_m, st_m
    
    def mask_prediction_high_res(self, s_t_m, sp_m, t_m, st_m):
        # masks the high resolution, optical data in the prediction timestep
        # high resolution channels are: 3 x s1, s2, landsat, so we retain the first 3 channels
        # NOTE: 0 = valid, 1 = masked
        assert self.exclude_prediction_high_res
        assert s_t_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS)
        s_t_m[:, :, -1, 3:] = 1
        return s_t_m, sp_m, t_m, st_m

    @staticmethod
    def create_valid_mask(
        s_t_x, sp_x, t_x, st_x
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        We need to adjust the mask function to account for no data values that occur due to the evaluation-specific export.

        This function will mask out 0 values, and NO_DATA_VALUES that are based on missing sensors.

        0: invalid data
        1: valid data
        """
        print("Creating valid mask for LandsatEvalDataset", flush=True)
        assert s_t_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x_galileo"])
        assert sp_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"])
        assert t_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"])
        assert st_x.shape[-1] == len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"])

        # TODO: assert the amount of 0 values in the input data, to check if they are int or float
        print("Amount of 0 values in s_t_h_x:", np.sum(s_t_x == 0), flush=True)
        print("Amount of 0.0 values in s_t_h_x:", np.sum(s_t_x == 0.0), flush=True)
        print("Amount of NO_DATA values in s_t_h_x:", np.sum(s_t_x == NO_DATA_VALUE), flush=True)

        # start by unmasking invalid data that is characterized by universal no data value
        valid_mask_s_t_h = (s_t_x != NO_DATA_VALUE) & (s_t_x != 0)
        valid_mask_sp = (sp_x != NO_DATA_VALUE) & (sp_x != 0)
        valid_mask_t = (t_x != NO_DATA_VALUE) & (t_x != 0)
        valid_mask_st = (st_x != NO_DATA_VALUE) & (st_x != 0)

        print("Amount of invalid data in s_t_h_x:", np.sum(~valid_mask_s_t_h), flush=True)

        # apply the channel-specific no-data bounds
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["s_t_h_x_galileo"].items():
            valid_mask_s_t_h[..., ch] &= s_t_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["sp_x"].items():
            valid_mask_sp[..., ch] &= sp_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["t_x"].items():
            valid_mask_t[..., ch] &= t_x[..., ch] >= lower_bound
        for ch, lower_bound in CHANNEL_WISE_INVALID_DATA_THRESHOLDS["st_x"].items():
            valid_mask_st[..., ch] &= st_x[..., ch] >= lower_bound

        return (
            valid_mask_s_t_h,
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
    
    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        size: int = GALILEO_HW,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        space_time_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        assert (space_time_x.shape[0] == space_x.shape[0]) & (
            space_time_x.shape[1] == space_x.shape[1]
        )
        possible_h = space_time_x.shape[0] - size
        possible_w = space_time_x.shape[1] - size
        assert (possible_h >= 0) & (possible_w >= 0)

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        return (
            space_time_x[
                start_h : start_h + size,
                start_w : start_w + size,
            ],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x,
            static_x,
        )

    @classmethod
    def _tif_to_array(cls, tif_path: Path) -> GalileoDatasetOutput:
        """
        Loads a spatiotemporal tif file, divides it into different array groups, and creates valid data masks.

        The different array types are:
        space_time_high_res_x: (H, W, T, C_STH)
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
        time_x = dynamic_in_time_x[
            :, :, :, -(len(TIME_BANDS) + len(CLOUD_BANDS)) : -len(CLOUD_BANDS)
        ]
        time_x = np.nanmean(time_x, axis=(0, 1))

        # NDVI = (NIR - Red) / (NIR + Red)
        if MODALITIES["ndvi"].get("active"):
            ndvi = cls.calculate_ndi_high_res(
                space_time_high_res_x, band_1="B8", band_2="B4"
            )
            space_time_high_res_x = np.concatenate((space_time_high_res_x, ndvi), axis=-1)

        space_x = rearrange(
            values[-len(SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = cls._check_and_fillna(space_x, np.array(SPACE_BANDS))

        static_x = to_cartesian(lat, lon)
        static_x = cls._check_and_fillna(static_x, np.array(STATIC_BANDS))

        #space_time_high_res_x, space_x, time_x, static_x = cls.subset_image(
        #    space_time_high_res_x, space_x, time_x, static_x, size=GALILEO_HW
        #)

        (
            valid_data_mask_s_t_h,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = cls.create_valid_mask(
            space_time_high_res_x,
            space_x,
            time_x,
            static_x,
        )

        # NOTE: We initialize with zeros and not with NaNs, because Galileo cannot handle NaNs and we will mask out the invalid data anyway
        galileo_s_t_x = np.zeros((GALILEO_HW, GALILEO_HW, GALILEO_TIMESTEPS, len(GALILEO_SPACE_TIME_BANDS)))
        galileo_sp_x = np.zeros((GALILEO_HW, GALILEO_HW, len(GALILEO_SPACE_BANDS)))
        galileo_t_x = np.zeros((GALILEO_TIMESTEPS, len(GALILEO_TIME_BANDS)))
        galileo_st_x = np.zeros((len(GALILEO_STATIC_BANDS),))

        galileo_months = cls.month_array_from_file(tif_path, int(GALILEO_TIMESTEPS))

        galileo_valid_data_mask_s_t = np.zeros((GALILEO_HW, GALILEO_HW, GALILEO_TIMESTEPS, len(GALILEO_SPACE_TIME_BANDS)))
        # no matching space data and no matching time data, so we can just fill it with zeros (= all invalid)
        galileo_valid_data_mask_sp = np.zeros((GALILEO_HW, GALILEO_HW, len(GALILEO_SPACE_BANDS)))
        galileo_valid_data_mask_t = np.zeros((GALILEO_TIMESTEPS, len(GALILEO_TIME_BANDS)))
        galileo_valid_data_mask_st = np.zeros((len(GALILEO_STATIC_BANDS),))

        # fill galileo bands with matching landsat bands
        # landsat bands have only 8 timesteps, so we need to slice accordingly
        galileo_s_t_x[:, :, :-4, GALILEO_SPACE_TIME_BANDS_TO_LANDSAT_SPACE_TIME_HIGH_RES_BANDS] = space_time_high_res_x[:, :, :, LANDSAT_SPACE_TIME_HIGH_RES_BANDS_TO_GALILEO_SPACE_TIME_BANDS]
        galileo_st_x[GALILEO_STATIC_BANDS_TO_LANDSAT_STATIC_BANDS] = static_x

        galileo_valid_data_mask_s_t[:, :, :-4, GALILEO_SPACE_TIME_BANDS_TO_LANDSAT_SPACE_TIME_HIGH_RES_BANDS] = valid_data_mask_s_t_h[:, :, :, LANDSAT_SPACE_TIME_HIGH_RES_BANDS_TO_GALILEO_SPACE_TIME_BANDS]
        galileo_valid_data_mask_st[GALILEO_STATIC_BANDS_TO_LANDSAT_STATIC_BANDS] = valid_data_mask_st[LANDSAT_STATIC_BANDS_TO_GALILEO_STATIC_BANDS]

        try:
            assert not np.isnan(galileo_s_t_x).any(), f"NaNs in s_t_x for {tif_path}"
            assert not np.isnan(galileo_sp_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(galileo_t_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(galileo_st_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(galileo_s_t_x).any(), f"Infs in s_t_x for {tif_path}"
            assert not np.isinf(galileo_sp_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(galileo_t_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(galileo_st_x).any(), f"Infs in st_x for {tif_path}"
            return (
                galileo_s_t_x,
                galileo_sp_x,
                galileo_t_x,
                galileo_st_x,
                galileo_months,
                galileo_valid_data_mask_s_t,
                galileo_valid_data_mask_sp,
                galileo_valid_data_mask_t,
                galileo_valid_data_mask_st,
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

    def load_tif(self, idx: int) -> GalileoDatasetOutput:
        if self.h5py_folder is None:
            (
                galileo_s_t_x,
                galileo_sp_x,
                galileo_t_x,
                galileo_st_x,
                galileo_months,
                galileo_valid_data_mask_s_t,
                galileo_valid_data_mask_sp,
                galileo_valid_data_mask_t,
                galileo_valid_data_mask_st,
            ) = self._tif_to_array_with_checks(idx)
            return GalileoDatasetOutput(
                galileo_s_t_x,
                galileo_sp_x,
                galileo_t_x,
                galileo_st_x,
                galileo_months,
                galileo_valid_data_mask_s_t,
                galileo_valid_data_mask_sp,
                galileo_valid_data_mask_t,
                galileo_valid_data_mask_st,
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
                        galileo_s_t_x,
                        galileo_sp_x,
                        galileo_t_x,
                        galileo_st_x,
                        galileo_months,
                        galileo_valid_data_mask_s_t,
                        galileo_valid_data_mask_sp,
                        galileo_valid_data_mask_t,
                        galileo_valid_data_mask_st,
                    ) = self._tif_to_array_with_checks(idx)
                    self.save_h5py(
                        galileo_s_t_x,
                        galileo_sp_x,
                        galileo_t_x,
                        galileo_st_x,
                        galileo_valid_data_mask_s_t,
                        galileo_valid_data_mask_sp,
                        galileo_valid_data_mask_t,
                        galileo_valid_data_mask_st,
                        self.input_tifs[idx].stem,
                    )
                    return GalileoDatasetOutput(
                        galileo_s_t_x,
                        galileo_sp_x,
                        galileo_t_x,
                        galileo_st_x,
                        galileo_months,
                        galileo_valid_data_mask_s_t,
                        galileo_valid_data_mask_sp,
                        galileo_valid_data_mask_t,
                        galileo_valid_data_mask_st,
                    )
            else:
                (
                    galileo_s_t_x,
                    galileo_sp_x,
                    galileo_t_x,
                    galileo_st_x,
                    galileo_months,
                    galileo_valid_data_mask_s_t,
                    galileo_valid_data_mask_sp,
                    galileo_valid_data_mask_t,
                    galileo_valid_data_mask_st,
                ) = self._tif_to_array_with_checks(idx)
                self.save_h5py(
                    galileo_s_t_x,
                    galileo_sp_x,
                    galileo_t_x,
                    galileo_st_x,
                    galileo_valid_data_mask_s_t,
                    galileo_valid_data_mask_sp,
                    galileo_valid_data_mask_t,
                    galileo_valid_data_mask_st,
                    self.input_tifs[idx].stem,
                )
                return GalileoDatasetOutput(
                    galileo_s_t_x,
                    galileo_sp_x,
                    galileo_t_x,
                    galileo_st_x,
                    galileo_months,
                    galileo_valid_data_mask_s_t,
                    galileo_valid_data_mask_sp,
                    galileo_valid_data_mask_t,
                    galileo_valid_data_mask_st,
                )

    def save_h5py(
        self,
        s_t_x,
        sp_x,
        t_x,
        st_x,
        valid_data_mask_s_t,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
        tif_stem,
    ):
        assert self.h5py_folder is not None
        with h5py.File(self.h5py_folder / f"{tif_stem}.h5", "w") as hf:
            hf.create_dataset("galileo_s_t_x", data=s_t_x)
            hf.create_dataset("galileo_sp_x", data=sp_x)
            hf.create_dataset("galileo_t_x", data=t_x)
            hf.create_dataset("galileo_st_x", data=st_x)
            hf.create_dataset("galileo_valid_data_mask_s_t", data=valid_data_mask_s_t)
            hf.create_dataset("galileo_valid_data_mask_sp", data=valid_data_mask_sp)
            hf.create_dataset("galileo_valid_data_mask_t", data=valid_data_mask_t)
            hf.create_dataset("galileo_valid_data_mask_st", data=valid_data_mask_st)

    @staticmethod
    def calculate_ndi_high_res(input_array: np.ndarray, band_1: str, band_2: str) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """

        # TODO: make this dynamic instead
        assert band_1 in SPACE_TIME_HIGH_RES_BANDS
        assert band_2 in SPACE_TIME_HIGH_RES_BANDS

        band_1_np = input_array[:, :, :, SPACE_TIME_HIGH_RES_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, SPACE_TIME_HIGH_RES_BANDS.index(band_2)]

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
                    NO_DATA_VALUE,
                ),
                -1,
            )

    def read_and_slice_h5py_file(self, h5py_path: Path):
        with h5py.File(h5py_path, "r") as hf:
            assert hf["galileo_s_t_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(GALILEO_SPACE_TIME_BANDS),
            )
            assert hf["galileo_sp_x"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(GALILEO_SPACE_BANDS),
            )
            assert hf["galileo_t_x"].shape == (self.output_timesteps, len(GALILEO_TIME_BANDS))
            assert hf["galileo_st_x"].shape == (len(GALILEO_STATIC_BANDS),)
            assert hf["galileo_valid_data_mask_s_t"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                self.output_timesteps,
                len(GALILEO_SPACE_TIME_BANDS),
            )
            assert hf["galileo_valid_data_mask_sp"].shape == (
                self.output_hw_high_res,
                self.output_hw_high_res,
                len(GALILEO_SPACE_BANDS),
            )
            assert hf["galileo_valid_data_mask_t"].shape == (self.output_timesteps, len(GALILEO_TIME_BANDS))
            assert hf["galileo_valid_data_mask_st"].shape == (len(GALILEO_STATIC_BANDS),)

            months = self.month_array_from_file(h5py_path, self.output_timesteps)
            output = GalileoDatasetOutput(
                hf["galileo_s_t_x"][:],
                hf["galileo_sp_x"][:],
                hf["galileo_t_x"][:],
                hf["galileo_st_x"][:],
                months,
                hf["galileo_valid_data_mask_s_t"][:],
                hf["galileo_valid_data_mask_sp"][:],
                hf["galileo_valid_data_mask_t"][:],
                hf["galileo_valid_data_mask_st"][:],
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
        # NOTE: input will be a GalileoDatasetOutput object
        h5py = self.load_tif(idx)
        (
            galileo_s_t_x,
            galileo_sp_x,
            galileo_t_x,
            galileo_st_x,
            galileo_month,
            galileo_valid_data_mask_s_t,
            galileo_valid_data_mask_sp,
            galileo_valid_data_mask_t,
            galileo_valid_data_mask_st,
        ) = h5py

        # empty bands should have a mask of one
        galileo_s_t_m = torch.ones((GALILEO_HW, GALILEO_HW, GALILEO_TIMESTEPS, len(GALILEO_SPACE_TIME_BANDS)))
        # no matching space bands between Galileo and SnowGalileo
        galileo_sp_m = torch.ones((GALILEO_HW, GALILEO_HW, len(GALILEO_SPACE_BANDS)))
        # era5 is matching, but not for entire channel group
        galileo_t_m = torch.ones((GALILEO_TIMESTEPS, len(GALILEO_TIME_BANDS)))
        galileo_st_m = torch.ones((len(GALILEO_STATIC_BANDS),))

        # all filled bands should be unmasked
        # the last 4 timesteps have to stay masked because they have no matching data
        galileo_s_t_m[:, :, :-4, GALILEO_SPACE_TIME_BANDS_TO_LANDSAT_SPACE_TIME_HIGH_RES_BANDS] = 0
        galileo_st_m[GALILEO_STATIC_BANDS_TO_LANDSAT_STATIC_BANDS] = 0

        # TODO: We will assume for now that data will be processed with Galileo Loader + collate, and that this works
        # TODO: Test this. Should mask everything that is either not present in the input, or not valid
        galileo_s_t_m = np.logical_or(galileo_s_t_m, np.logical_not(galileo_valid_data_mask_s_t))
        galileo_st_m = np.logical_or(galileo_st_m, np.logical_not(galileo_valid_data_mask_st))

        # turn it from band-space into band-group-space
        galileo_s_t_m = galileo_s_t_m[:, :, :, [g[0] for _, g in GALILEO_SPACE_TIME_BANDS_GROUPS_IDX.items()]]
        galileo_sp_m = galileo_sp_m[:, :, [g[0] for _, g in GALILEO_SPACE_BANDS_GROUPS_IDX.items()]]
        galileo_t_m = galileo_t_m[:, [g[0] for _, g in GALILEO_TIME_BANDS_GROUPS_IDX.items()]]
        galileo_st_m = galileo_st_m[[g[0] for _, g in GALILEO_STATIC_BANDS_GROUPS_IDX.items()]]

        label = self.label_tifs[idx]

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
                f"Label shape {label.shape} does not match expected shape for {label.name}"
            )
            self.label_tifs[idx] = self.label_tifs[idx + 1] if idx < len(self.label_tifs) - 1 else self.label_tifs[idx - 1]
            return self.__getitem__(idx)

        return (
            masked_output_np_to_tensor_galileo(
                self.normalizer(galileo_s_t_x),
                self.normalizer(galileo_sp_x),
                self.normalizer(galileo_t_x),
                self.normalizer(galileo_st_x),
                galileo_s_t_m,
                galileo_sp_m,
                galileo_t_m,
                galileo_st_m,
                galileo_month,
            ),
            label,
            self.input_tifs[idx].name,  # for logging purposes
        )

    def __len__(self) -> int:
        return len(self.label_tifs)


class LandsatEvalDatasetRandomForest(LandsatEvalDataset):
    """
    The Random Forest baseline uses the same dataset as the main LandsatEvalDataset,
    but doesn't group channel masks into channel group masks, so that we can directly
    remove masked channels for the Random Forest input.
    """
    def __init__(
        self,
        split: str = "train",
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        normalizer: Optional[Normalizer] = None,
        data_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            split=split,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            normalizer=normalizer,
            data_config=data_config,
        )

    def mask_prediction_high_res(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # masks the high resolution, optical data in the prediction timestep
        # high resolution channels are: s1, s2, landsat, so we retain the first 3 channels
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_high_res
        assert s_t_h_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS)
        s_t_h_m[:, :, -1, 3:] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    def __getitem__(self, idx):
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

        s_t_h_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_h))
        s_t_m_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_m))
        s_t_l_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_l))
        sp_m = torch.as_tensor(np.logical_not(valid_data_mask_sp))
        t_m = torch.as_tensor(np.logical_not(valid_data_mask_t))
        st_m = torch.as_tensor(np.logical_not(valid_data_mask_st))

        # since the prediction timestep function is channel-independent, we don't have to overwrite it
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
    
class LandsatEvalRandomForest(LandsatEval):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        resample: bool = False,
        eval_config: Dict = None,
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.resample = resample
        self.name = "ls_rf"

        super().__init__(
            normalization=normalization,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            resample=resample,
            eval_config=eval_config,
        )

    def remove_masked_data_and_flatten(
            self,
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
            month
    ):
        # returns: (N) where N is the number of unmasked values
        assert s_t_h_x.shape == s_t_h_m.shape
        assert s_t_m_x.shape == s_t_m_m.shape
        assert s_t_l_x.shape == s_t_l_m.shape
        assert sp_x.shape == sp_m.shape
        assert t_x.shape == t_m.shape
        assert st_x.shape == st_m.shape
        x = torch.cat([
            s_t_h_x.flatten(),
            s_t_m_x.flatten(),
            s_t_l_x.flatten(),
            sp_x.flatten(),
            t_x.flatten(),
            st_x.flatten(),
            month.flatten(),
        ])
        m = torch.cat([
            s_t_h_m.flatten(),
            s_t_m_m.flatten(),
            s_t_l_m.flatten(),
            sp_m.flatten(),
            t_m.flatten(),
            st_m.flatten(),
            torch.zeros_like(month).flatten(),  # month is never masked
        ])
        assert x.shape == m.shape
        return x[m == 0]
    
    def replace_masked_data_with_mean_per_channel(
        self,
        x,
        m,
    ):
        x = torch.masked_fill(x, m.bool(), float('nan'))
        x = torch.where(torch.isnan(x), torch.nanmean(x, dim=-1, keepdim=True), x)

        # if there are still NaNs (all values were masked), replace mean over timesteps (all channels)
        x = torch.where(torch.isnan(x), torch.nanmean(x, dim=-2, keepdim=True), x)

        return x

    # this option ends up in shapes of [B, S, N] where for RF, S is n_samples and n_features.
    # the dimension of N is (C * T) for space-time tokens, C for space tokens, (C * T) for time tokens, and C for static tokens
    # concatenated
    # Next step would we to concatenate along S
    def aggregate_per_output_pixel_and_replace_masked_data(
        self,
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
        replace_with="last"
    ):
        # replace masked data with (A) the last timestep of this sensor, (B) the mean over time of this sensor, (C) zeros, (D) NaNs to be handled by RF.
        # RF computes median for missing values
        # TODO: replace all invalid with mean per timestep
        assert replace_with in ["last", "mean", "zeros", "nan"]
        assert replace_with not in ["last", "mean"], "Not implemented yet"

        # TODO: make this more dynamic
        patch_size_high_res = 10
        p_m = patch_size_high_res // s_t_m_x.shape[1]
        p_l = patch_size_high_res // s_t_l_x.shape[1]

        # first, bring all data into token resolution (the output resolution)
        # t_h = token height, t_w = token width
        s_t_h_x = rearrange(
            reduce(
                s_t_h_x,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="mean",
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_h_m = rearrange(
            reduce(
                s_t_h_m,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="max", # if one value is masked, the entire patch is masked
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        # repeat medium and low resolution tokens over high resolution
        s_t_m_x = rearrange(
            repeat(
                s_t_m_x, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_m, p_w=p_m
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_m_m = rearrange(
            repeat(
                s_t_m_m, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_m, p_w=p_m
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        s_t_l_x = rearrange(
            repeat(
                s_t_l_x, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_l, p_w=p_l
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_l_m = rearrange(
            repeat(
                s_t_l_m, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_l, p_w=p_l
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        sp_x = rearrange(
            reduce(
                sp_x,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="mean",
            ),
            "b t_h t_w c -> b (t_h t_w) c",
        )
        sp_m = rearrange(
            reduce(
                sp_m,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="max", # if one value is masked, the entire patch is masked
        ),
            "b t_h t_w c -> b (t_h t_w) c",
        )

        # repeat time tokens over space
        t_x = repeat(
            t_x, "b t c -> b s t c", s=sp_x.shape[1]
        )
        t_m = repeat(t_m, "b t c -> b s t c", s=sp_x.shape[1])

        st_x = repeat(st_x, "b c -> b s c", s=sp_x.shape[1])
        st_m = repeat(st_m, "b c -> b s c", s=sp_x.shape[1])

        # also include month as a feature, repeat over space
        month = repeat(month, "b c -> b s c", s=sp_x.shape[1])

        assert s_t_h_x.shape[1] == 100

        if replace_with == "mean":
            s_t_h_x = self.replace_masked_data_with_mean_per_channel(s_t_h_x, s_t_h_m)
            s_t_m_x = self.replace_masked_data_with_mean_per_channel(s_t_m_x, s_t_m_m)
            s_t_l_x = self.replace_masked_data_with_mean_per_channel(s_t_l_x, s_t_l_m)
            t_x = self.replace_masked_data_with_mean_per_channel(t_x, t_m)
        elif replace_with == "nan":
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), float('nan'))
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), float('nan'))
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), float('nan'))
            t_x = t_x.masked_fill(t_m.bool(), float('nan'))
        elif replace_with == "zeros":
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), 0.0)
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), 0.0)
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), 0.0)
            t_x = t_x.masked_fill(t_m.bool(), 0.0)
        
        s_t_h_x = rearrange(s_t_h_x, "b s t c -> b s (t c)")
        s_t_h_m = rearrange(s_t_h_m, "b s t c -> b s (t c)")
        s_t_m_x = rearrange(s_t_m_x, "b s t c -> b s (t c)")
        s_t_m_m = rearrange(s_t_m_m, "b s t c -> b s (t c)")
        s_t_l_x = rearrange(s_t_l_x, "b s t c -> b s (t c)")
        s_t_l_m = rearrange(s_t_l_m, "b s t c -> b s (t c)")
        t_x = rearrange(t_x, "b s t c -> b s (t c)")
        t_m = rearrange(t_m, "b s t c -> b s (t c)")

        # assert there are no masked values in space, and static tokens
        assert not torch.any(sp_m.bool()), "Masked space tokens not handled yet."
        assert not torch.any(st_m.bool()), "Masked static tokens not handled yet."
        assert not torch.any(month.bool()), "Masked month tokens not handled yet."

        x = torch.cat([s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x, month], dim=2)  # B, S, N
        m = torch.cat([s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m, torch.zeros_like(month)], dim=2)  # B, S, N

        return x, m

    def fit_random_forest(self):
        train_ds = LandsatEvalDatasetRandomForest(
            split="train",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            data_config=self.data_config,
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

        # TODO: CHECK! does the masking work correctly here? (suspiciously small number of tokens are removed), from [100,311] to (21866,)
        train_dl = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

        all_samples = []
        all_labels = []

        for input, label, _ in train_dl:
            input = torch.squeeze(
                self.aggregate_per_output_pixel_and_remove_masked_data(
                    *input,
                    replace_with="mean"
                ))  # (N, num_features)
            label = torch.squeeze(label).flatten()  # (N,)
            all_samples.append(input)
            all_labels.append(label)

        rf_input = torch.cat(all_samples, dim=0).numpy()
        rf_labels = torch.cat(all_labels, dim=0).numpy()

        import pdb; pdb.set_trace()

        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(rf_input, rf_labels)

if __name__ == "__main__":
    with (Path(__file__).parents[2] / Path("eval_configs") / Path("landsat_eval_5_95.json")).open("r") as f:
        config = json.load(f)
    rf = LandsatEvalRandomForest(
        normalization="std",
        exclude_prediction_date=False,
        exclude_prediction_high_res=False,
        resample=False,
        eval_config=config,
    )
    rf.fit_random_forest()