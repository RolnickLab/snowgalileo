import os
import warnings
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, cast
from typing import OrderedDict as OrderedDictType

import numpy as np
import rioxarray
import xarray as xr
from einops import rearrange, repeat
from torch.utils.data import Dataset as PyTorchDataset

from .config import DATASET_OUTPUT_HW, EE_BUCKET_TIFS, EE_FOLDER_TIFS, NUM_TIMESTEPS
from .earthengine.eo import (
    ALL_DYNAMIC_IN_TIME_BANDS,
    DW_BANDS,
    ERA5_BANDS,
    S1_BANDS,
    SPACE_BANDS,
    SPACE_DIV_VALUES,
    SPACE_SHIFT_VALUES,
    SPACE_TIME_DIV_VALUES,
    SPACE_TIME_SHIFT_VALUES,
    SRTM_BANDS,
    TIME_BANDS,
    TIME_DIV_VALUES,
    TIME_SHIFT_VALUES,
)
from .earthengine.eo import SPACE_TIME_BANDS as EO_SPACE_TIME_BANDS

EO_DYNAMIC_IN_TIME_BANDS_NP = np.array(EO_SPACE_TIME_BANDS + TIME_BANDS)
SPACE_TIME_BANDS = EO_SPACE_TIME_BANDS + ["NDVI"]

SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
    }
)


DatasetOutput = namedtuple("DatasetOutput", ["space_time_x", "space_x", "time_x", "months"])


def _normalize(x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray) -> np.ndarray:
    return (x - shift_values) / div_values


def normalize_space_time(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] == len(SPACE_TIME_SHIFT_VALUES):
        d_s = SPACE_TIME_SHIFT_VALUES
        d_d = SPACE_TIME_DIV_VALUES
    else:
        # there is an additional NDVI band. We assume its already normalized - *N*DVI,
        # so we leave it alone
        assert x.shape[-1] == (len(SPACE_TIME_SHIFT_VALUES) + 1)
        d_s = np.append(SPACE_TIME_SHIFT_VALUES, [0])
        d_d = np.append(SPACE_TIME_DIV_VALUES, [1])
    return _normalize(x, d_s, d_d)


def normalize_space(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray)
    return _normalize(x, SPACE_SHIFT_VALUES, SPACE_DIV_VALUES)


def normalize_time(x: np.ndarray) -> np.ndarray:
    assert isinstance(x, np.ndarray)
    return _normalize(x, TIME_SHIFT_VALUES, TIME_DIV_VALUES)


class Dataset(PyTorchDataset):
    def __init__(
        self, data_folder: Path, download: bool = True, cache_folder: Optional[Path] = None
    ):
        self.data_folder = data_folder
        if download:
            self.download(data_folder)
        self.tifs = list(data_folder.glob("*.tif")) + list(data_folder.glob("*.tiff"))
        self.cache_folder = cache_folder
        self.cache = False
        if cache_folder is not None:
            self.cache = True

    def __len__(self) -> int:
        return len(self.tifs)

    @staticmethod
    def download(data_folder):
        # Download files (faster than using Python API)
        os.system(
            f"gcloud storage cp -n -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS}/* {data_folder}"
        )

    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        space_time_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]

        size must be greater or equal to H & W
        """
        assert (space_time_x.shape[0] == space_x.shape[0]) & (
            space_time_x.shape[1] == space_x.shape[1]
        )
        assert space_time_x.shape[2] == time_x.shape[0]
        possible_h = space_time_x.shape[0] - size
        possible_w = space_time_x.shape[1] - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = space_time_x.shape[2] - num_timesteps
        assert possible_t >= 0

        if possible_h > 0:
            start_h = np.random.choice(possible_h)
        else:
            start_h = possible_h

        if possible_w > 0:
            start_w = np.random.choice(possible_w)
        else:
            start_w = possible_w

        if possible_t > 0:
            start_t = np.random.choice(possible_t)
        else:
            start_t = possible_t

        return (
            space_time_x[
                start_h : start_h + size,
                start_w : start_w + size,
                start_t : start_t + num_timesteps,
            ],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x[start_t : start_t + num_timesteps],
            months[start_t : start_t + num_timesteps],
        )

    @staticmethod
    def _fillna(data: np.ndarray, bands_np: np.ndarray):
        """Fill in the missing values in the data array"""
        if len(data.shape) == 3:
            has_time = False
        elif len(data.shape) == 4:
            has_time = True
        else:
            raise ValueError(
                f"Expected data to be 3 or 4D (x, y, (time), band) - got {data.shape}"
            )
        if data.shape[-1] != len(bands_np):
            raise ValueError(f"Expected data to have {len(bands_np)} bands - got {data.shape[-1]}")

        is_nan = np.isnan(data)
        if not is_nan.any():
            return data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_per_time_band = np.nanmean(data, axis=(0, 1))  # t, b or b

        if np.isnan(mean_per_time_band).any():
            # If a band has all nan values, fill with default: 0
            mean_per_time_band = np.nan_to_num(mean_per_time_band, nan=0)
        if is_nan.any():
            if has_time:
                means_to_fill = (
                    repeat(
                        np.nanmean(mean_per_time_band, axis=0),
                        "b -> h w t b",
                        h=data.shape[0],
                        w=data.shape[1],
                        t=data.shape[2],
                    )
                    * is_nan
                )
            else:
                means_to_fill = (
                    repeat(mean_per_time_band, "b -> h w b", h=data.shape[0], w=data.shape[1])
                    * is_nan
                )
            data = np.nan_to_num(data, nan=0) + means_to_fill
        return data

    def tif_to_npy_paths(self, tif_path: Path) -> Tuple[Path, Path, Path]:
        assert self.cache_folder is not None
        tif_name = tif_path.stem
        return (
            self.cache_folder / f"{tif_name}_space_time.npy",
            self.cache_folder / f"{tif_name}_space.npy",
            self.cache_folder / f"{tif_name}_time.npy",
        )

    @classmethod
    def month_array_from_file(cls, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        start_date = tif_path.name.partition("dates=")[2][:10]
        start_month = int(start_date.split("-")[1])
        # >>> np.fmod(np.array([9., 10, 11, 12, 13, 14]), 12)
        # array([ 9., 10., 11.,  0.,  1.,  2.])
        # - 1 because we want to index from 0
        return np.fmod(np.arange(start_month - 1, start_month - 1 + num_timesteps), 12)

    @classmethod
    def _tif_to_array(cls, tif_path: Path) -> DatasetOutput:
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

        num_timesteps = (values.shape[0] - len(SPACE_BANDS)) / len(ALL_DYNAMIC_IN_TIME_BANDS)
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        dynamic_in_time_x = rearrange(
            values[: -len(SPACE_BANDS)],
            "(t c) h w -> h w t c",
            c=len(ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = cls._fillna(dynamic_in_time_x, EO_DYNAMIC_IN_TIME_BANDS_NP)
        space_time_x = dynamic_in_time_x[:, :, :, : -len(TIME_BANDS)]
        space_time_x = np.concatenate((space_time_x, cls.calculate_ndvi(space_time_x)), axis=-1)
        space_time_x = normalize_space_time(space_time_x)

        time_x = dynamic_in_time_x[:, :, :, -len(TIME_BANDS) :]
        time_x = np.nanmean(time_x, axis=(0, 1))
        time_x = normalize_time(time_x)

        space_x = rearrange(values[-len(SPACE_BANDS) :], "c h w -> h w c")
        space_x = cls._fillna(space_x, np.array(SPACE_BANDS))
        space_x = normalize_space(space_x)

        months = cls.month_array_from_file(tif_path, int(num_timesteps))
        return DatasetOutput(
            space_time_x.astype(np.half), space_x.astype(np.half), time_x.astype(np.half), months
        )

    def load_tif(self, tif_path: Path) -> DatasetOutput:
        if self.cache_folder is None:
            return self._tif_to_array(tif_path)
        else:
            cache_path_s_t, cache_path_s, cache_path_t = self.tif_to_npy_paths(tif_path)
            if cache_path_s_t.exists():
                assert cache_path_s.exists(), f"Missing static in time data for {tif_path}"
                assert cache_path_t.exists(), f"Missing static in space data for {tif_path}"
                # check if the files exists in cache
                s_t_x = np.load(cache_path_s_t)
                num_timesteps = s_t_x.shape[2]
                months = self.month_array_from_file(tif_path, num_timesteps)
                return DatasetOutput(s_t_x, np.load(cache_path_s), np.load(cache_path_t), months)
            else:
                s_t_x, s_x, t_x, months = self._tif_to_array(tif_path)
                np.save(cache_path_s_t, s_t_x)
                np.save(cache_path_s, s_x)
                np.save(cache_path_t, t_x)
                return DatasetOutput(s_t_x, s_x, t_x, months)

    @staticmethod
    def calculate_ndvi(input_array: np.ndarray) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDVI,
        (b08 - b04) / (b08 + b04)
        """
        band_1, band_2 = "B8", "B4"
        band_1_np = input_array[:, :, :, EO_SPACE_TIME_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, EO_SPACE_TIME_BANDS.index(band_2)]

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

    def __getitem__(self, idx):
        s_t_x, s_x, t_x, months = self.load_tif(self.tifs[idx])
        s_t_x, s_x, t_x, months = self.subset_image(
            s_t_x, s_x, t_x, months, DATASET_OUTPUT_HW, NUM_TIMESTEPS
        )
        return DatasetOutput(s_t_x, s_x, t_x, months)
