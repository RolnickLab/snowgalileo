import json
import logging
import math
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union, cast
from typing import OrderedDict as OrderedDictType

import h5py
import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import rearrange, repeat
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from .config import (
    DATASET_OUTPUT_HW,
    EE_BUCKET_TIFS,
    EE_FOLDER_H5PYS,
    EE_FOLDER_TIFS,
    NORMALIZATION_DICT_FILENAME,
    NUM_TIMESTEPS,
)
from .dataset_stats import RunningStatistics
from .earthengine.eo import (
    ALL_DYNAMIC_IN_TIME_BANDS,
    DW_BANDS,
    DW_DIV_VALUES,
    DW_SHIFT_VALUES,
    ERA5_BANDS,
    LANDSCAN_BANDS,
    LOCATION_BANDS,
    S1_BANDS,
    SPACE_BANDS,
    SPACE_DIV_VALUES,
    SPACE_SHIFT_VALUES,
    SRTM_BANDS,
    TC_BANDS,
    TIME_BANDS,
    TIME_DIV_VALUES,
    TIME_SHIFT_VALUES,
    VIIRS_BANDS,
    WC_BANDS,
    WC_DIV_VALUES,
    WC_SHIFT_VALUES,
)
from .earthengine.eo import SPACE_TIME_BANDS as EO_SPACE_TIME_BANDS
from .earthengine.eo import SPACE_TIME_DIV_VALUES as EO_SPACE_TIME_DIV_VALUES
from .earthengine.eo import SPACE_TIME_SHIFT_VALUES as EO_SPACE_TIME_SHIFT_VALUES
from .earthengine.eo import STATIC_BANDS as EO_STATIC_BANDS
from .earthengine.eo import STATIC_DIV_VALUES as EO_STATIC_DIV_VALUES
from .earthengine.eo import STATIC_SHIFT_VALUES as EO_STATIC_SHIFT_VALUES

logger = logging.getLogger("__main__")

EO_DYNAMIC_IN_TIME_BANDS_NP = np.array(EO_SPACE_TIME_BANDS + TIME_BANDS)
SPACE_TIME_BANDS = EO_SPACE_TIME_BANDS + ["NDVI"]
SPACE_TIME_SHIFT_VALUES = np.append(EO_SPACE_TIME_SHIFT_VALUES, [0])
SPACE_TIME_DIV_VALUES = np.append(EO_SPACE_TIME_DIV_VALUES, [1])

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
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]
STATIC_BANDS = EO_STATIC_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS
STATIC_DIV_VALUES = np.append(EO_STATIC_DIV_VALUES, (DW_DIV_VALUES + WC_DIV_VALUES))
STATIC_SHIFT_VALUES = np.append(EO_STATIC_SHIFT_VALUES, (DW_SHIFT_VALUES + WC_SHIFT_VALUES))

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


class Normalizer:
    # these are the bands we will replace with the 2*std computation
    # if std_clip = True
    std_bands = {
        "space_time": [b for b in SPACE_TIME_BANDS if b != "NDVI"],
        "space": SRTM_BANDS,
        "time": TIME_BANDS,
        "static": LANDSCAN_BANDS,
    }

    def __init__(self, std_clip: bool = True, normalizing_dicts: Optional[Dict] = None):
        self.shift_div_dict = {
            "space_time": {
                "shift": deepcopy(SPACE_TIME_SHIFT_VALUES),
                "div": deepcopy(SPACE_TIME_DIV_VALUES),
            },
            "space": {"shift": deepcopy(SPACE_SHIFT_VALUES), "div": deepcopy(SPACE_DIV_VALUES)},
            "time": {"shift": deepcopy(TIME_SHIFT_VALUES), "div": deepcopy(TIME_DIV_VALUES)},
            "static": {"shift": deepcopy(STATIC_SHIFT_VALUES), "div": deepcopy(STATIC_DIV_VALUES)},
        }

        self.std_clip = std_clip
        self.normalizing_dicts = normalizing_dicts
        if std_clip:
            name_to_bands = {
                "space_time": SPACE_TIME_BANDS,
                "space": SPACE_BANDS,
                "time": TIME_BANDS,
                "static": STATIC_BANDS,
            }
            assert normalizing_dicts is not None
            for key, val in normalizing_dicts.items():
                if key == "n":
                    continue
                bands_to_replace = self.std_bands[key]
                for band in bands_to_replace:
                    band_idx = name_to_bands[key].index(band)
                    mean = val["mean"][band_idx]
                    std = val["std"][band_idx]
                    div = (mean + (2 * std)) - (mean - (2 * std))  # max_val - min_val
                    self.shift_div_dict[key]["shift"][band_idx] = mean
                    self.shift_div_dict[key]["div"][band_idx] = div

    def _normalize(
        self, x: np.ndarray, shift_values: np.ndarray, div_values: np.ndarray
    ) -> np.ndarray:
        x = (x - shift_values) / div_values
        if self.std_clip:
            return np.clip(x, a_min=0, a_max=1)
        return x

    def __call__(self, x: np.ndarray):
        for _, val in self.shift_div_dict.items():
            if x.shape[-1] == len(val["shift"]):
                return self._normalize(x, val["shift"], val["div"])
        raise ValueError(f"Unexpected input shape {x.shape}")


class DatasetOutput(NamedTuple):
    space_time_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    months: np.ndarray

    @classmethod
    def concatenate(cls, datasetoutputs: Sequence["DatasetOutput"]) -> "DatasetOutput":
        s_t_x = np.stack([o.space_time_x for o in datasetoutputs], axis=0)
        sp_x = np.stack([o.space_x for o in datasetoutputs], axis=0)
        t_x = np.stack([o.time_x for o in datasetoutputs], axis=0)
        st_x = np.stack([o.static_x for o in datasetoutputs], axis=0)
        months = np.stack([o.months for o in datasetoutputs], axis=0)
        return cls(s_t_x, sp_x, t_x, st_x, months)

    def normalize(self, normalizer: Normalizer) -> "DatasetOutput":
        return DatasetOutput(
            normalizer(self.space_time_x),
            normalizer(self.space_x),
            normalizer(self.time_x),
            normalizer(self.static_x),
            self.months,
        )


def to_cartesian(
    lat: Union[float, np.ndarray, torch.Tensor], lon: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(lat, float):
        assert -90 <= lat <= 90, f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert -180 <= lon <= 180, f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert -90 <= lat.min(), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert 90 >= lat.max(), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert -180 <= lon.min(), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert 180 >= lon.max(), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")


class Dataset(PyTorchDataset):
    def __init__(
        self,
        data_folder: Path,
        download: bool = True,
        h5py_folder: Optional[Path] = None,
        h5pys_only: bool = False,
        output_hw: int = DATASET_OUTPUT_HW,
        output_timesteps: int = NUM_TIMESTEPS,
        normalizer: Optional[Normalizer] = None,
    ):
        self.data_folder = data_folder
        self.h5pys_only = h5pys_only
        self.h5py_folder = h5py_folder
        self.cache = False
        if normalizer is None:
            self.normalizer = Normalizer(std_clip=False, normalizing_dicts=None)
        else:
            self.normalizer = normalizer

        if h5py_folder is not None:
            self.cache = True
        if h5pys_only:
            assert h5py_folder is not None, "Can't use h5pys only if there is no cache folder"
            self.tifs: List[Path] = []
            if download:
                self.download_h5pys(h5py_folder)
            self.h5pys = list(h5py_folder.glob("*.h5"))
        else:
            if download:
                self.download_tifs(data_folder)
            self.tifs = []
            tifs = list(data_folder.glob("*.tif")) + list(data_folder.glob("*.tiff"))
            for tif in tifs:
                try:
                    _ = self.start_month_from_file(tif)
                    self.tifs.append(tif)
                except IndexError:
                    warnings.warn(f"IndexError for {tif}")
            self.h5pys = []

        self.output_hw = output_hw
        self.output_timesteps = output_timesteps

    def __len__(self) -> int:
        if self.h5pys_only:
            return len(self.h5pys)
        return len(self.tifs)

    @staticmethod
    def download_tifs(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_TIFS} {data_folder}")

    @staticmethod
    def download_h5pys(data_folder):
        # Download files (faster than using Python API)
        os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/{EE_FOLDER_H5PYS} {data_folder}")

    @staticmethod
    def return_subset_indices(
        total_h,
        total_w,
        total_t,
        size: int,
        num_timesteps: int,
    ) -> Tuple[int, int, int]:
        """
        space_time_x: array of shape [H, W, T, D]
        space_x: array of shape [H, W, D]
        time_x: array of shape [T, D]
        static_x: array of shape [D]

        size must be greater or equal to H & W
        """
        possible_h = total_h - size
        possible_w = total_w - size
        assert (possible_h >= 0) & (possible_w >= 0)
        possible_t = total_t - num_timesteps
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

        return start_h, start_w, start_t

    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            static_x,
            months[start_t : start_t + num_timesteps],
        )

    @staticmethod
    def _fillna(data: np.ndarray, bands_np: np.ndarray):
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
            raise ValueError(
                f"Expected data to be 3D or 4D (x, y, (time), band) - got {data.shape}"
            )

        # treat infinities as NaNs
        data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
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
    def start_month_from_file(cls, tif_path: Path) -> int:
        start_date = tif_path.name.partition("dates=")[2][:10]
        start_month = int(start_date.split("-")[1])
        return start_month

    @classmethod
    def month_array_from_file(cls, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        # assumes all files are exported with filenames including:
        # *dates=<start_date>*, where the start_date is in a YYYY-MM-dd format
        start_month = cls.start_month_from_file(tif_path)
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
            lon = np.mean(cast(np.ndarray, data.x)).item()
            lat = np.mean(cast(np.ndarray, data.y)).item()

        # this is a bit hackey but is a unique edge case for locations,
        # which are not part of the exported bands but are instead
        # computed here
        static_bands_in_tif = len(EO_STATIC_BANDS) - len(LOCATION_BANDS)

        num_timesteps = (values.shape[0] - len(SPACE_BANDS) - static_bands_in_tif) / len(
            ALL_DYNAMIC_IN_TIME_BANDS
        )
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        dynamic_in_time_x = rearrange(
            values[: -(len(SPACE_BANDS) + static_bands_in_tif)],
            "(t c) h w -> h w t c",
            c=len(ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = cls._fillna(dynamic_in_time_x, EO_DYNAMIC_IN_TIME_BANDS_NP)
        space_time_x = dynamic_in_time_x[:, :, :, : -len(TIME_BANDS)]

        # calculate indices, which have shape [h, w, t, 1]
        ndvi = cls.calculate_ndi(space_time_x, band_1="B8", band_2="B4")

        space_time_x = np.concatenate((space_time_x, ndvi), axis=-1)

        time_x = dynamic_in_time_x[:, :, :, -len(TIME_BANDS) :]
        time_x = np.nanmean(time_x, axis=(0, 1))

        space_x = rearrange(
            values[-(len(SPACE_BANDS) + static_bands_in_tif) : -static_bands_in_tif],
            "c h w -> h w c",
        )
        space_x = cls._fillna(space_x, np.array(SPACE_BANDS))

        static_x = values[-static_bands_in_tif:]
        # add DW_STATIC and WC_STATIC
        dw_bands = space_x[:, :, [i for i, v in enumerate(SPACE_BANDS) if v in DW_BANDS]]
        wc_bands = space_x[:, :, [i for i, v in enumerate(SPACE_BANDS) if v in WC_BANDS]]
        static_x = np.concatenate(
            [
                np.nanmean(static_x, axis=(1, 2)),
                to_cartesian(lat, lon),
                np.nanmean(dw_bands, axis=(0, 1)),
                np.nanmean(wc_bands, axis=(0, 1)),
            ]
        )
        static_x = cls._fillna(static_x, np.array(STATIC_BANDS))

        months = cls.month_array_from_file(tif_path, int(num_timesteps))

        try:
            assert not np.isnan(space_time_x).any(), f"NaNs in s_t_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_x).any(), f"Infs in s_t_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
            )
        except AssertionError as e:
            raise e

    def _tif_to_array_with_checks(self, idx):
        tif_path = self.tifs[idx]
        try:
            output = self._tif_to_array(tif_path)
            return output
        except Exception as e:
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.tifs[idx] = self.tifs[new_idx]
            tif_path = self.tifs[idx]
        output = self._tif_to_array(tif_path)
        return output

    def load_tif(self, idx: int) -> DatasetOutput:
        if self.h5py_folder is None:
            s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
            return DatasetOutput(
                *self.subset_image(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    months,
                    size=self.output_hw,
                    num_timesteps=self.output_timesteps,
                )
            )
        else:
            h5py_path = self.tif_to_h5py_path(self.tifs[idx])
            if h5py_path.exists():
                try:
                    return self.read_and_slice_h5py_file(self.tifs[idx])
                except Exception as e:
                    logger.warn(f"Exception {e} for {self.tifs[idx]}")
                    h5py_path.unlink()
                    s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
                    self.save_h5py(s_t_x, sp_x, t_x, st_x, self.tifs[idx].stem)
                    return DatasetOutput(
                        *self.subset_image(
                            s_t_x, sp_x, t_x, st_x, months, self.output_hw, self.output_timesteps
                        )
                    )
            else:
                s_t_x, sp_x, t_x, st_x, months = self._tif_to_array_with_checks(idx)
                self.save_h5py(s_t_x, sp_x, t_x, st_x, self.tifs[idx].stem)
                return DatasetOutput(
                    *self.subset_image(
                        s_t_x, sp_x, t_x, st_x, months, self.output_hw, self.output_timesteps
                    )
                )

    def save_h5py(self, s_t_x, sp_x, t_x, st_x, tif_stem):
        assert self.h5py_folder is not None
        with h5py.File(self.h5py_folder / f"{tif_stem}.h5", "w") as hf:
            hf.create_dataset("s_t_x", data=s_t_x)
            hf.create_dataset("sp_x", data=sp_x)
            hf.create_dataset("t_x", data=t_x)
            hf.create_dataset("st_x", data=st_x)

    @staticmethod
    def calculate_ndi(input_array: np.ndarray, band_1: str, band_2: str) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """
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

    def read_and_slice_h5py_file(self, h5py_path: Path):
        with h5py.File(h5py_path, "r") as hf:
            h, w, t, _ = hf["s_t_x"].shape
            start_h, start_w, start_t = self.return_subset_indices(
                h, w, t, self.output_hw, self.output_timesteps
            )
            months = self.month_array_from_file(h5py_path, t)
            output = DatasetOutput(
                hf["s_t_x"][
                    start_h : start_h + self.output_hw,
                    start_w : start_w + self.output_hw,
                    start_t : start_t + self.output_timesteps,
                ],
                hf["sp_x"][
                    start_h : start_h + self.output_hw,
                    start_w : start_w + self.output_hw,
                ],
                hf["t_x"][start_t : start_t + self.output_timesteps],
                hf["st_x"][:],
                months[start_t : start_t + self.output_timesteps],
            )
        return output

    def __getitem__(self, idx):
        if self.h5pys_only:
            return self.read_and_slice_h5py_file(self.h5pys[idx]).normalize(self.normalizer)
        else:
            return self.load_tif(idx).normalize(self.normalizer)

    def process_h5pys(self):
        # iterate through the dataset and save it all as h5pys
        assert self.h5py_folder is not None
        assert not self.h5pys_only
        assert self.cache

        for i in tqdm(range(len(self))):
            # loading the tifs also saves them
            # if they don't exist
            _ = self[i]

    @staticmethod
    def _update_normalizing_values(array, interim_dict):
        # given an input array of shape [timesteps, bands]
        # update the normalizing dict
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # https://www.johndcook.com/blog/standard_deviation/

        # we want a 2d array [instances, bands]
        if len(array.shape) > 2:
            array = np.reshape(array, [-1, array.shape[-1]])
        elif len(array.shape) == 1:
            array = np.expand_dims(array, 0)

        for b in range(array.shape[0]):
            x = array[b, :]
            interim_dict["n"] += 1
            delta = x - interim_dict["mean"]
            interim_dict["mean"] += delta / interim_dict["n"]
            interim_dict["M2"] += delta * (x - interim_dict["mean"])
        return interim_dict

    @staticmethod
    def _calculate_normalizing_dict(interim_dict):
        variance = interim_dict["M2"] / (interim_dict["n"] - 1)
        std = np.sqrt(variance)
        return {
            "mean": cast(np.ndarray, interim_dict["mean"]).tolist(),
            "std": cast(np.ndarray, std).tolist(),
        }

    def load_compute_normalization_values(
        self, output_hw: int = 96, output_timesteps: int = 24, save: bool = True
    ):
        normalizing_dict_path = self.data_folder / NORMALIZATION_DICT_FILENAME
        # check to see if the normalization dict already exists
        if normalizing_dict_path.exists():
            with normalizing_dict_path.open("r") as f:
                norm_dict = json.load(f)
            if norm_dict["n"] == len(self):
                # we computed the normalizing dict using the same datset
                return norm_dict
            else:
                normalizing_dict_path.unlink()

        org_hw = self.output_hw
        self.output_hw = output_hw

        org_t = self.output_timesteps
        self.output_timesteps = output_timesteps

        s_t_interim = RunningStatistics()
        sp_interim = RunningStatistics()
        t_interim = {"n": 0, "mean": np.zeros(len(TIME_BANDS)), "M2": np.zeros(len(TIME_BANDS))}
        st_interim = {
            "n": 0,
            "mean": np.zeros(len(STATIC_BANDS)),
            "M2": np.zeros(len(STATIC_BANDS)),
        }

        for i in tqdm(range(len(self))):
            s_t_x, sp_x, t_x, st_x, _ = self[i]
            s_t_interim.update(s_t_x)
            sp_interim.update(sp_x)
            t_interim = self._update_normalizing_values(t_x, t_interim)
            st_interim = self._update_normalizing_values(st_x, st_interim)

        self.output_hw = org_hw
        self.output_timesteps = org_t

        norm_dict = {
            "n": len(self),
            "space_time": {"mean": s_t_interim.mean, "std": s_t_interim.std},
            "space": {"mean": sp_interim.mean, "std": sp_interim.std},
            "time": self._calculate_normalizing_dict(t_interim),
            "static": self._calculate_normalizing_dict(st_interim),
        }

        if save:
            with normalizing_dict_path.open("w") as f:
                json.dump(norm_dict, f)

        return norm_dict
