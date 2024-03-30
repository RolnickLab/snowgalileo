import logging
from collections import namedtuple
from typing import Tuple, Optional

import h5py
import numpy as np
import torch.multiprocessing
from einops import repeat
from torch.utils.data import Dataset as PyTorchDataset

from ..data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    normalize_dynamic,
)
from ..masked_datasets import MaskedOutput

torch.multiprocessing.set_sharing_strategy("file_system")


class So2SatDataset(PyTorchDataset):
    """
    So2Sat data is provided as .h5 files in the following shapes:
    sen1: [n, 32, 32, 8]
    sen2: [n, 32, 32, 10]
    label: [n, 17] (one-hot encoded labels for 17 LCV classes)

    With n=352366 for the training set, n=24119 for the validation set, n=24188 for the testing set.
    """

    input_height_width = 32
    num_timesteps = 1

    def __init__(
        self,
        split: str = "training",
        merge_train_val: bool = True,
        h5_files_dir: Optional[str] = "so2sat/TUM",
    ):
        assert split in ["training", "validation", "testing"]

        self.split = split
        self.h5_files_dir = h5_files_dir
        self._len = None

        self.masks = self.create_so2sat_masks()

    def h5_to_eo_array(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_files_dir, "r") as data:
            assert data["sen1"].shape == (self.__len__(), 32, 32, 8)
            assert data["sen2"].shape == (self.__len__(), 32, 32, 10)
            assert data["label"].shape == (self.__len__(), 17)

            # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
            s1 = np.array(data["sen1"][i, :, :, 4:6])

            # sen2 bands provided by so2sat correspond to the bands used by presto
            s2 = np.array(data["sen2"][i, :, :, :10])

            label = np.array(data["label"][i, :])

        d_x = np.stack([s1, s2], axis=-1)

        return (d_x, label)
    
    def image_to_dynamic_eo_array(self, tif_filename: str) -> Tuple[np.ndarray, np.ndarray]:
        kept_dynamic_bands = [
            idx
            for idx, x in enumerate(DYNAMIC_BANDS)
            if ((x in ALL_S2_BANDS) and (x not in REMOVED_BANDS) or (x in S1_BANDS))
        ]

        tif_file = self.image_name_to_path(tif_filename)

        with cast(xarray.DataArray, xr.open_rasterio(tif_file)) as image:
            eo_style_array = np.zeros(
                [
                    self.input_height_width,
                    self.input_height_width,
                    self.num_timesteps,
                    len(DYNAMIC_BANDS),
                ]
            )
            image_kept_bands = image.values[kept_s2_bands]
            eo_style_array[:, :, :, kept_dynamic_bands] = repeat(
                image_kept_bands, "c h w -> h w t c", t=self.num_timesteps
            )

        return (
            normalize_dynamic(eo_style_array),
            np.array([self.labels_to_int[tif_file.parents[0].name]]),
        )

    def create_so2sat_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        dynamic_channels = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key or "S1" in key
        ]

        # everything is masked by default
        dynamic_mask = np.ones([len(DYNAMIC_BANDS_GROUPS_IDX)])
        # unmask available s1 and s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask, "d -> h w t d", h=self.input_size, w=self.input_size, t=self.num_timesteps
        )

        # no static channels are available
        static_mask = np.ones([self.input_height_width, self.input_height_width, len(STATIC_BAND_GROUPS_IDX)])

        assert ((dynamic_mask == 0) | (dynamic_mask == 1)).all()
        assert (static_mask == 1).all()

        return (dynamic_mask, static_mask)

    @staticmethod
    def add_missing_channels(d_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # s_x is not provided by so2sat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], 2))

        # 12 presto channels are provided by so2sat
        d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 12))
        d_x = np.concatenate((d_x, d_x_missing), axis=-1)

        return (d_x, s_x)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image = self.images[idx]
        d_x, label = self.image_to_dynamic_eo_array(image.strip())

        d_x, s_x = self.add_missing_channels(d_x)

        # static bands are not provided by so2sat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], len(STATIC_BANDS)))

        d_m, s_m = self.masks
        month = np.zeros((self.num_timesteps,))

        d_x_torch = torch.as_tensor(d_x, dtype=torch.float32)
        s_x_torch = torch.as_tensor(s_x, dtype=torch.float32)
        d_m_torch = torch.as_tensor(d_m, dtype=torch.float32)
        s_m_torch = torch.as_tensor(s_m, dtype=torch.float32)
        month_torch = torch.as_tensor(month, dtype=torch.long)
        label_torch = torch.as_tensor(label, dtype=torch.long)

        return (MaskedOutput(d_x_torch, s_x_torch, d_m_torch, s_m_torch, month_torch), label_torch)

    def __len__(self) -> int:
        if self._len is None:
            with h5py.File(self.h5_files_dir, "r") as data:
                self._len = data["sen1"].shape[0]
        return self._len