import logging
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch.multiprocessing
from datasets import load_dataset
from einops import repeat
from torch.utils.data import Dataset as PyTorchDataset

from src.data.dataset import (
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_DYNAMIC_BAND_GROUPS,
    NUM_DYNAMIC_BANDS,
    NUM_STATIC_BAND_GROUPS,
)

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")
logger = logging.getLogger("__main__")
MaskedOutput = namedtuple(
    "MaskedOutput", ["dynamic_x", "static_x", "dynamic_mask", "static_mask", "months"]
)


class EuroSatDataset(PyTorchDataset):
    """
    EuroSat provides two datasets:
    - 27000 RGB images of 64x64 pixels (3 sen2 bands), 10 land cover classes
    - 27000 MSI images of 64x64 pixels (13 sen2 bands), 10 land cover classes
    """

    # this is not the true start month!
    start_month = 1

    def __init__(
        self,
        rgb: bool = True,
        split: str = "train",
        merge_train_val: bool = True,
    ):
        assert split in ["train", "validation", "test"]

        self.split = split
        self.rgb = rgb
        self.input_size = 64

        if self.rgb:
            self.data = load_dataset("blanchon/EuroSAT_RGB", split=self.split)

        # MSI data
        else:
            self.data = load_dataset("blanchon/EuroSAT_MSI", split=self.split)

    def create_eurosat_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.rgb:
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2_RGB" in key
            ]

        else:
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key
            ]

        # everything is masked by default
        dynamic_mask = np.ones([NUM_DYNAMIC_BAND_GROUPS])
        # unmask available s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask, "d -> h w t d", h=self.input_size, w=self.input_size, t=1
        )

        # no static channels are available
        static_mask = np.ones([self.input_size, self.input_size, NUM_STATIC_BAND_GROUPS])

        assert np.unique(dynamic_mask).tolist() == [0, 1]
        assert np.unique(static_mask).tolist() == [1]

        return (dynamic_mask, static_mask)

    def add_missing_channels(self, d_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # s_x is not provided by eurosat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], 2))

        # 3 presto channels are provided by RGB
        if self.rgb:
            d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 3))
        else:
            d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 10))

        d_x = np.concatenate((d_x, d_x_missing), axis=-1)

        return (d_x, s_x)

    def image_to_eo_array(self, idx: int):
        image = np.array(self.data[idx]["image"])
        label = self.data[idx]["label"]

        # for MSI, remove band 1,9 and 10
        if not self.rgb:
            image = np.delete(image, [0, 9, 10], axis=2)

        return (image, label)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        d_x, label = self.image_to_eo_array(idx)
        d_x = d_x.reshape(d_x.shape[0], d_x.shape[1], 1, d_x.shape[2])

        d_x, s_x = self.add_missing_channels(d_x)

        d_m, s_m = self.create_eurosat_masks()
        month = np.zeros((1,))

        d_x_torch = torch.as_tensor(d_x, dtype=torch.float32)
        s_x_torch = torch.as_tensor(s_x, dtype=torch.float32)
        d_m_torch = torch.as_tensor(d_m, dtype=torch.float32)
        s_m_torch = torch.as_tensor(s_m, dtype=torch.float32)
        month_torch = torch.as_tensor(month, dtype=torch.long)
        label_torch = torch.as_tensor(label, dtype=torch.long)

        return (MaskedOutput(d_x_torch, s_x_torch, d_m_torch, s_m_torch, month_torch), label_torch)

    def __len__(self):
        return len(self.data)
