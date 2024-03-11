import logging
from collections import namedtuple
from typing import Tuple

import h5py
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset as PyTorchDataset

from src.config import PRESTO_INPUT_SIZE
from src.data.dataset import (
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_DYNAMIC_BAND_GROUPS,
    NUM_DYNAMIC_BANDS,
    NUM_STATIC_BAND_GROUPS,
)

torch.multiprocessing.set_sharing_strategy("file_system")

logger = logging.getLogger("__main__")

MaskedOutput = namedtuple(
    "MaskedOutput", ["dynamic_x", "static_x", "dynamic_mask", "static_mask", "months"]
)


h5_data_dir = "../data/so2sat/TUM/"


class So2SatDataset(PyTorchDataset):
    """
    So2Sat data is provided as .h5 files in the following shapes:
    sen1: [n, 32, 32, 8]
    sen2: [n, 32, 32, 10]
    label: [n, 17] (one-hot encoded labels for 17 LCV classes)
    With n=352366 for the training set, n=24119 for the validation set, n=24188 for the test set.
    """

    def __init__(
        self,
        split: str = "training",
    ):
        assert split in ["training", "validation", "testing"]

        self.split = split
        self.data = h5py.File(h5_data_dir + split + ".h5", "r")

    def h5_to_eo_array(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.data["sen1"].shape == (self.__len__(), 32, 32, 8)
        assert self.data["sen2"].shape == (self.__len__(), 32, 32, 10)
        assert self.data["label"].shape == (self.__len__(), 17)

        # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
        vh = np.array(self.data["sen1"][i, :, :, 4])
        vv = np.array(self.data["sen1"][i, :, :, 5])

        # sen2 bands provided by so2sat correspond to the bands used by presto
        b2 = np.array(self.data["sen2"][i, :, :, 0])
        b3 = np.array(self.data["sen2"][i, :, :, 1])
        b4 = np.array(self.data["sen2"][i, :, :, 2])

        b5 = np.array(self.data["sen2"][i, :, :, 3])
        b6 = np.array(self.data["sen2"][i, :, :, 4])
        b7 = np.array(self.data["sen2"][i, :, :, 5])

        b8 = np.array(self.data["sen2"][i, :, :, 6])

        b8a = np.array(self.data["sen2"][i, :, :, 7])

        b11 = np.array(self.data["sen2"][i, :, :, 8])
        b12 = np.array(self.data["sen2"][i, :, :, 9])

        label = np.array(self.data["label"][i, :])

        # labels should be one-hot encoded
        assert np.sum(label) == 1
        assert np.all(np.logical_or(label == 0, label == 1))

        d_x = np.stack([vv, vh, b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12], axis=-1)

        return (d_x, label)

    @staticmethod
    def create_so2sat_masks() -> Tuple[np.ndarray, np.ndarray]:
        dynamic_channels = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key or "S1" in key
        ]

        # everything is masked by default
        dynamic_mask = np.ones([PRESTO_INPUT_SIZE, PRESTO_INPUT_SIZE, 1, NUM_DYNAMIC_BAND_GROUPS])
        # unmask available bands
        dynamic_mask[dynamic_channels] = 0

        # no static channels are available
        static_mask = np.ones([PRESTO_INPUT_SIZE, PRESTO_INPUT_SIZE, NUM_STATIC_BAND_GROUPS])

        assert np.unique(dynamic_mask).tolist() == [0, 1]
        assert np.unique(static_mask).tolist() == [1]

        return (dynamic_mask, static_mask)

    @staticmethod
    def add_missing_channels(d_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # s_x is not provided by so2sat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], 2))

        # 12 presto channels are provided by so2sat
        d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 12))
        d_x = np.concatenate((d_x, d_x_missing), axis=-1)

        return (d_x, s_x)

    def __len__(self):
        return self.data["sen1"].shape[0]

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        d_x, label = self.h5_to_eo_array(idx)
        d_x = d_x.reshape(d_x.shape[0], d_x.shape[1], 1, d_x.shape[2])

        d_x, s_x = self.add_missing_channels(d_x)

        d_m, s_m = self.create_so2sat_masks()
        month = np.zeros((1,))

        d_x = torch.from_numpy(d_x).float()
        s_x = torch.from_numpy(s_x).float()
        d_m = torch.from_numpy(d_m).long()
        s_m = torch.from_numpy(s_m).long()
        month = torch.from_numpy(month).long()
        label = torch.from_numpy(label).long()

        return (MaskedOutput(d_x, s_x, d_m, s_m, month), label)