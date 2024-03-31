from typing import Tuple

import h5py
import numpy as np
import torch.multiprocessing
from einops import repeat
from torch.utils.data import Dataset as PyTorchDataset

from ..data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    S1_BANDS,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    normalize_dynamic,
)
from ..data.earthengine.s2 import S2_BANDS
from ..masked_datasets import MaskedOutput
from ..utils import data_dir

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
        so2sat_dir: str = "so2sat/TUM/",
    ):
        assert split in ["training", "validation", "testing"]

        self.split = split
        self.so2sat_dir = so2sat_dir
        self._len = None
        self.images, self.labels = self.h5_to_eo_array()
        self.masks = self.create_so2sat_masks()

    def h5_to_eo_array(self) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
            assert data["sen1"].shape == (self.__len__(), 32, 32, 8)
            assert data["sen2"].shape == (self.__len__(), 32, 32, 10)
            assert data["label"].shape == (self.__len__(), 17)

            # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
            s1 = np.array(data["sen1"][:, :, :, 4:6])
            # sen2 bands provided by so2sat correspond to the bands used by presto
            s2 = np.array(data["sen2"][:, :, :, :10])

            labels = np.array(data["label"][:, :])

        images = np.concatenate([s1, s2], axis=-1)

        return images, labels

    def create_so2sat_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        dynamic_channels = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S" in key]

        # everything is masked by default
        dynamic_mask = np.ones([len(DYNAMIC_BANDS_GROUPS_IDX)])
        # unmask available s1 and s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # no static channels are available
        static_mask = np.ones(
            [self.input_height_width, self.input_height_width, len(STATIC_BAND_GROUPS_IDX)]
        )

        assert ((dynamic_mask == 0) | (dynamic_mask == 1)).all()
        assert (static_mask == 1).all()

        return (dynamic_mask, static_mask)

    def image_to_dynamic_eo_array(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kept_dynamic_bands = [
            idx for idx, x in enumerate(DYNAMIC_BANDS) if (x in S2_BANDS or x in S1_BANDS)
        ]

        eo_style_array = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(DYNAMIC_BANDS),
            ]
        )
        eo_style_array[:, :, :, kept_dynamic_bands] = repeat(
            image, "h w c -> h w t c", t=self.num_timesteps
        )

        return normalize_dynamic(eo_style_array)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        d_x = self.image_to_dynamic_eo_array(image)

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
            with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
                self._len = data["sen1"].shape[0]
        return self._len
