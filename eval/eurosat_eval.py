import logging
from datasets import load_dataset
from collections import namedtuple
from typing import Tuple

import h5py
import tqdm
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import DataLoader

from src.config import PRESTO_INPUT_SIZE
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

    The classes are:
        "AnnualCrop": 0
        "Forest": 1
        "HerbaceousVegetation": 2
        "Highway": 3
        "Industrial": 4
        "Pasture": 5
        "PermanentCrop": 6
        "Residential": 7
        "River": 8
        "SeaLake": 9
    """

    # this is not the true start month!
    start_month = 1

    def __init__(
        self,
        rgb: bool = True,
        split: str = "train",
        merge_train_val: bool = True,
    ):
        assert split in ["train", "val", "test"]

        self.split = split
        self.rgb = rgb

        if self.rgb:
            self.data = load_dataset("blanchon/EuroSAT_RGB", split=self.split)
        # load MSI data
        else:
            self.data = load_dataset("blanchon/EuroSAT_MSI", split=self.split)

    
    def patchify():
        NotImplementedError


    def image_to_eo_array(self, idx: int):
        
        image = np.array(self.data[idx]['image'])
        label = self.data[idx]['label']

        # for MSI, remove band 9 and 10
        if not self.rgb:
            image = np.delete(image, [9, 10], axis=2)

        return (image, label)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        NotImplementedError

    def __len__(self):
        NotImplementedError
