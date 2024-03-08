import h5py
import numpy as np
import logging

from torch.utils.data import Dataset as PyTorchDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger("__main__")


h5_data_dir = "./data/so2sat/TUM/"


class So2SatDataset(PyTorchDataset):
    """
    So2Sat data is provided as .h5 files in the following shapes:
    sen1: [n, 32, 32, 8]
    sen2: [n, 32, 32, 10]
    label: [n, 17] (one-hot encoded labels for 17 LCV classes)
    With n=352366 for the training set, n=24119 for the validation set, n=24188 for the test set.
    """

    """
    Presto expects the following shape:
    d_x: [32, 32, 12, 9] (vv/vh, b2/b3/b4, b5/b6/b7, b8, b8a, b11/b12, 2era5, 1dw (with 9 class values), 1ndvi) dim=15
    s_x: [32, 32, 12, 1] (2 srtm) dim=2
    all_x = [32, 32, 12, 10]

    We can provide:
    d_x: [32, 32, 1, 9]
    s_x: [32, 32, 1, 0] (so no s_x)
    all_x = [32, 32, 1, 9]

    Presto expects the following channels:
    sen1: [vv, vh]
    sen2: [b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12]


    So we need to mask out the following channels:
    """
    def __init__(
            self, 
            split: str = "training",
    ):
        assert split in ["training", "validation", "testing"]

        self.split = split
        self.data = h5py.File(h5_data_dir + split + ".h5", 'r')
    
    def h5_to_eo_array(self, i: int):

        assert self.data['sen1'].shape == (self.__len__, 32, 32, 8)
        assert self.data['sen2'].shape == (self.__len__, 32, 32, 10)
        assert self.data['label'].shape == (self.__len__, 17)

        # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
        vh = np.array(self.data['sen1'][i,:,:,4])
        vv = np.array(self.data['sen1'][i,:,:,5])
        s1 = np.stack([vv, vh], axis=-1)

        # the sen2 bands provided by so2sat correspond to the bands used by presto
        b2 = np.array(self.data['sen2'][i,:,:,0])
        b3 = np.array(self.data['sen2'][i,:,:,1])
        b4 = np.array(self.data['sen2'][i,:,:,2])
        s2_1 = np.stack([b2, b3, b4], axis=-1)

        b5 = np.array(self.data['sen2'][i,:,:,3])
        b6 = np.array(self.data['sen2'][i,:,:,4])
        b7 = np.array(self.data['sen2'][i,:,:,5])
        s2_2 = np.stack([b5, b6, b7], axis=-1)

        b8 = np.array(self.data['sen2'][i,:,:,6])
        s2_3 = b8

        b8a = np.array(self.data['sen2'][i,:,:,7])
        s2_4 = b8a

        b11 = np.array(self.data['sen2'][i,:,:,8])
        b12 = np.array(self.data['sen2'][i,:,:,9])
        s2_5 = np.stack([b11, b12], axis=-1)

        label = np.array(self.data['label'][i,:])

        # labels should be one-hot encoded
        assert np.sum(label) == 1
        assert np.all(np.logical_or(label == 0, label == 1))

        return s1, s2_1, s2_2, s2_3, s2_4, s2_5, label
    
    """
    @staticmethod
    def collate_fn(data):
        x, labels, dw, latlons, month = default_collate(data)
        return (
            rearrange(x, "b bp t d -> (b bp) t d"),
            # ... is an optional dimension: for TreeSat labels which are arrays
            rearrange(labels, "b bp ... -> (b bp) ..."),
            rearrange(dw, "b bp t -> (b bp) t"),
            rearrange(latlons, "b bp d -> (b bp) d"),
            # ... = optional dimension for sequences with timesteps > 1
            rearrange(month, "b bp ... -> (b bp) ..."),
        )

    @staticmethod
    def finetuning_collate_fn(data):
        # NOTE: the finetuning function expects data in
        # a different order than the sklearn function.
        # this is confusing and should be fixed
        x, labels, dw, latlons, month = default_collate(data)
        if len(labels.shape) > 2:
            # RuntimeError: Expected floating point type for target with
            # class probabilities
            labels = labels.float()
        return (x, dw, latlons, labels[:, 0], month)
    """


    def __len__(self):
        return self.data['sen1'].shape[0]


    def __getitem__(self, idx):

        s1, s2_1, s2_2, s2_3, s2_4, s2_5, label = self.h5_to_array(self.split, idx)
        logger.info(f"s1: {s1.shape}, s2_1: {s2_1.shape}, s2_2: {s2_2.shape}, s2_3: {s2_3.shape}, s2_4: {s2_4.shape}, s2_5: {s2_5.shape}, label: {label.shape}")
        logger.info(f"s1 unique: {np.unique(s1)}, s2_1 unique: {np.unique(s2_1)}, s2_2 unique: {np.unique(s2_2)}, s2_3 unique: {np.unique(s2_3)}, s2_4 unique: {np.unique(s2_4)}, s2_5 unique: {np.unique(s2_5)}, label unique: {np.unique(label)}")

        # provide a default class for dw, ndvi, era5 and srtm

        return (
            s1, 
            s2_1, 
            s2_2, 
            s2_3, 
            s2_4, 
            s2_5, 
            label
        )







# b is a MaskedOutput object d_x, s_x, d_m, s_m, months
# d_x has shape [32, 32, 1, 9] with the correct channels masked out in d_m

# write a test for the dataset
