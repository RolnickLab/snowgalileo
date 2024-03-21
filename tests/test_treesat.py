import unittest

import torch

from src.data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
)
from src.eval.treesat import TreeSatDataset

TEST_FILE = "Tilia_spec._9_99911_WEFL_NLF.tif"


class TestTreeSat(unittest.TestCase):
    def check_dynamic_shape(self, dynamic_x, dynamic_m):
        self.assertEqual(
            dynamic_x.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                TreeSatDataset.num_timesteps,
                len(DYNAMIC_BANDS),
            ),
        )
        self.assertEqual(
            dynamic_m.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                TreeSatDataset.num_timesteps,
                len(DYNAMIC_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(dynamic_x)))

    def check_static_shape(self, static_x, static_m):
        self.assertEqual(
            static_x.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                len(STATIC_BANDS),
            ),
        )
        self.assertEqual(
            static_m.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                len(STATIC_BAND_GROUPS_IDX),
            ),
        )

        # no static data in eurosat so added as zeros and masked out
        self.assertTrue(torch.all(static_x == 0))
        self.assertTrue(torch.all(static_m == 1))

    def check_month(self, month):
        self.assertEqual(month.shape, (TreeSatDataset.num_timesteps,))
        # no month in eurosat so set to zero
        self.assertEqual(month[0], TreeSatDataset.start_month)

    def test_treesat_dataset_s2(self):
        dataset = TreeSatDataset(mode="s2", split="train")
        dataset.images = [TEST_FILE]
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]

        self.check_dynamic_shape(dynamic_x=d_x, dynamic_m=d_m)
        self.check_static_shape(static_x=s_x, static_m=s_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))
