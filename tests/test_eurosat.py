import unittest

import torch

from eval.eurosat_eval import EuroSatDataset
from src.data.dataset import DYNAMIC_BANDS_GROUPS_IDX


class TestEuroSat(unittest.TestCase):
    def test_eurosat_dataset_rgb(self):
        dataset = EuroSatDataset(
            rgb=True,
            split="validation",
        )
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]
        # input shape expected
        self.assertEqual(d_x.shape, (64, 64, 1, 24))
        self.assertEqual(s_x.shape, (64, 64, 2))
        self.assertEqual(d_m.shape, (64, 64, 1, 9))
        self.assertEqual(s_m.shape, (64, 64, 1))
        # eurosat has only one timestep
        self.assertEqual(m.shape, (1,))
        self.assertFalse(torch.any(torch.isnan(d_x)))
        # no month in eurosat so set to zero
        self.assertEqual(m[0], 0)
        self.assertTrue(torch.all(torch.logical_or(d_m == 0, s_m == 1)))
        # no static data in eurosat so added as zeros and masked out
        self.assertTrue(torch.all(s_x == 0))
        self.assertTrue(torch.all(s_m == 1))
        # labels are one-hot encoded
        self.assertTrue(torch.all(torch.logical_or(label == 0, label == 1)))

        # will test if the right channels are masked out
        present_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2_RGB" in key
        ]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2_RGB" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))

    def test_eurosat_dataset_msi(self):
        dataset = EuroSatDataset(
            rgb=False,
            split="validation",
        )
        sample = dataset[0]
        _, _, d_m, _, _ = sample[0]

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))
