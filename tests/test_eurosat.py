import unittest

import torch

from src.data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_DYNAMIC_BAND_GROUPS,
    NUM_STATIC_BAND_GROUPS,
    STATIC_BANDS,
)
from src.eval.eurosat_eval import EuroSatDataset


class TestEuroSat(unittest.TestCase):
    def test_dynamic(self, dynamic_x, dynamic_m):
        self.assertEqual(
            dynamic_x.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                EuroSatDataset.num_timesteps,
                len(DYNAMIC_BANDS),
            ),
        )
        self.assertEqual(
            dynamic_m.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                EuroSatDataset.num_timesteps,
                NUM_DYNAMIC_BAND_GROUPS,
            ),
        )
        self.assertFalse(torch.any(torch.isnan(dynamic_x)))

    def test_static(self, static_x, static_m):
        self.assertEqual(
            static_x.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                len(STATIC_BANDS),
            ),
        )
        self.assertEqual(
            static_m.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                NUM_STATIC_BAND_GROUPS,
            ),
        )

        # no static data in eurosat so added as zeros and masked out
        self.assertTrue(torch.all(static_x == 0))
        self.assertTrue(torch.all(static_m == 1))

    def test_month(self, month):
        self.assertEqual(month.shape, (EuroSatDataset.num_timesteps,))
        # no month in eurosat so set to zero
        self.assertEqual(month[0], 0)

    def test_label(self, label):
        # labels are one-hot encoded
        self.assertTrue(torch.all(torch.logical_or(label == 0, label == 1)))

    def test_eurosat_dataset_rgb(self):
        dataset = EuroSatDataset(
            rgb=True,
            split="test",
        )
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]

        self.test_dynamic(d_x, d_m)
        self.test_static(s_x, s_m)
        self.test_month(m)
        self.test_label(label)

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
            split="test",
        )
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]

        self.test_dynamic(d_x, d_m)
        self.test_static(s_x, s_m)
        self.test_month(m)
        self.test_label(label)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))
