import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
)
from src.eval.eurosat_eval import EuroSatDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/eurosat/eurosat_test"


class TestEuroSat(unittest.TestCase):
    def check_dynamic(self, dynamic_x, dynamic_m):
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
                len(DYNAMIC_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(dynamic_x)))

    def check_static(self, static_x, static_m):
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
                len(STATIC_BAND_GROUPS_IDX),
            ),
        )

        # no static data in eurosat so added as zeros and masked out
        self.assertTrue(torch.all(static_x == 0))
        self.assertTrue(torch.all(static_m == 1))

    def check_month(self, month):
        self.assertEqual(month.shape, (EuroSatDataset.num_timesteps,))
        # no month in eurosat so set to zero
        self.assertEqual(month[0], 0)

    def test_eurosat_dataset_rgb(self):
        dataset = EuroSatDataset(rgb=True, split="test", tif_files_dir=DATA_FOLDER)
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]

        self.check_dynamic(dynamic_x=d_x, dynamic_m=d_m)
        self.check_static(static_x=s_x, static_m=s_m)
        self.check_month(month=m)

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
        dataset = EuroSatDataset(rgb=False, split="test", tif_files_dir=DATA_FOLDER)
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]

        self.check_dynamic(dynamic_x=d_x, dynamic_m=d_m)
        self.check_static(static_x=s_x, static_m=s_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_band_groups = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key
        ]
        unpresent_band_groups = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" not in key
        ]
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS) if "B" in key]

        self.assertTrue(torch.all(d_x[:, :, :, present_bands] != 0))
        self.assertTrue(torch.all(d_m[:, :, :, present_band_groups] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_band_groups] == 1))
