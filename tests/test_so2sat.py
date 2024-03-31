import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
)
from src.eval.so2sat_eval import So2SatDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/so2sat/so2sat_test"


class TestSo2Sat(unittest.TestCase):
    def check_dynamic(self, dynamic_x, dynamic_m):
        self.assertEqual(
            dynamic_x.shape,
            (
                So2SatDataset.input_height_width,
                So2SatDataset.input_height_width,
                So2SatDataset.num_timesteps,
                len(DYNAMIC_BANDS),
            ),
        )
        self.assertEqual(
            dynamic_m.shape,
            (
                So2SatDataset.input_height_width,
                So2SatDataset.input_height_width,
                So2SatDataset.num_timesteps,
                len(DYNAMIC_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(dynamic_x)))

    def check_static(self, static_x, static_m):
        self.assertEqual(
            static_x.shape,
            (
                So2SatDataset.input_height_width,
                So2SatDataset.input_height_width,
                len(STATIC_BANDS),
            ),
        )
        self.assertEqual(
            static_m.shape,
            (
                So2SatDataset.input_height_width,
                So2SatDataset.input_height_width,
                len(STATIC_BAND_GROUPS_IDX),
            ),
        )

        # no static data in so2sat so added as zeros and masked out
        self.assertTrue(torch.all(static_x == 0))
        self.assertTrue(torch.all(static_m == 1))

    def check_month(self, month):
        self.assertEqual(month.shape, (So2SatDataset.num_timesteps,))
        # no month in so2sat so set to zero
        self.assertEqual(month[0], 0)

    def check_label(self, label):
        # labels are one-hot encoded
        self.assertTrue(torch.all(torch.logical_or(label == 0, label == 1)))

    def test_so2sat_dataset(self):
        dataset = So2SatDataset(split="validation", so2sat_dir=DATA_FOLDER)
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]

        self.check_dynamic(dynamic_x=d_x, dynamic_m=d_m)
        self.check_static(static_x=s_x, static_m=s_m)
        self.check_month(month=m)
        self.check_label(label=label)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))
        print("passed")


if __name__ == "__main__":
    TestSo2Sat().test_so2sat_dataset()
