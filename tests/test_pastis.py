import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
)
from src.eval.pastis_eval import PastisDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/pastis/pastis_test"


class TestPastis(unittest.TestCase):
    def check_dynamic(self, dynamic_x, dynamic_m, num_timesteps):
        self.assertEqual(
            dynamic_x.shape,
            (
                PastisDataset.input_height_width // 4,
                PastisDataset.input_height_width // 4,
                num_timesteps,
                len(DYNAMIC_BANDS),
            ),
        )
        self.assertEqual(
            dynamic_m.shape,
            (
                PastisDataset.input_height_width // 4,
                PastisDataset.input_height_width // 4,
                num_timesteps,
                len(DYNAMIC_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(dynamic_x)))

    def check_static(self, static_x, static_m):
        self.assertEqual(
            static_x.shape,
            (
                PastisDataset.input_height_width // 4,
                PastisDataset.input_height_width // 4,
                len(STATIC_BANDS),
            ),
        )
        self.assertEqual(
            static_m.shape,
            (
                PastisDataset.input_height_width // 4,
                PastisDataset.input_height_width // 4,
                len(STATIC_BAND_GROUPS_IDX),
            ),
        )

        # no static data in pastis so added as zeros and masked out
        self.assertTrue(torch.all(static_x == 0))
        self.assertTrue(torch.all(static_m == 1))

    def check_month(self, month, num_timesteps):
        self.assertEqual(month.shape, (num_timesteps,))

    def check_target(self, labels):
        self.assertTrue(
            torch.all(torch.isin(labels, torch.tensor(list(PastisDataset.labels_to_int.values()))))
        )

    def test_pastis_dataset(self):
        dataset = PastisDataset(folds=[1, 2, 3], data_path=DATA_FOLDER)
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]
        num_timesteps = d_x.shape[2]

        self.check_dynamic(dynamic_x=d_x, dynamic_m=d_m, num_timesteps=num_timesteps)
        self.check_static(static_x=s_x, static_m=s_m)
        self.check_month(month=m, num_timesteps=num_timesteps)
        self.check_target(labels=label)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        self.assertTrue(torch.all(d_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(d_m[:, :, :, unpresent_bands] == 1))
