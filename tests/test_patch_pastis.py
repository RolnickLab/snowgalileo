import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
)
from src.eval.pastis_patch_eval import PastisPatchDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/pastis/pastis_test"


class TestPastis(unittest.TestCase):
    def check_space_time(self, s_t_x, s_t_m, num_timesteps):
        self.assertEqual(
            s_t_x.shape,
            (
                PastisPatchDataset.input_height_width // 2,
                PastisPatchDataset.input_height_width // 2,
                num_timesteps,
                len(SPACE_TIME_BANDS),
            ),
        )
        self.assertEqual(
            s_t_m.shape,
            (
                PastisPatchDataset.input_height_width // 2,
                PastisPatchDataset.input_height_width // 2,
                num_timesteps,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(s_t_x)))

    def check_space(self, s_x, s_m):
        self.assertEqual(
            s_x.shape,
            (
                PastisPatchDataset.input_height_width // 2,
                PastisPatchDataset.input_height_width // 2,
                len(SPACE_BANDS),
            ),
        )
        self.assertEqual(
            s_m.shape,
            (
                PastisPatchDataset.input_height_width // 2,
                PastisPatchDataset.input_height_width // 2,
                len(SPACE_BAND_GROUPS_IDX),
            ),
        )

        # no static data so added as zeros and masked out
        self.assertTrue(torch.all(s_x == 0))
        self.assertTrue(torch.all(s_m == 1))

    def check_time(self, t_x, t_m, num_timesteps):
        self.assertEqual(
            t_x.shape,
            (
                num_timesteps,
                len(TIME_BANDS),
            ),
        )
        self.assertEqual(
            t_m.shape,
            (
                num_timesteps,
                len(TIME_BAND_GROUPS_IDX),
            ),
        )

        # no time-only data so added as zeros and masked out
        self.assertTrue(torch.all(t_x == 0))
        self.assertTrue(torch.all(t_m == 1))

    def check_month(self, month, num_timesteps):
        self.assertEqual(month.shape, (num_timesteps,))

    def check_target(self, labels):
        self.assertTrue(
            torch.all(
                torch.isin(labels, torch.tensor(list(PastisPatchDataset.labels_to_int.values())))
            )
        )

    def test_pastis_month_average(self):
        dataset = PastisPatchDataset(
            folds=[1, 2, 3], data_path=DATA_FOLDER, average_s2_over_month=True
        )
        sample = dataset[1]
        s_t_x, s_x, t_x, s_t_m, s_m, t_m, m = sample[0]
        label = sample[1]

        self.check_space_time(s_t_x=s_t_x, s_t_m=s_t_m, num_timesteps=12)
        self.check_space(s_x=s_x, s_m=s_m)
        self.check_time(t_x=t_x, t_m=t_m, num_timesteps=12)
        self.check_month(month=m, num_timesteps=12)
        self.check_target(labels=label)

        # will test if the right channels are masked out
        unpresent_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        self.assertTrue(torch.all(s_t_m[:, :, :, unpresent_bands] == 1))

    def test_pastis_max_timesteps(self):
        dataset = PastisPatchDataset(
            folds=[1, 2, 3], data_path=DATA_FOLDER, average_s2_over_month=False
        )
        sample = dataset[1]
        s_t_x, s_x, t_x, s_t_m, s_m, t_m, m = sample[0]
        label = sample[1]

        # max number of timesteps in pastis are 61, missing get padded and masked
        self.check_space_time(s_t_x=s_t_x, s_t_m=s_t_m, num_timesteps=61)
        self.check_space(s_x=s_x, s_m=s_m)
        self.check_time(t_x=t_x, t_m=t_m, num_timesteps=61)
        self.check_month(month=m, num_timesteps=61)
        self.check_target(labels=label)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" not in key
        ]

        # 38 is the minimum number of timesteps present in all observations
        self.assertTrue(torch.all(s_t_m[:, :, :38, present_bands] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :38, unpresent_bands] == 1))
        print("Finishing the test")
