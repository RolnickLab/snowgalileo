import unittest
from pathlib import Path

import torch

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
)
from src.eval.so2sat_eval import So2SatBaseDataset, So2SatGeobenchDataset, So2SatTUMDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/so2sat/so2sat_test"


class TestSo2Sat(unittest.TestCase):
    def check_space_time(self, s_t_x, s_t_m):
        self.assertEqual(
            s_t_x.shape,
            (
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.num_timesteps,
                len(SPACE_TIME_BANDS),
            ),
        )
        self.assertEqual(
            s_t_m.shape,
            (
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.num_timesteps,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(s_t_x)))

    def check_space(self, sp_x, sp_m):
        self.assertEqual(
            sp_x.shape,
            (
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.input_height_width,
                len(SPACE_BANDS),
            ),
        )
        self.assertEqual(
            sp_m.shape,
            (
                So2SatBaseDataset.input_height_width,
                So2SatBaseDataset.input_height_width,
                len(SPACE_BAND_GROUPS_IDX),
            ),
        )

        # no static data so added as zeros and masked out
        self.assertTrue(torch.all(sp_x == 0))
        self.assertTrue(torch.all(sp_m == 1))

    def check_time(self, t_x, t_m):
        self.assertEqual(
            t_x.shape,
            (
                So2SatBaseDataset.num_timesteps,
                len(TIME_BANDS),
            ),
        )
        self.assertEqual(
            t_m.shape,
            (
                So2SatBaseDataset.num_timesteps,
                len(TIME_BAND_GROUPS_IDX),
            ),
        )

        # no time-only data so added as zeros and masked out
        self.assertTrue(torch.all(t_x == 0))
        self.assertTrue(torch.all(t_m == 1))

    def check_static(self, st_x, st_m):
        self.assertEqual(
            st_x.shape,
            (len(STATIC_BANDS),),
        )
        self.assertEqual(
            st_m.shape,
            (len(STATIC_BAND_GROUPS_IDX),),
        )

        # no static data so added as zeros and masked out
        self.assertTrue(torch.all(st_x == 0))
        self.assertTrue(torch.all(st_m == 1))

    def check_month(self, month):
        self.assertEqual(month.shape, (So2SatBaseDataset.num_timesteps,))
        # no month in so2sat so set to zero
        self.assertEqual(month[0], 0)

    def check_label(self, label):
        # labels are one-hot encoded
        self.assertTrue(torch.all(torch.logical_or(label == 0, label == 1)))

    def test_so2sat_geobench_dataset(self):
        dataset = So2SatGeobenchDataset(split="test", so2sat_dir=DATA_FOLDER)
        sample = dataset[0]
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, m = sample[0]
        label = sample[1]
        self.check_space_time(s_t_x, s_t_m)
        self.check_space(sp_x, sp_m)
        self.check_time(t_x, t_m)
        self.check_static(st_x, st_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" not in key
        ]

        self.assertTrue(torch.all(s_t_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, unpresent_bands] == 1))

    def test_so2sat_tum_dataset(self):
        dataset = So2SatTUMDataset(split="testing", so2sat_dir=DATA_FOLDER)
        sample = dataset[0]
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, m = sample[0]
        label = sample[1]
        self.check_space_time(s_t_x, s_t_m)
        self.check_space(sp_x, sp_m)
        self.check_time(t_x, t_m)
        self.check_static(st_x, st_m)
        self.check_month(month=m)
        self.check_label(label=label)

        # will test if the right channels are masked out
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key]
        unpresent_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" not in key
        ]

        self.assertTrue(torch.all(s_t_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, unpresent_bands] == 1))
