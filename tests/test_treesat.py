import unittest

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
from src.eval.treesat_eval import TreeSatDataset

TEST_FILE = "Tilia_spec._9_99911_WEFL_NLF.tif"


class TestTreeSat(unittest.TestCase):
    def check_space_time(self, s_t_x, s_t_m):
        self.assertEqual(
            s_t_x.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                TreeSatDataset.num_timesteps,
                len(SPACE_TIME_BANDS),
            ),
        )
        self.assertEqual(
            s_t_m.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                TreeSatDataset.num_timesteps,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(s_t_x)))

    def check_space(self, sp_x, sp_m):
        self.assertEqual(
            sp_x.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
                len(SPACE_BANDS),
            ),
        )
        self.assertEqual(
            sp_m.shape,
            (
                TreeSatDataset.input_height_width,
                TreeSatDataset.input_height_width,
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
                TreeSatDataset.num_timesteps,
                len(TIME_BANDS),
            ),
        )
        self.assertEqual(
            t_m.shape,
            (
                TreeSatDataset.num_timesteps,
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

        self.assertFalse(torch.any(torch.isnan(st_x)))

    def check_month(self, month):
        self.assertEqual(month.shape, (TreeSatDataset.num_timesteps,))
        self.assertEqual(month[0], TreeSatDataset.start_month)

    def test_treesat_dataset_s2(self):
        dataset = TreeSatDataset(mode="s2", split="train")
        dataset.images = [TEST_FILE]
        sample = dataset[0]
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, m = sample[0]

        self.check_space_time(s_t_x, s_t_m)
        self.check_space(sp_x, sp_m)
        self.check_time(t_x, t_m)
        self.check_static(st_x, st_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        absent_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" not in key
        ]
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS) if "B" in key]

        self.assertTrue(torch.all(s_t_x[:, :, :, present_bands] != 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, present_band_groups] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, absent_band_groups] == 1))

    def test_treesat_dataset_s1(self):
        dataset = TreeSatDataset(mode="s1", split="train")
        dataset.images = [TEST_FILE]
        sample = dataset[0]
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, m = sample[0]

        self.check_space_time(s_t_x, s_t_m)
        self.check_space(sp_x, sp_m)
        self.check_time(t_x, t_m)
        self.check_static(st_x, st_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S1" in key
        ]
        absent_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S1" not in key
        ]
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS) if key in ["VV", "VH"]]

        self.assertTrue(torch.all(s_t_x[:, :, :, present_bands] != 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, present_band_groups] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, absent_band_groups] == 1))

    def test_treesat_dataset_combined(self):
        dataset = TreeSatDataset(mode="combined", split="train")
        dataset.images = [TEST_FILE]
        sample = dataset[0]
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, m = sample[0]

        self.check_space_time(s_t_x, s_t_m)
        self.check_space(sp_x, sp_m)
        self.check_time(t_x, t_m)
        self.check_static(st_x, st_m)
        self.check_month(month=m)

        # will test if the right channels are masked out
        present_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key
        ]
        absent_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" not in key
        ]
        present_bands = [
            idx
            for idx, key in enumerate(SPACE_TIME_BANDS)
            if (("B" in key) or (key in ["VV", "VH"]))
        ]
        self.assertTrue(torch.all(s_t_x[:, :, :, present_bands] != 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, present_band_groups] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, absent_band_groups] == 1))


if __name__ == "__main__":
    unittest.main()
