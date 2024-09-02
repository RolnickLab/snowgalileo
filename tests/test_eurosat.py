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
from src.eval.eurosat_eval import EuroSatDataset, EuroSatEval
from src.masking import UNMASKING_CHANNEL_GROUPS

DATA_FOLDER = Path(__file__).parents[1] / "data/eurosat/eurosat_test"


class TestEuroSat(unittest.TestCase):
    def check_space_time(self, s_t_x, s_t_m):
        self.assertEqual(
            s_t_x.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                EuroSatDataset.num_timesteps,
                len(SPACE_TIME_BANDS),
            ),
        )
        self.assertEqual(
            s_t_m.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                EuroSatDataset.num_timesteps,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
            ),
        )
        self.assertFalse(torch.any(torch.isnan(s_t_x)))

    def check_space(self, sp_x, sp_m):
        self.assertEqual(
            sp_x.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                len(SPACE_BANDS),
            ),
        )
        self.assertEqual(
            sp_m.shape,
            (
                EuroSatDataset.input_height_width,
                EuroSatDataset.input_height_width,
                len(SPACE_BAND_GROUPS_IDX),
            ),
        )

        # no static data in eurosat so added as zeros and masked out
        self.assertTrue(torch.all(sp_x == 0))
        self.assertTrue(torch.all(sp_m == 1))

    def check_time(self, t_x, t_m):
        self.assertEqual(
            t_x.shape,
            (
                EuroSatDataset.num_timesteps,
                len(TIME_BANDS),
            ),
        )
        self.assertEqual(
            t_m.shape,
            (
                EuroSatDataset.num_timesteps,
                len(TIME_BAND_GROUPS_IDX),
            ),
        )

        # no time-only data in eurosat so added as zeros and masked out
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
        self.assertEqual(month.shape, (EuroSatDataset.num_timesteps,))
        # no month in eurosat so set to zero
        self.assertEqual(month[0], 0)

    def check_label(self, label):
        self.assertTrue(label in EuroSatDataset.labels_to_int.values())

    def test_eurosat_dataset_rgb(self):
        dataset = EuroSatDataset(
            normalizer=EuroSatEval.load_eurosat_normalizer(),
            rgb=True,
            split="test",
            tif_files_dir=DATA_FOLDER,
        )
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
        present_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2_RGB" in key
        ]
        unpresent_bands = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2_RGB" not in key
        ]

        self.assertTrue(torch.all(s_t_m[:, :, :, present_bands] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, unpresent_bands] == 1))

    def test_eurosat_dataset_msi(self):
        dataset = EuroSatDataset(
            normalizer=EuroSatEval.load_eurosat_normalizer(),
            rgb=False,
            split="test",
            tif_files_dir=DATA_FOLDER,
        )
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
        present_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
        ]
        unpresent_band_groups = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" not in key
        ]
        present_bands = [idx for idx, key in enumerate(SPACE_TIME_BANDS) if key.startswith("B")]

        self.assertTrue(torch.all(s_t_x[:, :, :, present_bands] != 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, present_band_groups] == 0))
        self.assertTrue(torch.all(s_t_m[:, :, :, unpresent_band_groups] == 1))

    def test_eurosat_conditions(self):
        task = EuroSatEval(normalization="std")

        self.assertEqual(len(task.condition["output_channels"]), len(UNMASKING_CHANNEL_GROUPS))

        for idx, val in enumerate(task.condition["output_channels"]):
            if val == 1:
                self.assertTrue(UNMASKING_CHANNEL_GROUPS[idx] == ("static", "DW_static"))
