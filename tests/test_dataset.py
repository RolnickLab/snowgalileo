import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import DYNAMIC_BANDS, Dataset

TEST_FILE = (
    Path(__file__).parents[1]
    / "data/tifs_min_lat=18.9712_min_lon=-97.0113_max_lat=18.9838_max_lon=-96.9981_dates=2022-01-01_2023-12-31.tiff"
)


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        dynamic_data, static_data = Dataset.tif_to_array(TEST_FILE)
        self.assertEqual(static_data.shape[1], dynamic_data.shape[2])
        self.assertEqual(static_data.shape[2], dynamic_data.shape[3])

        # one way to check this is correct is to see if all the DYNAMIC_WORLD bands
        # sum to 1
        dynamic_world_bands = [x for x in DYNAMIC_BANDS if x.startswith("DW_")]
        dynamic_world_only = dynamic_data[-len(dynamic_world_bands) :].sum(axis=0)
        self.assertTrue(
            np.allclose(dynamic_world_only[~np.isnan(dynamic_world_only)], 1, atol=0.01)
        )
