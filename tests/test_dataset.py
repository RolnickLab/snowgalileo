import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import DYNAMIC_BANDS, STATIC_BANDS, Dataset

TEST_FILE = (
    Path(__file__).parents[1]
    / "data/tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2022-01-01_2023-12-31.tiff"
)


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        dynamic_data, static_data = Dataset.tif_to_array(TEST_FILE)
        self.assertEqual(static_data.shape[0], dynamic_data.shape[0])
        self.assertEqual(static_data.shape[1], dynamic_data.shape[1])
        self.assertEqual(len(DYNAMIC_BANDS), dynamic_data.shape[-1])
        self.assertEqual(len(STATIC_BANDS), static_data.shape[-1])

        # one way to check this is correct is to see if all the DYNAMIC_WORLD bands
        # sum to 1
        dynamic_world_bands = [x for x in DYNAMIC_BANDS if x.startswith("DW_")]
        dynamic_world_only = dynamic_data[:, :, :, -len(dynamic_world_bands) - 1 : -1].sum(axis=-1)
        self.assertTrue(
            np.allclose(dynamic_world_only[~np.isnan(dynamic_world_only)], 1, atol=0.01)
        )
