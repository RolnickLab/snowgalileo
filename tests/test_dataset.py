import unittest

import numpy as np

from src.data.config import DATA_FOLDER
from src.data.dataset import DYNAMIC_BANDS, STATIC_BANDS, Dataset

TEST_FILE = (
    DATA_FOLDER
    / "test_files"
    / "presto_tif"
    / "tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2022-01-01_2023-12-31.tif"
)


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        dynamic_data, static_data, months = Dataset._tif_to_array(TEST_FILE)
        self.assertFalse(np.isnan(dynamic_data).any())
        self.assertFalse(np.isnan(static_data).any())
        self.assertEqual(static_data.shape[0], dynamic_data.shape[0])
        self.assertEqual(static_data.shape[1], dynamic_data.shape[1])
        self.assertEqual(len(DYNAMIC_BANDS), dynamic_data.shape[-1])
        self.assertEqual(len(STATIC_BANDS), static_data.shape[-1])
        # visual test with the filepath above. The assert
        # makes sure that file hasn't changed.
        assert "dates=2022-01-01_2023-12-31" in TEST_FILE.name
        self.assertEqual(months[0], 0)
