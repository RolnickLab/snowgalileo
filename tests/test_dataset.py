import unittest

from pathlib import Path
from src.data.dataset import Dataset


TEST_FILE = Path(__file__).parents[1] / "data/tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2021-01-01_2023-12-31.tiff"

class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        if not TEST_FILE.exists():
            return None
        dynamic_data, static_data = Dataset.tif_to_array(TEST_FILE)
        self.assertEqual(static_data.shape[1], dynamic_data.shape[2])
