import os
import unittest
from pathlib import Path

from src.data.config import NUM_TIMESTEPS
from src.eval.utils import retrieve_season_from_filename

DATA_FOLDER = Path(__file__).parents[1] / "data/eval_tifs"

class TestRetrieveSeasonFromFilename(unittest.TestCase):
    def test_map_int_to_cloud_states(self):
        # Test cases: (filename, expected_season)
        test_cases = [
            ("LC_20201216_FSC_lat_lon.tif", "mid"),
            ("LC_20201215_FSC_lat_lon.tif", "early"),
            ("LC_20200630_FSC_lat_lon.tif", "late"),
            ("LC_20200110_FSC_lat_lon.tif", "mid"),
            ("LC_20221001_FSC_lat_lon.tif", "early"),
            ("LC_20240731_FSC_lat_lon.tif", "late"),
            ("LC_20210228_FSC_lat_lon.tif", "mid"),
        ]

        for filename, expected_season in test_cases:
            season = retrieve_season_from_filename(filename)
            self.assertEqual(season, expected_season)

        expected_test_seasons = ["mid", "late"]
        
        for idx, filename in enumerate(os.listdir(DATA_FOLDER)):
            season = retrieve_season_from_filename(filename)
            self.assertEqual(season, expected_test_seasons[idx])