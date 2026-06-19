import os
import unittest
from pathlib import Path

from src.fsc.utils import extract_season_from_filename

DATA_FOLDER = Path(__file__).parents[1] / "data/eval_tifs"


class TestRetrieveSeasonFromFilename(unittest.TestCase):
    def test_map_int_to_cloud_states(self):
        # Test cases: (filename, expected_season)
        test_cases = [
            ("LC_20201216_lat_lon.tif", "mid"),
            ("LC_20201215_lat_lon.tif", "early"),
            ("LC_20200630_lat_lon.tif", "late"),
            ("LC_20200110_lat_lon.tif", "mid"),
            ("LC_20221001_lat_lon.tif", "early"),
            ("LC_20240731_lat_lon.tif", "out_of_range"),
            ("LC_20210228_lat_lon.tif", "mid"),
        ]

        for filename, expected_season in test_cases:
            season = extract_season_from_filename(filename)
            self.assertEqual(season, expected_season)

        expected_test_seasons = ["mid", "late"]

        for idx, filename in enumerate(sorted(os.listdir(DATA_FOLDER))):
            season = extract_season_from_filename(filename)
            self.assertEqual(season, expected_test_seasons[idx])


if __name__ == "__main__":
    unittest.main()
