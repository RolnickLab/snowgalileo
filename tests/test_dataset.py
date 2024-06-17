import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS, Dataset

BROKEN_FILE = "min_lat=24.7979_min_lon=-105.1508_max_lat=24.8069_max_lon=-105.141_dates=2022-01-01_2023-12-31.tif"
TEST_FILENAMES = [
    "min_lat=5.4427_min_lon=101.4016_max_lat=5.4518_max_lon=101.4107_dates=2022-01-01_2023-12-31.tif",
    "min_lat=-27.6721_min_lon=25.6796_max_lat=-27.663_max_lon=25.6897_dates=2022-01-01_2023-12-31.tif",
]
TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs"
TEST_FILES = [TIFS_FOLDER / x for x in TEST_FILENAMES]


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        for test_file in TEST_FILES:
            s_t_x, sp_x, t_x, st_x, months = Dataset._tif_to_array(test_file)
            self.assertFalse(np.isnan(s_t_x).any())
            self.assertFalse(np.isnan(sp_x).any())
            self.assertFalse(np.isnan(t_x).any())
            self.assertFalse(np.isnan(st_x).any())
            self.assertFalse(np.isinf(s_t_x).any())
            self.assertFalse(np.isinf(sp_x).any())
            self.assertFalse(np.isinf(t_x).any())
            self.assertFalse(np.isinf(st_x).any())
            self.assertEqual(sp_x.shape[0], s_t_x.shape[0])
            self.assertEqual(sp_x.shape[1], s_t_x.shape[1])
            self.assertEqual(t_x.shape[0], s_t_x.shape[2])
            self.assertEqual(len(SPACE_TIME_BANDS), s_t_x.shape[-1])
            self.assertEqual(len(SPACE_BANDS), sp_x.shape[-1])
            self.assertEqual(len(TIME_BANDS), t_x.shape[-1])
            self.assertEqual(len(STATIC_BANDS), st_x.shape[-1])
            self.assertEqual(months[0], 0)

    def test_files_are_replaced(self):
        ds = Dataset(TIFS_FOLDER, download=False)
        assert TIFS_FOLDER / BROKEN_FILE in ds.tifs

        for b in ds:
            assert len(b) == 5
        assert TIFS_FOLDER / BROKEN_FILE not in ds.tifs

    def test_subset_image_with_minimum_size(self):
        input = np.ones((3, 3, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, months, static, months, 3, 1)
        self.assertTrue(np.equal(input, output[0]).all())
        self.assertTrue(np.equal(input, output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        months = static = np.ones(1)
        self.assertRaises(
            AssertionError, Dataset.subset_image, input, input, months, static, months, 3, 1
        )

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, months, static, months, 3, 1)
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[0]).all())
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())
