import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS, Dataset

TEST_FILE = (
    Path(__file__).parents[1]
    / "data/tifs/min_lat=-27.6721_min_lon=25.6796_max_lat=-27.663_max_lon=25.6897_dates=2022-01-01_2023-12-31.tif"
)


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        s_t_x, sp_x, t_x, st_x, months = Dataset._tif_to_array(TEST_FILE)
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
        # visual test with the filepath above. The assert
        # makes sure that file hasn't changed.
        assert "dates=2022-01-01_2023-12-31" in TEST_FILE.name
        self.assertEqual(months[0], 0)

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
