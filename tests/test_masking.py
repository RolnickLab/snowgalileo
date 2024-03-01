import unittest

import numpy as np

from src.data.masking import subset_image


class TestMasking(unittest.TestCase):
    def test_subset_image_with_minimum_size(self):
        input = np.ones((3, 3, 1))
        output = subset_image(input, input, 3)
        self.assertTrue(np.equal(input, output).all())

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        self.assertRaises(AssertionError, subset_image, input, input, 3)

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        output = subset_image(input, input, 3)
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output).all())
