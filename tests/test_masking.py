import unittest

import numpy as np

from src.data.masking import CROMA_INPUT_SIZE, mask_presto_to_croma, subset_image


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

    def test_mask_presto_to_croma(self):
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        mask_ratio = 0.25

        output = mask_presto_to_croma(dynamic_input, static_input, mask_ratio)
        self.assertEqual(np.sum(output.dynamic_mask) / output.dynamic_mask.size, mask_ratio)
        self.assertEqual(np.sum(output.static_mask) / output.static_mask.size, mask_ratio)
