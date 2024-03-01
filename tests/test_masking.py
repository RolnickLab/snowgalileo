import unittest

import numpy as np

from src.data.masking import (
    CROMA_INPUT_SIZE,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    VIT_PATCH_SIZE,
    mask_by_croma_blocks_random,
    mask_by_croma_spatial_blocks,
    subset_image,
)


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

    def test_mask_by_croma_blocks_spatial(self):
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        mask_ratio = 0.25

        output = mask_by_croma_spatial_blocks(dynamic_input, static_input, mask_ratio)
        self.assertEqual(np.sum(output.dynamic_mask) / output.dynamic_mask.size, mask_ratio)
        self.assertEqual(np.sum(output.static_mask) / output.static_mask.size, mask_ratio)

    def test_mask_by_croma_blocks_random(self):
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        mask_ratio = 0.25

        output = mask_by_croma_blocks_random(dynamic_input, static_input, mask_ratio)

        first_index_of_dynamic_band_group = [
            value[0] for _, value in DYNAMIC_BANDS_GROUPS_IDX.items()
        ]
        first_index_of_static_band_group = [
            value[0] for _, value in STATIC_BAND_GROUPS_IDX.items()
        ]

        dynamic_mask = output.dynamic_mask[:, :, :, first_index_of_dynamic_band_group]
        static_mask = output.static_mask[:, :, first_index_of_static_band_group]

        num_dynamic_tokens_masked = np.sum(dynamic_mask) / (VIT_PATCH_SIZE**2)
        total_dynamic_tokens = dynamic_mask.size / (VIT_PATCH_SIZE**2)
        num_static_tokens_masked = np.sum(static_mask) / (VIT_PATCH_SIZE**2)
        total_static_tokens = static_mask.size / (VIT_PATCH_SIZE**2)

        self.assertEqual(
            (num_dynamic_tokens_masked + num_static_tokens_masked)
            / (total_dynamic_tokens + total_static_tokens),
            mask_ratio,
        )
