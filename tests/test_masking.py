import unittest

import numpy as np

from src.masked_datasets import (
    CROMA_INPUT_SIZE,
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_TIMESTEPS,
    STATIC_BAND_GROUPS_IDX,
    VIT_PATCH_SIZE,
    PrestoToPrestoMaskedDataset,
    mask_by_croma_blocks_random,
    mask_by_croma_spatial_blocks,
    subset_image,
)


class TestMasking(unittest.TestCase):
    def test_subset_image_with_minimum_size(self):
        input = np.ones((3, 3, 1))
        months = np.ones(1)
        output = subset_image(input, input, months, 3, 1)
        self.assertTrue(np.equal(input, output[0]).all())
        self.assertTrue(np.equal(input, output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        months = np.ones(1)
        self.assertRaises(AssertionError, subset_image, input, input, months, 3, 1)

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        months = np.ones(1)
        output = subset_image(input, input, months, 3, 1)
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[0]).all())
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())

    def test_mask_by_croma_blocks_spatial(self):
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 24, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        months = np.arange(0, 24)
        mask_ratio = 0.25

        output = mask_by_croma_spatial_blocks(dynamic_input, static_input, months, mask_ratio)
        self.assertEqual(np.sum(output.dynamic_mask) / output.dynamic_mask.size, mask_ratio)
        self.assertEqual(np.sum(output.static_mask) / output.static_mask.size, mask_ratio)
        self.assertEqual(len(output.months), NUM_TIMESTEPS)
        self.assertEqual(output.dynamic_mask.shape[2], NUM_TIMESTEPS)

    def test_mask_by_croma_blocks_random(self):
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 24, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        months = np.arange(0, 24)
        mask_ratio = 0.25

        output = mask_by_croma_blocks_random(dynamic_input, static_input, months, mask_ratio)

        num_dynamic_tokens_masked = np.sum(output.dynamic_mask) / (VIT_PATCH_SIZE**2)
        total_dynamic_tokens = output.dynamic_mask.size / (VIT_PATCH_SIZE**2)
        num_static_tokens_masked = np.sum(output.static_mask) / (VIT_PATCH_SIZE**2)
        total_static_tokens = output.static_mask.size / (VIT_PATCH_SIZE**2)

        self.assertEqual(
            (num_dynamic_tokens_masked + num_static_tokens_masked)
            / (total_dynamic_tokens + total_static_tokens),
            mask_ratio,
        )

    def test_mask_by_presto_pixels_time(self):
        num_timesteps = 24
        dynamic_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, num_timesteps, 8))
        static_input = np.ones((CROMA_INPUT_SIZE + 15, CROMA_INPUT_SIZE, 8))
        months = np.arange(0, num_timesteps)
        mask_ratio = 0.25

        output = PrestoToPrestoMaskedDataset.mask_by_presto_pixels_time(
            dynamic_input, static_input, months, mask_ratio
        )

        # collapse the dynamic_mask along the time dimension
        dynamic_mask_along_t = output.dynamic_mask.mean(axis=(0, 1, 3))
        self.assertTrue(np.isin(dynamic_mask_along_t, (0, 1)).all())
        self.assertEqual(sum(dynamic_mask_along_t) / len(dynamic_mask_along_t), 0.25)
