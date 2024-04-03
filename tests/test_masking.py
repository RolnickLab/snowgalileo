import unittest

import numpy as np
import torch
from einops import repeat

from src.masking import (
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_TIMESTEPS,
    STATIC_BAND_GROUPS_IDX,
    batch_mask_presto,
    batch_mask_space,
    batch_mask_time,
)


class TestMasking(unittest.TestCase):
    def test_mask_by_time(self):
        b, t, h, w = 2, NUM_TIMESTEPS, 16, 16
        dynamic_input = torch.ones((b, h, w, t, 8))
        static_input = torch.ones((b, h, w, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_mask_time(dynamic_input, static_input, months, mask_ratio)
        self.assertEqual((b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX)), output.dynamic_mask.shape)
        self.assertEqual((b, h, w, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
        # collapse the dynamic_mask along the time dimension
        dynamic_mask_along_t = output.dynamic_mask.mean(axis=(1, 2, 4))  # b, t
        self.assertTrue(np.isin(dynamic_mask_along_t, (0, 1)).all())
        self.assertTrue(
            (dynamic_mask_along_t.sum(axis=1) / dynamic_mask_along_t.shape[1] == mask_ratio).all()
        )

    def test_mask_by_space(self):
        b, t, h, w, p = 2, NUM_TIMESTEPS, 16, 16, 4
        dynamic_input = torch.ones((b, h, w, t, 8))
        static_input = torch.ones((b, h, w, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_mask_space(dynamic_input, static_input, months, mask_ratio, p)
        self.assertEqual((b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX)), output.dynamic_mask.shape)
        self.assertEqual((b, h, w, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
        # collapse the masks along h, w dimensions
        d_along_hw = output.dynamic_mask.mean(axis=(3, 4))  # b, h, w
        s_along_hw = output.static_mask.mean(axis=(3))  # b, h, w
        self.assertTrue(torch.equal(d_along_hw, s_along_hw))
        self.assertTrue((d_along_hw.sum(axis=1).sum(axis=1) / (h * w) == mask_ratio).all())

        for i in range(1, p):
            self.assertTrue(
                torch.equal(s_along_hw[:, i::p, i::p], s_along_hw[:, i - 1 :: p, i - 1 :: p])
            )

    def test_mask_combined(self):
        b, t, h, w, p = 2, NUM_TIMESTEPS, 16, 16, 4
        dynamic_input = torch.ones((b, h, w, t, 8))
        static_input = torch.ones((b, h, w, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_mask_presto(
            dynamic_input, static_input, months, mask_ratio, p, time_ratio=0.5
        )
        self.assertEqual((b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX)), output.dynamic_mask.shape)
        self.assertEqual((b, h, w, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
