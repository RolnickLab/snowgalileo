import unittest

import torch
from einops import repeat

from src.data_augmentation import FlipAndRotateSpace


class TestAugmentation(unittest.TestCase):
    def test_flip_and_rotate_space(self):
        aug = FlipAndRotateSpace(enabled=True)
        space_x = torch.randn(100, 10, 10, 3)  # (b, h, w, c)
        space_time_high_x = repeat(space_x.clone(), "b h w c -> b h w t c", t=8)
        space_time_med_x = torch.randn(100, 3, 3, 8, 3)
        space_time_low_x = torch.randn(100, 2, 2, 8, 3)
        valid_data_mask_s_t_h = torch.ones_like(space_time_high_x)  # (b, h, w, t, c)
        valid_data_mask_s_t_m = torch.ones_like(space_time_med_x)  # (b, h, w, t, c)
        valid_data_mask_s_t_l = torch.ones_like(space_time_low_x)  # (b, h, w, t, c)
        valid_data_mask_sp = torch.ones_like(space_x)  # (b, h, w, c)
        (
            new_space_time_high_x,
            new_space_time_med_x,
            new_space_time_low_x,
            new_space_x,
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
        ) = aug.apply(
            space_time_high_x,
            space_time_med_x,
            space_time_low_x,
            space_x,
            valid_data_mask_s_t_h=valid_data_mask_s_t_h,
            valid_data_mask_s_t_m=valid_data_mask_s_t_m,
            valid_data_mask_s_t_l=valid_data_mask_s_t_l,
            valid_data_mask_sp=valid_data_mask_sp,
        )

        # check that space_x and space_time_x are transformed the *same* way
        self.assertTrue(torch.equal(new_space_time_high_x.mean(dim=-2), new_space_x))

        # check that tensors were changed when flip+rotate=True
        self.assertFalse(torch.equal(new_space_time_high_x, space_time_high_x))
        self.assertFalse(torch.equal(new_space_time_med_x, space_time_med_x))
        self.assertFalse(torch.equal(new_space_time_low_x, space_time_low_x))
        self.assertFalse(torch.equal(new_space_x, space_x))

        aug = FlipAndRotateSpace(enabled=False)
        space_x = torch.randn(100, 10, 10, 3)  # (b, h, w, c)
        space_time_high_x = repeat(space_x.clone(), "b h w c -> b h w t c", t=8)
        space_time_med_x = torch.randn(100, 3, 3, 8, 3)
        space_time_low_x = torch.randn(100, 2, 2, 8, 3)
        valid_data_mask_s_t_h = torch.ones_like(space_time_high_x)
        valid_data_mask_s_t_m = torch.ones_like(space_time_med_x)
        valid_data_mask_s_t_l = torch.ones_like(space_time_low_x)
        valid_data_mask_sp = torch.ones_like(space_x)
        new_space_time_x, new_space_x, valid_data_mask_s_t_h, valid_data_mask_sp = aug.apply(
            space_time_high_x,
            space_time_med_x,
            space_time_low_x,
            space_x,
            valid_data_mask_s_t_h=valid_data_mask_s_t_h,
            valid_data_mask_s_t_m=valid_data_mask_s_t_m,
            valid_data_mask_s_t_l=valid_data_mask_s_t_l,
            valid_data_mask_sp=valid_data_mask_sp,
        )

        # check that tensors were not changed when flip+rotate=False
        self.assertTrue(torch.equal(new_space_time_x, space_time_high_x))
        self.assertTrue(torch.equal(new_space_time_med_x, space_time_med_x))
        self.assertTrue(torch.equal(new_space_time_low_x, space_time_low_x))
        self.assertTrue(torch.equal(new_space_x, space_x))
