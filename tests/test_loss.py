import unittest

import torch

from src.data.earthengine.eo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
)
from src.loss import mae_loss


class TestLoss(unittest.TestCase):
    def test_mae_loss(self):
        (
            b,
            t_h_h,
            t_h_w,
            t_m_h,
            t_m_w,
            t_l_h,
            t_l_w,
            t,
            patch_size_high_res,
            patch_size_med_res,
            patch_size_low_res,
        ) = 16, 4, 4, 3, 3, 2, 2, 3, 2, 1, 1
        pixel_h_h, pixel_h_w = t_h_h * patch_size_high_res, t_h_w * patch_size_high_res
        pixel_m_h, pixel_m_w = t_m_h * patch_size_med_res, t_m_w * patch_size_med_res
        pixel_l_h, pixel_l_w = t_l_h * patch_size_low_res, t_l_w * patch_size_low_res
        max_patch_size = 8
        max_group_length = max(
            [
                max([len(v) for _, v in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in TIME_BANDS_GROUPS_IDX.items()]),
                max([len(v) for _, v in SPACE_BAND_GROUPS_IDX.items()]),
                max([len(v) for _, v in STATIC_BAND_GROUPS_IDX.items()]),
            ]
        )
        p_s_t_h = torch.randn(
            (
                b,
                t_h_h,
                t_h_w,
                t,
                len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX),
                max_group_length * (max_patch_size**2),
            )
        )
        p_s_t_m = torch.randn(
            (
                b,
                t_m_h,
                t_m_w,
                t,
                len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX),
                max_group_length * (max_patch_size**2),
            )
        )
        p_s_t_l = torch.randn(
            (
                b,
                t_l_h,
                t_l_w,
                t,
                len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX),
                max_group_length * (max_patch_size**2),
            )
        )
        p_sp = torch.randn(
            (b, t_h_h, t_h_w, len(SPACE_BAND_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        p_t = torch.randn(
            (b, t, len(TIME_BANDS_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        p_st = torch.randn(
            (b, len(STATIC_BAND_GROUPS_IDX), max_group_length * (max_patch_size**2))
        )
        s_t_h_x = torch.randn(
            b,
            pixel_h_h,
            pixel_h_w,
            t,
            sum([len(x) for _, x in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.items()]),
        )
        s_t_m_x = torch.randn(
            b,
            pixel_m_h,
            pixel_m_w,
            t,
            sum([len(x) for _, x in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.items()]),
        )
        s_t_l_x = torch.randn(
            b,
            pixel_l_h,
            pixel_l_w,
            t,
            sum([len(x) for _, x in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.items()]),
        )
        sp_x = torch.randn(
            b, pixel_h_h, pixel_h_w, sum([len(x) for _, x in SPACE_BAND_GROUPS_IDX.items()])
        )
        t_x = torch.randn(b, t, sum([len(x) for _, x in TIME_BANDS_GROUPS_IDX.items()]))
        st_x = torch.randn(b, sum([len(x) for _, x in STATIC_BAND_GROUPS_IDX.items()]))
        s_t_h_m = (
            torch.ones((b, pixel_h_h, pixel_h_w, t, len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX))) * 2
        )
        s_t_m_m = (
            torch.ones((b, pixel_m_h, pixel_m_w, t, len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX))) * 2
        )
        s_t_l_m = (
            torch.ones((b, pixel_l_h, pixel_l_w, t, len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX))) * 2
        )
        sp_m = torch.ones((b, pixel_h_h, pixel_h_w, len(SPACE_BAND_GROUPS_IDX))) * 2
        t_m = torch.ones((b, t, len(TIME_BANDS_GROUPS_IDX))) * 2
        st_m = torch.ones((b, len(STATIC_BAND_GROUPS_IDX))) * 2
        max_patch_size = 8

        loss = mae_loss(
            p_s_t_h,
            p_s_t_m,
            p_s_t_l,
            p_sp,
            p_t,
            p_st,
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            patch_size_high_res,
            patch_size_med_res,
            patch_size_low_res,
            max_patch_size,
        )
        self.assertFalse(torch.isnan(loss))
