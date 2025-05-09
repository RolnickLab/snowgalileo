import random
import unittest

import torch
from einops import repeat

from src.masking import (
    MASKING_MODES,
    MAX_MASKING_STRATEGIES,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
    batch_mask_random,
    batch_mask_space,
    batch_mask_time,
    check_modes_for_conflicts,
    weighted_sample_without_replacement,
)


class TestMasking(unittest.TestCase):
    def check_all_values_in_masks(
        self,
        space_time_high_mask,
        space_mask,
        time_mask,
        static_mask,
        masking_modes,
        unmasking_modes,
    ):
        self.assertTrue(
            (space_time_high_mask == 2).any()
            | (space_mask == 2).any()
            | (time_mask == 2).any()
            | (static_mask == 2).any(),
            f"2 check failed for {masking_modes}, {unmasking_modes}",
        )
        self.assertTrue(
            (space_time_high_mask == 0).any()
            | (space_mask == 0).any()
            | (time_mask == 0).any()
            | (static_mask == 0).any(),
            f"0 check failed for {masking_modes}, {unmasking_modes}",
        )
        self.assertTrue(
            (space_time_high_mask == 1).any()
            | (space_mask == 1).any()
            | (time_mask == 1).any()
            | (static_mask == 1).any(),
            f"1 check failed for {masking_modes}, {unmasking_modes}",
        )

    def test_mask_by_time(self):
        # testing specific failure modes
        self._test_mask_by_for_f(
            batch_mask_time,
            [
                ("static", "LS"),
                ("static", "location"),
                ("space", "WC"),
                ("space_time", "S2_SWIR"),
                ("space", "DW"),
                ("space_time", "S2_NIR_20m"),
            ],
            [("time", "TC"), ("time", "VIIRS")],
        )

        for _ in range(100):
            num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
            num_unmasking_modes = 1

            masking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_masking_modes
            )
            unmasking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            masking_modes, unmasking_modes = check_modes_for_conflicts(
                masking_modes, unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            self.assertTrue(len(masking_modes) >= 1, f"Got {len(masking_modes)}")
            for m_m in masking_modes:
                self.assertTrue(m_m not in unmasking_modes, f"{m_m} in {unmasking_modes}")
            for u_m in unmasking_modes:
                self.assertTrue(u_m not in masking_modes, f"{u_m} in {masking_modes}")
            self.assertTrue(len(masking_modes) >= 1)
            self.assertTrue(len(unmasking_modes) >= 1)
            self._test_mask_by_for_f(batch_mask_space, masking_modes, unmasking_modes)

    def test_mask_by_space(self):
        for _ in range(100):
            num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
            num_unmasking_modes = 1

            masking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_masking_modes
            )
            unmasking_modes = weighted_sample_without_replacement(
                MASKING_MODES, weights=[1] * len(MASKING_MODES), k=num_unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            masking_modes, unmasking_modes = check_modes_for_conflicts(
                masking_modes, unmasking_modes
            )
            self.assertTrue(
                len(unmasking_modes) == num_unmasking_modes, f"Got {len(unmasking_modes)}"
            )
            self.assertTrue(len(masking_modes) >= 1, f"Got {len(masking_modes)}")
            for m_m in masking_modes:
                self.assertTrue(m_m not in unmasking_modes, f"{m_m} in {unmasking_modes}")
            for u_m in unmasking_modes:
                self.assertTrue(u_m not in masking_modes, f"{u_m} in {masking_modes}")
            self.assertTrue(len(masking_modes) >= 1)
            self.assertTrue(len(unmasking_modes) >= 1)
            self._test_mask_by_for_f(batch_mask_space, masking_modes, unmasking_modes)

    def _test_mask_by_for_f(self, f, masking_modes, unmasking_modes):
        for t in range(4, 8):
            b, h, w = 2, 16, 16
            space_time_input = torch.ones((b, h, w, t, 8))
            space_input = torch.ones((b, h, w, 8))
            time_input = torch.ones((b, t, 8))
            static_input = torch.ones((b, 8))
            months = repeat(torch.arange(0, t), "t -> b t", b=b)
            valid_data_mask_s_t_h = torch.ones_like(space_time_input)
            valid_data_mask_sp = torch.ones_like(space_input)
            valid_data_mask_t = torch.ones_like(time_input)
            valid_data_mask_st = torch.ones_like(static_input)
            ratio = 0.25
            output = f(
                space_time_input,
                space_input,
                time_input,
                static_input,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
                encode_ratio=ratio,
                decode_ratio=ratio,
                mode=masking_modes,
                decoder_mode=unmasking_modes,
                patch_size=4,
            )
            self.check_all_values_in_masks(
                output.space_time_high_mask,
                output.space_mask,
                output.time_mask,
                output.static_mask,
                masking_modes,
                unmasking_modes,
            )
            self.assertEqual(
                (b, h, w, t, len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)),
                output.space_time_high_mask.shape,
            )
            self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
            self.assertEqual((b, t, len(TIME_BANDS_GROUPS_IDX)), output.time_mask.shape)
            self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)

    def test_mask_by_random(self):
        b, t, h, w, p = 2, 8, 16, 16, 4
        h_tokens, w_tokens = h / p, w / p
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        valid_data_mask_s_t_h = torch.ones_like(space_time_input)  # (b, h, w, t, c)
        valid_data_mask_sp = torch.ones_like(space_input)  # (b, h, w, c)
        valid_data_mask_t = torch.ones_like(time_input)  # (b, t, c)
        valid_data_mask_st = torch.ones_like(static_input)  # (b, c)
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        ratio = 0.25

        output = batch_mask_random(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            valid_data_mask_s_t_h,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
            ratio,
            ratio,
            p,
        )
        self.check_all_values_in_masks(
            output.space_time_high_mask,
            output.space_mask,
            output.time_mask,
            output.static_mask,
            "random",
            "random",
        )
        self.assertEqual(
            (b, h, w, t, len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)),
            output.space_time_high_mask.shape,
        )
        self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
        self.assertEqual((b, t, len(TIME_BANDS_GROUPS_IDX)), output.time_mask.shape)
        self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)

        for i in range(1, p):
            self.assertTrue(
                torch.equal(
                    output.space_time_high_mask[:, i::p, i::p],
                    output.space_time_high_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )
            self.assertTrue(
                torch.equal(
                    output.space_mask[:, i::p, i::p],
                    output.space_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )
        space_time_per_token = output.space_time_high_mask[:, i::p, i::p]
        space_time_masked_per_instance = space_time_per_token[space_time_per_token == 1].sum()
        space_time_decode_per_instance = space_time_per_token[space_time_per_token == 2].sum()
        space_per_token = output.space_mask[:, i::p, i::p]
        space_masked_per_instance = space_per_token[space_per_token == 1].sum()
        space_decode_per_instance = space_per_token[space_per_token == 2].sum()
        time_per_token = output.time_mask
        time_masked_per_instance = time_per_token[time_per_token == 1].sum()
        time_decode_per_instance = time_per_token[time_per_token == 2].sum()
        static_per_token = output.static_mask
        static_masked_per_instance = static_per_token[static_per_token == 1].sum()
        static_decode_per_instance = static_per_token[static_per_token == 2].sum()
        total_tokens = (
            (h_tokens * w_tokens * t * len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX))
            + (h_tokens * w_tokens * len(SPACE_BAND_GROUPS_IDX))
            + (t * len(TIME_BANDS_GROUPS_IDX))
            + len(STATIC_BAND_GROUPS_IDX)
        ) * b
        self.assertTrue(
            (
                space_time_masked_per_instance
                + space_masked_per_instance
                + time_masked_per_instance
                + static_masked_per_instance
                == total_tokens * (1 - (ratio * 2))
            ).all()
        )
        self.assertTrue(
            (
                space_time_decode_per_instance
                + space_decode_per_instance
                + time_decode_per_instance
                + static_decode_per_instance
                # hacky but the *2 lets us easily handle the fact
                # we are summing over values == 2, not 1
                == total_tokens * ratio * 2
            ).all()
        )
