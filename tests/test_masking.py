import unittest

import numpy as np
import torch
from einops import repeat

from src.masking import (
    MASKING_MODES,
    MASKING_MULTIPLIER,
    NON_S1_BANDS,
    NON_S1_S2_BANDS,
    NON_S2_BANDS,
    NON_S2_RGB_BANDS,
    S1_BANDS,
    S1_S2_BANDS,
    S2_BANDS,
    S2_RGB_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    UNMASKING_MODES,
    batch_mask_channels,
    batch_mask_random,
    batch_mask_space,
    batch_mask_time,
    batch_subset_mask_presto_augmented,
)

MASK_TO_BANDS = {
    "s2rgb": {"masked": NON_S2_RGB_BANDS, "unmasked": S2_RGB_BANDS},
    "s2": {"masked": NON_S2_BANDS, "unmasked": S2_BANDS},
    "s1": {"masked": NON_S1_BANDS, "unmasked": S1_BANDS},
    "s1+s2": {"masked": NON_S1_S2_BANDS, "unmasked": S1_S2_BANDS},
}


class TestMasking(unittest.TestCase):
    def test_mask_by_time(self):
        for t in [1, 8]:
            b, h, w = 2, 16, 16
            space_time_input = torch.ones((b, h, w, t, 8))
            space_input = torch.ones((b, h, w, 8))
            time_input = torch.ones((b, t, 8))
            static_input = torch.ones((b, 8))
            months = repeat(torch.arange(0, t), "t -> b t", b=b)
            mask_ratio = 0.25
            decoder_unmask_ratio = 0.25

            for mode in MASKING_MODES:
                for decoder_mode in UNMASKING_MODES:
                    output = batch_mask_time(
                        space_time_input,
                        space_input,
                        time_input,
                        static_input,
                        months,
                        mask_ratio=mask_ratio,
                        decoder_unmask_ratio=decoder_unmask_ratio,
                        mode=mode,
                        decoder_mode=decoder_mode,
                    )
                    self.assertEqual(
                        (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)),
                        output.space_time_mask.shape,
                    )
                    self.assertEqual(
                        (b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape
                    )
                    self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
                    self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
                    if mode is None:
                        # collapse the dynamic_mask along the time dimension
                        space_time_mask_along_t = output.space_time_mask.float().mean(
                            axis=(1, 2, 4)
                        )  # b, t
                        time_mask_along_t = output.time_mask.float().mean(axis=-1)  # b, t
                        self.assertTrue(torch.equal(space_time_mask_along_t, time_mask_along_t))
                        self.assertTrue(np.isin(space_time_mask_along_t, (0, 1)).all())
                        self.assertTrue(
                            (
                                space_time_mask_along_t.sum(axis=1)
                                / space_time_mask_along_t.shape[1]
                                == (mask_ratio if t > 1 else 1)
                            ).all()
                        )
                    else:
                        self.assertTrue(
                            (
                                output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["masked"]]
                                == 1
                            ).all()
                        )
                        space_time_unmasked = (
                            output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["unmasked"]]
                            .float()
                            .mean(axis=(1, 2, 4))
                        )
                        self.assertTrue(
                            (
                                space_time_unmasked.sum(axis=1) / space_time_unmasked.shape[1]
                                == (mask_ratio if t > 1 else 0)
                            ).all()
                        )
                        self.assertTrue(np.isin(space_time_unmasked, (0, 1)).all())
                        self.assertTrue((output.space_mask == 1).all())
                        self.assertTrue((output.time_mask == 1).all())
                        self.assertTrue((output.static_mask == 1).all())

    def test_mask_by_channel(self):
        b, t, h, w = 2, 8, 16, 16
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_mask_channels(
            space_time_input, space_input, time_input, static_input, months, mask_ratio
        )
        self.assertEqual(
            (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
        )
        self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
        self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
        self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
        # collapse the space_time_mask along the time and space dimensions
        space_time_mask_along_c = output.space_time_mask.float().mean(axis=(1, 2, 3))  # b, c
        expected_num_channels_masked = int(len(SPACE_TIME_BANDS_GROUPS_IDX) * mask_ratio)
        self.assertTrue(
            (space_time_mask_along_c.sum(axis=1) == expected_num_channels_masked).all()
        )

    def test_mask_by_space(self):
        b, t, h, w, p = 2, 8, 16, 16, 4
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        for mode in MASKING_MODES:
            output = batch_mask_space(
                space_time_input,
                space_input,
                time_input,
                static_input,
                months,
                mask_ratio,
                p,
                mode=mode,
            )
            self.assertEqual(
                (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
            )
            self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
            self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
            self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
            if mode is None:
                sp_along_hw = output.space_time_mask.float().mean(axis=(3, 4))  # b, h, w
                s_along_hw = output.space_mask.float().mean(axis=(3))  # b, h, w
                self.assertTrue(torch.equal(sp_along_hw, s_along_hw))
                self.assertTrue(
                    (sp_along_hw.sum(axis=1).sum(axis=1) / (h * w) == mask_ratio).all()
                )
                for i in range(1, p):
                    self.assertTrue(
                        torch.equal(
                            s_along_hw[:, i::p, i::p], s_along_hw[:, i - 1 :: p, i - 1 :: p]
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            sp_along_hw[:, i::p, i::p], sp_along_hw[:, i - 1 :: p, i - 1 :: p]
                        )
                    )
            else:
                sp_along_hw = output.space_time_mask.float()[
                    :, :, :, :, MASK_TO_BANDS[mode]["unmasked"]
                ].mean(axis=(3, 4))  # b, h, w
                self.assertTrue((output.space_mask == 1).all())
                self.assertTrue((output.time_mask == 1).all())
                self.assertTrue(
                    (output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["masked"]] == 1).all()
                )
                self.assertFalse(
                    (
                        output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["unmasked"]] == 1
                    ).all()
                )
                self.assertTrue((output.space_mask == 1).all())
                self.assertTrue((output.time_mask == 1).all())
                self.assertTrue((output.static_mask == 1).all())
                for i in range(1, p):
                    self.assertTrue(
                        torch.equal(
                            sp_along_hw[:, i::p, i::p], sp_along_hw[:, i - 1 :: p, i - 1 :: p]
                        )
                    )

    def test_mask_by_random(self):
        b, t, h, w, p = 2, 8, 16, 16, 4
        h_tokens, w_tokens = h / p, w / p
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_mask_random(
            space_time_input, space_input, time_input, static_input, months, mask_ratio, p
        )
        self.assertEqual(
            (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
        )
        self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
        self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
        self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)

        for i in range(1, p):
            self.assertTrue(
                torch.equal(
                    output.space_time_mask[:, i::p, i::p],
                    output.space_time_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )
            self.assertTrue(
                torch.equal(
                    output.space_mask[:, i::p, i::p],
                    output.space_mask[:, i - 1 :: p, i - 1 :: p],
                )
            )

        space_time_masked_per_instance = output.space_time_mask[:, i::p, i::p].sum(
            dim=(1, 2, 3, 4)
        )
        space_masked_per_instance = output.space_mask[:, i::p, i::p].sum(dim=(1, 2, 3))
        time_masked_per_instance = output.time_mask.sum(dim=(1, 2))
        static_masked_per_instance = output.static_mask.sum(dim=1)
        total_tokens = (
            (h_tokens * w_tokens * t * len(SPACE_TIME_BANDS_GROUPS_IDX))
            + (h_tokens * w_tokens * len(SPACE_BAND_GROUPS_IDX))
            + (t * len(TIME_BAND_GROUPS_IDX))
        )
        self.assertTrue(
            (
                space_time_masked_per_instance
                + space_masked_per_instance
                + time_masked_per_instance
                + static_masked_per_instance
                == total_tokens * mask_ratio
            ).all()
        )

    def test_mask_combined(self):
        b, t, h, w, p = 4, 8, 16, 16, 4
        i, t_o = 8, 4
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25

        output = batch_subset_mask_presto_augmented(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            mask_ratio,
            p,
            image_size=i,
            num_timesteps=t_o,
        )
        self.assertEqual(
            (b * MASKING_MULTIPLIER, i, i, t_o, len(SPACE_TIME_BANDS_GROUPS_IDX)),
            output.space_time_mask.shape,
        )
        self.assertEqual(
            (b * MASKING_MULTIPLIER, i, i, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape
        )
        self.assertEqual(
            (b * MASKING_MULTIPLIER, t_o, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape
        )
        self.assertEqual(
            (b * MASKING_MULTIPLIER, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape
        )
