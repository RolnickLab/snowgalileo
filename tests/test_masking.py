import unittest

import numpy as np
import torch
from einops import repeat

from src.masking import (
    MASKING_MODES,
    MASKING_MULTIPLIER,
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
    return_masked_unmasked_bands,
)

MASK_TO_BANDS = {
    "S2_RGB": {
        "masked": return_masked_unmasked_bands(["S2_RGB"], SPACE_TIME_BANDS_GROUPS_IDX)[1],
        "unmasked": return_masked_unmasked_bands(["S2_RGB"], SPACE_TIME_BANDS_GROUPS_IDX)[0],
    },
    "S2": {
        "masked": return_masked_unmasked_bands(["S2"], SPACE_TIME_BANDS_GROUPS_IDX)[1],
        "unmasked": return_masked_unmasked_bands(["S2"], SPACE_TIME_BANDS_GROUPS_IDX)[0],
    },
    "S1": {
        "masked": return_masked_unmasked_bands(["S1"], SPACE_TIME_BANDS_GROUPS_IDX)[1],
        "unmasked": return_masked_unmasked_bands(["S1"], SPACE_TIME_BANDS_GROUPS_IDX)[0],
    },
    "S1+S2": {
        "masked": return_masked_unmasked_bands("S1+S2".split("+"), SPACE_TIME_BANDS_GROUPS_IDX)[1],
        "unmasked": return_masked_unmasked_bands("S1+S2".split("+"), SPACE_TIME_BANDS_GROUPS_IDX)[
            0
        ],
    },
}

DECODER_MASK_TO_BANDS = {
    "DW": return_masked_unmasked_bands(["DW"], SPACE_BAND_GROUPS_IDX)[0],
    "WC": return_masked_unmasked_bands(["WC"], SPACE_BAND_GROUPS_IDX)[0],
    "DW+WC": return_masked_unmasked_bands("DW+WC".split("+"), SPACE_BAND_GROUPS_IDX)[0],
}


class TestMasking(unittest.TestCase):
    def check_all_values_in_masks(self, space_time_mask, space_mask, time_mask, static_mask):
        self.assertTrue(
            (space_time_mask == 2).any()
            | (space_mask == 2).any()
            | (time_mask == 2).any()
            | (static_mask == 2).any()
        )
        self.assertTrue(
            (space_time_mask == 0).any()
            | (space_mask == 0).any()
            | (time_mask == 0).any()
            | (static_mask == 0).any()
        )
        self.assertTrue(
            (space_time_mask == 1).any()
            | (space_mask == 1).any()
            | (time_mask == 1).any()
            | (static_mask == 1).any()
        )

    def test_mask_by_time(self):
        for t in range(4, 8):
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
                        patch_size=4,
                    )
                    self.check_all_values_in_masks(
                        output.space_time_mask,
                        output.space_mask,
                        output.time_mask,
                        output.static_mask,
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

                    # the branching in the test below is a bit ugly and could be more concise,
                    # but I think it does test for all combinations
                    expected_unmasked_timesteps = max(int(t * mask_ratio), 1)
                    expected_decoder_timesteps = max(int(t * decoder_unmask_ratio), 1)
                    if (mode == "random") and (decoder_mode == "random"):
                        # collapse the dynamic_mask along the time dimension
                        space_time_mask_along_t = output.space_time_mask.float().mean(
                            axis=(1, 2, 4)
                        )  # b, t
                        time_mask_along_t = output.time_mask.float().mean(axis=-1)  # b, t
                        self.assertTrue(torch.equal(space_time_mask_along_t, time_mask_along_t))
                        self.assertTrue(np.isin(space_time_mask_along_t, (0, 1, 2)).all())
                        self.assertTrue(
                            (
                                (space_time_mask_along_t == 0).sum(axis=1)
                                == expected_unmasked_timesteps
                            ).all()
                        )
                        self.assertTrue(
                            (
                                (space_time_mask_along_t == 2).sum(axis=1)
                                == expected_decoder_timesteps
                            ).all()
                        )
                    else:
                        if mode != "random":
                            self.assertTrue(
                                0
                                in output.space_time_mask[
                                    :, :, :, :, MASK_TO_BANDS[mode]["unmasked"]
                                ]
                            )
                            self.assertFalse(
                                0
                                in output.space_time_mask[
                                    :, :, :, :, MASK_TO_BANDS[mode]["masked"]
                                ]
                            )
                        elif mode == "random":
                            # decoder mode is not random here
                            # collapse the dynamic_mask along the time dimension
                            space_time_mask_along_t = output.space_time_mask.float().mean(
                                axis=(1, 2, 4)
                            )  # b, t
                            time_mask_along_t = output.time_mask.float().mean(axis=-1)  # b, t
                            self.assertTrue(
                                torch.equal(space_time_mask_along_t, time_mask_along_t)
                            )
                            self.assertTrue(np.isin(space_time_mask_along_t, (0, 1, 2)).all())
                            self.assertTrue(
                                (
                                    (space_time_mask_along_t == 0).sum(axis=1)
                                    == expected_unmasked_timesteps
                                ).all()
                            )

                        if decoder_mode == "random":
                            # collapse the dynamic_mask along the time dimension,
                            # ignoring the values masked by the mode
                            space_time_mask_along_t = (
                                output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["masked"]]
                                .float()
                                .mean(axis=(1, 2, 4))
                            )  # b, t
                            self.assertTrue(
                                (
                                    (space_time_mask_along_t == 2).sum(axis=1)
                                    == expected_decoder_timesteps
                                ).all()
                            )
                            time_mask_along_t = output.time_mask.float().mean(axis=-1)  # b, t
                            self.assertTrue(
                                (
                                    (space_time_mask_along_t == 2).sum(axis=1)
                                    == expected_decoder_timesteps
                                ).all()
                            )

                        elif decoder_mode != "random":
                            self.assertTrue((output.time_mask <= 1).all())
                            self.assertTrue((output.static_mask <= 1).all())
                            self.assertTrue((output.static_mask == 1).all())
                            self.assertTrue(
                                (
                                    output.space_mask[:, :, :, DECODER_MASK_TO_BANDS[decoder_mode]]
                                    == 2
                                ).all()
                            )

    def test_mask_by_channel(self):
        b, t, h, w = 2, 8, 16, 16
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25
        decoder_mask_ratio = 0.25

        output = batch_mask_channels(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            mask_ratio,
            decoder_mask_ratio,
        )
        self.check_all_values_in_masks(
            output.space_time_mask, output.space_mask, output.time_mask, output.static_mask
        )
        self.assertEqual(
            (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
        )
        self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
        self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
        self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
        # collapse the space_time_mask along the time and space dimensions
        space_time_mask_along_c = output.space_time_mask.float().mean(axis=(1, 2, 3))  # b, c
        expected_num_channels_masked = int(len(SPACE_TIME_BANDS_GROUPS_IDX) * mask_ratio) * b
        self.assertTrue(
            (
                space_time_mask_along_c[space_time_mask_along_c == 1].sum()
                == expected_num_channels_masked
            ).all()
        )
        expected_num_channels_to_decode = (
            int(len(SPACE_TIME_BANDS_GROUPS_IDX) * decoder_mask_ratio) * b
        )
        self.assertTrue(
            (
                space_time_mask_along_c[space_time_mask_along_c == 2].sum()
                # hacky but the *2 lets us easily handle the fact
                # we are summing over values == 2, not 1
                == expected_num_channels_to_decode * 2
            ).all()
        )

    def test_mask_by_space(self):
        b, t, h, w, p = 2, 8, 16, 16, 4
        space_time_input = torch.ones((b, h, w, t, 8))
        space_input = torch.ones((b, h, w, 8))
        time_input = torch.ones((b, t, 8))
        static_input = torch.ones((b, 8))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25
        decoder_unmask_ratio = 0.25

        for mode in MASKING_MODES:
            for decoder_mode in UNMASKING_MODES:
                output = batch_mask_space(
                    space_time_input,
                    space_input,
                    time_input,
                    static_input,
                    months,
                    p,
                    mask_ratio,
                    decoder_unmask_ratio,
                    mode=mode,
                    decoder_mode=decoder_mode,
                )
                self.check_all_values_in_masks(
                    output.space_time_mask, output.space_mask, output.time_mask, output.static_mask
                )
                self.assertEqual(
                    (b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), output.space_time_mask.shape
                )
                self.assertEqual((b, h, w, len(SPACE_BAND_GROUPS_IDX)), output.space_mask.shape)
                self.assertEqual((b, t, len(TIME_BAND_GROUPS_IDX)), output.time_mask.shape)
                self.assertEqual((b, len(STATIC_BAND_GROUPS_IDX)), output.static_mask.shape)
                if (mode == "random") and (decoder_mode == "random"):
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
                elif (mode != "random") and (decoder_mode == "random"):
                    sp_along_hw = output.space_time_mask.float()[
                        :, :, :, :, MASK_TO_BANDS[mode]["unmasked"]
                    ].mean(axis=(3, 4))  # b, h, w
                    self.assertTrue((output.space_mask == 1).all())
                    self.assertTrue((output.time_mask == 1).all())
                    self.assertTrue(
                        (
                            output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["masked"]] == 1
                        ).all()
                    )
                    self.assertFalse(
                        (
                            output.space_time_mask[:, :, :, :, MASK_TO_BANDS[mode]["unmasked"]]
                            == 1
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
        decoder_unmask_ratio = 0.25

        output = batch_mask_random(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            mask_ratio,
            decoder_unmask_ratio,
            p,
        )
        self.check_all_values_in_masks(
            output.space_time_mask, output.space_mask, output.time_mask, output.static_mask
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
        space_time_per_token = output.space_time_mask[:, i::p, i::p]
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
            (h_tokens * w_tokens * t * len(SPACE_TIME_BANDS_GROUPS_IDX))
            + (h_tokens * w_tokens * len(SPACE_BAND_GROUPS_IDX))
            + (t * len(TIME_BAND_GROUPS_IDX))
        ) * b
        self.assertTrue(
            (
                space_time_masked_per_instance
                + space_masked_per_instance
                + time_masked_per_instance
                + static_masked_per_instance
                == total_tokens * mask_ratio
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
                == total_tokens * decoder_unmask_ratio * 2
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
        decoder_unmask_ratio = 0.25

        output = batch_subset_mask_presto_augmented(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            mask_ratio,
            decoder_unmask_ratio,
            p,
            image_size=i,
            num_timesteps=t_o,
            augmentation_strategies=None,
        )
        self.check_all_values_in_masks(
            output.space_time_mask, output.space_mask, output.time_mask, output.static_mask
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
