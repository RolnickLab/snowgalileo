import unittest

import torch
from einops import repeat

from src.conditioner import LearnedMixture
from src.data.dataset import SPACE_BANDS, SPACE_TIME_BANDS, STATIC_BANDS, TIME_BANDS
from src.flexipresto import Encoder, PrestoPixelDecoder
from src.masking import MASKING_MODES_COARSE, batch_mask_random


class TestConditioner(unittest.TestCase):
    def test_end_to_end_grads_nonzero(self):
        conditioner_dict = {"num_output_channels": len(MASKING_MODES_COARSE)}
        mixer = LearnedMixture(**conditioner_dict)
        encoder = Encoder(conditioner=mixer)
        decoder = PrestoPixelDecoder()

        conditioner_inputs = {"output_channels": torch.zeros(len(MASKING_MODES_COARSE)).float()}
        conditioner_inputs["output_channels"][0] = 1

        masked_output, patch_size = self.construct_inputs()
        encoder_output = encoder(
            masked_output.space_time_x,
            masked_output.space_x,
            masked_output.time_x,
            masked_output.static_x,
            masked_output.space_time_mask,
            masked_output.space_mask,
            masked_output.time_mask,
            masked_output.static_mask,
            masked_output.months.long(),
            patch_size=patch_size,
            c_i=conditioner_inputs,
        )
        decoder_output = decoder(*encoder_output)
        sum([d.sum() for d in decoder_output]).backward()

        for t_i, t in enumerate(encoder.conditioner.e_templates[0]):
            for n, p in t.named_parameters():
                if ("bias" not in n) and ("norm" not in n):
                    self.assertTrue(
                        p.grad is not None, f"{t_i}, {n} has an unexpectedly None grad"
                    )
                else:
                    self.assertTrue(
                        p.grad is None, f"{t_i}, {n} has an unexpectedly not None grad"
                    )
        for t_i, t in enumerate(encoder.conditioner.e_templates[1:]):
            self.assertTrue(p.grad is None, f"{t_i}, {n} has an unexpectedly not None grad")

        # next, test with c_i = None
        encoder.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)
        # check zero grad worked
        for p in encoder.conditioner.parameters():
            self.assertTrue(p.grad is None)

        encoder_output = encoder(
            masked_output.space_time_x,
            masked_output.space_x,
            masked_output.time_x,
            masked_output.static_x,
            masked_output.space_time_mask,
            masked_output.space_mask,
            masked_output.time_mask,
            masked_output.static_mask,
            masked_output.months.long(),
            patch_size=patch_size,
            c_i=None,
        )
        decoder_output = decoder(*encoder_output)
        sum([d.sum() for d in decoder_output]).backward()

        for p in encoder.conditioner.parameters():
            self.assertTrue(p.grad is None)

    def construct_inputs(self, b=2, t=8, h=16, w=16, p=4):
        space_time_input = torch.ones((b, h, w, t, len(SPACE_TIME_BANDS)))
        space_input = torch.ones((b, h, w, len(SPACE_BANDS)))
        time_input = torch.ones((b, t, len(TIME_BANDS)))
        static_input = torch.ones((b, len(STATIC_BANDS)))
        months = repeat(torch.arange(0, t), "t -> b t", b=b)
        mask_ratio = 0.25
        decoder_unmask_ratio = 0.25

        return batch_mask_random(
            space_time_input,
            space_input,
            time_input,
            static_input,
            months,
            mask_ratio,
            decoder_unmask_ratio,
            p,
        ), p
