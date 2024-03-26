import unittest
from pathlib import Path

import torch

from src.config import NUM_TIMESTEPS, PRESTO_INPUT_SIZE
from src.data import DYNAMIC_BANDS_GROUPS_IDX
from src.flexipresto import Encoder, PrestoPixelDecoder, PrestoRepresentationDecoder
from src.masked_datasets import (
    STATIC_BAND_GROUPS_IDX,
    MaskedOutput,
    PrestoToPrestoMaskedDataset,
)

DATA_FOLDER = Path(__file__).parents[1] / "data/tifs"


class TestPresto(unittest.TestCase):
    @staticmethod
    def to_tensor_with_batch_d(input: MaskedOutput):
        return (
            torch.from_numpy(input.dynamic_x).float().unsqueeze(0),
            torch.from_numpy(input.static_x).float().unsqueeze(0),
            torch.from_numpy(input.dynamic_mask).float().unsqueeze(0),
            torch.from_numpy(input.static_mask).float().unsqueeze(0),
            torch.from_numpy(input.months).long().unsqueeze(0),
        )

    def test_end_to_end(self):
        self._end_to_end_run_ijepa(16, 8, 8)
        self._end_to_end_run_mae(16, 8)

    def test_end_to_end_different_inputs_per_dim_than_default(self):
        self._end_to_end_run_ijepa(16, 8, 4)
        self._end_to_end_run_mae(16, 4)

    def _end_to_end_run_ijepa(self, encoder_embedding_size, decoder_embedding_size, patch_size):
        encoder = Encoder(encoder_embedding_size=encoder_embedding_size, num_heads=1)
        decoder = PrestoRepresentationDecoder(
            encoder_embedding_size=encoder_embedding_size,
            decoder_embedding_size=decoder_embedding_size,
            num_heads=1,
        )
        ds = PrestoToPrestoMaskedDataset(DATA_FOLDER, 0.25, False)
        output = ds[0]
        with torch.no_grad():
            # for now, we just make sure it all runs
            encoder_output = encoder(*self.to_tensor_with_batch_d(output), patch_size=patch_size)

        self.assertTrue(
            list(encoder_output[0].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                NUM_TIMESTEPS,
                len(DYNAMIC_BANDS_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                len(STATIC_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(encoder_output[0]).any())
        self.assertFalse(torch.isnan(encoder_output[1]).any())

        with torch.no_grad():
            output = decoder(*encoder_output)
        self.assertTrue(
            list(output[0].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                NUM_TIMESTEPS,
                len(DYNAMIC_BANDS_GROUPS_IDX),
                # decoder outputs should be mapped back to the encoder embedding size
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                len(STATIC_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())

    def _end_to_end_run_mae(self, embedding_size, patch_size):
        encoder = Encoder(embedding_size=embedding_size, num_heads=1)
        decoder = PrestoPixelDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        ds = PrestoToPrestoMaskedDataset(DATA_FOLDER, 0.25, False)
        output = ds[0]
        with torch.no_grad():
            # for now, we just make sure it all runs
            encoder_output = encoder(*self.to_tensor_with_batch_d(output), patch_size=patch_size)

        self.assertTrue(
            list(encoder_output[0].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                NUM_TIMESTEPS,
                len(DYNAMIC_BANDS_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE / patch_size,
                PRESTO_INPUT_SIZE / patch_size,
                len(STATIC_BAND_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(encoder_output[0]).any())
        self.assertFalse(torch.isnan(encoder_output[1]).any())

        with torch.no_grad():
            output = decoder(*encoder_output)
        self.assertTrue(
            list(output[0].shape)
            == [
                1,
                PRESTO_INPUT_SIZE,
                PRESTO_INPUT_SIZE,
                NUM_TIMESTEPS,
                len(DYNAMIC_BANDS_GROUPS_IDX),
            ]
        )
        self.assertTrue(
            list(output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE,
                PRESTO_INPUT_SIZE,
                len(STATIC_BAND_GROUPS_IDX),
            ]
        )
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())

    def test_presto_representation_decoder_add_masks(self):
        enc_embedding_size = 16
        dec_embedding_size = 8
        decoder = PrestoRepresentationDecoder(
            encoder_embedding_size=enc_embedding_size,
            decoder_embedding_size=dec_embedding_size,
            num_heads=1,
        )
        b, h, w, t = 5, 6, 7, 8
        d_x = torch.ones(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX), dec_embedding_size)
        d_m = torch.zeros(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX))
        d_m[:, :, :, 0] = 1  # mask the first timestep
        with torch.no_grad():
            o, o_m = decoder.add_masks(d_x, d_m)
        self.assertTrue((o_m == 0).all())
        self.assertTrue((o[:, :, :, 0] == 0).all())
        self.assertTrue((o[:, :, :, 1:] == 1).all())

    def test_presto_pixel_decoder_add_masks(self):
        embedding_size = 16
        decoder = PrestoPixelDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        b, h, w, t = 5, 6, 7, 8
        d_x = torch.ones(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX), embedding_size)
        d_m = torch.zeros(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX))
        d_m[:, :, :, 0] = 1  # mask the first timestep
        with torch.no_grad():
            o, o_m = decoder.add_masks(d_x, d_m)
        self.assertTrue((o_m == 0).all())
        self.assertTrue((o[:, :, :, 0] == 0).all())
        self.assertTrue((o[:, :, :, 1:] == 1).all())

    def test_mean_of_tokens(self):
        b, t, d, h, w, d_c_g, s_c_g = 1, 2, 8, 3, 3, 5, 6
        d_x = torch.ones((b, h, w, t, d_c_g, d))
        s_x = torch.ones((b, h, w, s_c_g, d))

        # the first timestep and the first column are masked
        d_m = torch.zeros((b, h, w, t, d_c_g))
        d_m[:, :, 0, :] = 1
        d_m[:, :, :, 0] = 1
        # the last row is masked
        s_m = torch.zeros((b, h, w, s_c_g))
        s_m[:, -1, :] = 1

        d_x[:, :, 0, :] = 0
        d_x[:, :, :, 0] = 0
        s_x[:, -1, :] = 0

        mean = Encoder.average_tokens(d_x, s_x, d_m, s_m)
        self.assertEqual(mean.shape, (b, d))
        self.assertTrue((mean == 1).all())
