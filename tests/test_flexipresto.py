import unittest
from pathlib import Path

import torch

from src.data import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    Dataset,
)
from src.data.config import NUM_TIMESTEPS
from src.data.dataset import SPACE_BANDS, SPACE_TIME_BANDS, TIME_BANDS, DatasetOutput
from src.flexipresto import Encoder, PrestoPixelDecoder, PrestoRepresentationDecoder
from src.masking import batch_mask_presto, subset_batch_of_images

DATA_FOLDER = Path(__file__).parents[1] / "data/tifs"


class TestPresto(unittest.TestCase):
    @staticmethod
    def to_tensor_with_batch_d(input: DatasetOutput):
        return (
            torch.from_numpy(input.space_time_x).float().unsqueeze(0),
            torch.from_numpy(input.space_x).float().unsqueeze(0),
            torch.from_numpy(input.time_x).float().unsqueeze(0),
            torch.from_numpy(input.months).long().unsqueeze(0),
        )

    def test_end_to_end(self):
        self._end_to_end_run_ijepa(32, 16, 8)
        self._end_to_end_run_mae(16, 8)

    def test_end_to_end_different_inputs_per_dim_than_default(self):
        self._end_to_end_run_ijepa(32, 16, 4)
        self._end_to_end_run_mae(16, 4)

    def _end_to_end_run_ijepa(self, encoder_embedding_size, decoder_embedding_size, patch_size):
        image_size = patch_size * 4
        encoder = Encoder(embedding_size=encoder_embedding_size, num_heads=1)
        decoder = PrestoRepresentationDecoder(
            encoder_embedding_size=encoder_embedding_size,
            decoder_embedding_size=decoder_embedding_size,
            num_heads=1,
        )
        ds = Dataset(DATA_FOLDER, False)
        s_t_x, s_x, t_x, months = self.to_tensor_with_batch_d(ds[0])
        s_t_x, s_x = subset_batch_of_images(s_t_x, s_x, size=image_size)
        masked_output = batch_mask_presto(
            s_t_x,
            s_x,
            t_x,
            months,
            mask_ratio=0.5,
            patch_size=patch_size,
            time_ratio=0.25,
            space_ratio=0.25,
        )
        with torch.no_grad():
            # for now, we just make sure it all runs
            encoder_output = encoder(
                masked_output.space_time_x.float(),
                masked_output.space_x.float(),
                masked_output.time_x.float(),
                masked_output.space_time_mask.float(),
                masked_output.space_mask.float(),
                masked_output.time_mask.float(),
                masked_output.months.long(),
                patch_size=patch_size,
            )

        self.assertTrue(
            list(encoder_output[0].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                NUM_TIMESTEPS,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[1].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                len(SPACE_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[2].shape)
            == [
                1,
                NUM_TIMESTEPS,
                len(TIME_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(encoder_output[0]).any())
        self.assertFalse(torch.isnan(encoder_output[1]).any())
        self.assertFalse(torch.isnan(encoder_output[2]).any())

        with torch.no_grad():
            output = decoder(*encoder_output)
        self.assertTrue(
            list(output[0].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                NUM_TIMESTEPS,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
                # decoder outputs should be mapped back to the encoder embedding size
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(output[1].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                len(SPACE_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertTrue(
            list(output[2].shape)
            == [
                1,
                NUM_TIMESTEPS,
                len(TIME_BAND_GROUPS_IDX),
                encoder_embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())
        self.assertFalse(torch.isnan(output[2]).any())

    def _end_to_end_run_mae(self, embedding_size, patch_size):
        image_size = patch_size * 4
        encoder = Encoder(embedding_size=embedding_size, num_heads=1)
        decoder = PrestoPixelDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        max_patch_size = decoder.max_patch_size
        ds = Dataset(DATA_FOLDER, False)
        s_t_x, s_x, t_x, months = self.to_tensor_with_batch_d(ds[0])
        s_t_x, s_x = subset_batch_of_images(s_t_x, s_x, size=image_size)
        masked_output = batch_mask_presto(
            s_t_x,
            s_x,
            t_x,
            months,
            mask_ratio=0.5,
            patch_size=patch_size,
            time_ratio=0.25,
            space_ratio=0.25,
        )
        with torch.no_grad():
            # for now, we just make sure it all runs
            encoder_output = encoder(
                masked_output.space_time_x.float(),
                masked_output.space_x.float(),
                masked_output.time_x.float(),
                masked_output.space_time_mask.float(),
                masked_output.space_mask.float(),
                masked_output.time_mask.float(),
                masked_output.months.long(),
                patch_size=patch_size,
            )

        self.assertTrue(
            list(encoder_output[0].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                NUM_TIMESTEPS,
                len(SPACE_TIME_BANDS_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[1].shape)
            == [
                1,
                image_size / patch_size,
                image_size / patch_size,
                len(SPACE_BAND_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[2].shape)
            == [
                1,
                NUM_TIMESTEPS,
                len(TIME_BAND_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(encoder_output[0]).any())
        self.assertFalse(torch.isnan(encoder_output[1]).any())
        self.assertFalse(torch.isnan(encoder_output[2]).any())

        with torch.no_grad():
            output = decoder(*encoder_output)
        self.assertTrue(
            list(output[0].shape)
            == [
                1,
                image_size * (max_patch_size / patch_size),
                image_size * (max_patch_size / patch_size),
                NUM_TIMESTEPS,
                len(SPACE_TIME_BANDS),
            ]
        )
        self.assertTrue(
            list(output[1].shape)
            == [
                1,
                image_size * (max_patch_size / patch_size),
                image_size * (max_patch_size / patch_size),
                len(SPACE_BANDS),
            ]
        )
        self.assertTrue(list(output[2].shape) == [1, NUM_TIMESTEPS, len(TIME_BANDS)])
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())
        self.assertFalse(torch.isnan(output[2]).any())

    def test_presto_representation_decoder_add_masks(self):
        enc_embedding_size = 16
        dec_embedding_size = 8
        decoder = PrestoRepresentationDecoder(
            encoder_embedding_size=enc_embedding_size,
            decoder_embedding_size=dec_embedding_size,
            num_heads=1,
        )
        b, h, w, t = 5, 6, 7, 8
        s_t_x = torch.ones(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX), dec_embedding_size)
        s_t_m = torch.zeros(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX))
        s_t_m[:, :, :, 0] = 1  # mask the first timestep

        s_x = torch.ones(b, h, w, len(SPACE_BAND_GROUPS_IDX), dec_embedding_size)
        s_m = torch.zeros(b, h, w, len(SPACE_BAND_GROUPS_IDX))
        s_m[:, 0] = 1

        t_x = torch.ones(b, t, len(TIME_BAND_GROUPS_IDX), dec_embedding_size)
        t_m = torch.zeros(b, t, len(TIME_BAND_GROUPS_IDX))
        t_m[:, 0] = 1
        with torch.no_grad():
            o = decoder.add_masks(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        for i in range(3):
            self.assertTrue((o[i + 3] == 0).all())

        self.assertTrue((o[0][:, :, :, 0] == 0).all())
        self.assertTrue((o[0][:, :, :, 1:] == 1).all())
        self.assertTrue((o[1][:, 0] == 0).all())
        self.assertTrue((o[1][:, 1:] == 1).all())
        self.assertTrue((o[2][:, 0] == 0).all())
        self.assertTrue((o[2][:, 1:] == 1).all())

    def test_presto_pixel_decoder_add_masks(self):
        embedding_size = 16
        decoder = PrestoPixelDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        b, h, w, t = 5, 6, 7, 8
        s_t_x = torch.ones(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX), embedding_size)
        s_t_m = torch.zeros(b, h, w, t, len(SPACE_TIME_BANDS_GROUPS_IDX))
        s_t_m[:, :, :, 0] = 1  # mask the first timestep

        s_x = torch.ones(b, h, w, len(SPACE_BAND_GROUPS_IDX), embedding_size)
        s_m = torch.zeros(b, h, w, len(SPACE_BAND_GROUPS_IDX))
        s_m[:, 0] = 1

        t_x = torch.ones(b, t, len(TIME_BAND_GROUPS_IDX), embedding_size)
        t_m = torch.zeros(b, t, len(TIME_BAND_GROUPS_IDX))
        t_m[:, 0] = 1
        with torch.no_grad():
            o = decoder.add_masks(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        for i in range(3):
            self.assertTrue((o[i + 3] == 0).all())

        self.assertTrue((o[0][:, :, :, 0] == 0).all())
        self.assertTrue((o[0][:, :, :, 1:] == 1).all())
        self.assertTrue((o[1][:, 0] == 0).all())
        self.assertTrue((o[1][:, 1:] == 1).all())
        self.assertTrue((o[2][:, 0] == 0).all())
        self.assertTrue((o[2][:, 1:] == 1).all())

    def test_mean_of_tokens(self):
        b, t, d, h, w, s_t_c_g, s_c_g, t_c_g = 1, 2, 8, 3, 3, 5, 6, 2
        s_t_x = torch.ones((b, h, w, t, s_t_c_g, d))
        s_x = torch.ones((b, h, w, s_c_g, d))
        t_x = torch.ones((b, t, t_c_g, d))

        # the first timestep and the first column are masked
        s_t_m = torch.zeros((b, h, w, t, s_t_c_g))
        s_t_m[:, :, 0, :] = 1
        s_t_m[:, :, :, 0] = 1
        s_t_x[:, :, 0, :] = 0
        s_t_x[:, :, :, 0] = 0
        # the last row is masked
        s_m = torch.zeros((b, h, w, s_c_g))
        s_m[:, -1, :] = 1
        s_x[:, -1, :] = 0
        # the first timestep is masked
        t_m = torch.zeros((b, t, t_c_g))
        t_m[:, 0] = 1
        t_x[:, 0] = 0

        mean = Encoder.average_tokens(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        self.assertEqual(mean.shape, (b, d))
        self.assertTrue((mean == 1).all())
