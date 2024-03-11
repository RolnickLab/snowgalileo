import unittest
from pathlib import Path

import torch

from src.config import NUM_TIMESTEPS, PRESTO_INPUT_SIZE
from src.data import DYNAMIC_BANDS_GROUPS_IDX
from src.data.config import DATA_FOLDER
from src.masked_datasets import (
    STATIC_BAND_GROUPS_IDX,
    MaskedOutput,
    PrestoToPrestoMaskedDataset,
)
from src.presto import Encoder, PrestoAttn, PrestoDecoder

TEST_FILE = (
    DATA_FOLDER / "test_files" / "presto_tif" / "tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2022-01-01_2023-12-31.tif"
)


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

    def test_presto_end_to_end(self):
        embedding_size = 8
        encoder = Encoder(embedding_size=embedding_size, num_heads=1)
        decoder = PrestoDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        )
        ds = PrestoToPrestoMaskedDataset(DATA_FOLDER, 0.25, False)
        output = ds[0]
        with torch.no_grad():
            # for now, we just make sure it all runs
            encoder_output = encoder(*self.to_tensor_with_batch_d(output))

        self.assertTrue(
            list(encoder_output[0].shape)
            == [
                1,
                PRESTO_INPUT_SIZE,
                PRESTO_INPUT_SIZE,
                NUM_TIMESTEPS,
                len(DYNAMIC_BANDS_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertTrue(
            list(encoder_output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE,
                PRESTO_INPUT_SIZE,
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
                embedding_size,
            ]
        )
        self.assertTrue(
            list(output[1].shape)
            == [
                1,
                PRESTO_INPUT_SIZE,
                PRESTO_INPUT_SIZE,
                len(STATIC_BAND_GROUPS_IDX),
                embedding_size,
            ]
        )
        self.assertFalse(torch.isnan(output[0]).any())
        self.assertFalse(torch.isnan(output[1]).any())

    def test_presto_decoder_add_masks(self):
        embedding_size = 8
        decoder = PrestoDecoder(
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

    def test_presto_time_channel_encodings(self):
        b, h, w, t, d, c_r, m_r = 1, 2, 3, 4, 8, 0.25, 0.25
        model = PrestoAttn(
            embedding_size=d, num_heads=1, channel_embed_ratio=c_r, month_embed_ratio=m_r
        )
        months = torch.arange(0, t).unsqueeze(0)
        with torch.no_grad():
            d_encodings, s_encodings = model.construct_temporal_channel_embeddings(
                b=b, h=h, w=w, months=months
            )
        self.assertEqual(list(d_encodings.shape), [b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX), d])
        self.assertEqual(list(s_encodings.shape), [b, h, w, len(STATIC_BAND_GROUPS_IDX), d])

        for i in range(len(DYNAMIC_BANDS_GROUPS_IDX)):
            channel_dims = int(d * c_r)
            channel_encodings = d_encodings[:, :, :, :, i, :channel_dims]  # [b, h, w, t, d]
            channel_mean = channel_encodings.mean(dim=[0, 1, 2, 3])
            self.assertTrue(torch.equal(channel_encodings[0, 0, 0, 0], channel_mean))

        for i in range(len(STATIC_BAND_GROUPS_IDX)):
            channel_dims = int(d * c_r)
            channel_encodings = s_encodings[:, :, :, i, :channel_dims]  # [b, h, w, d]
            channel_mean = channel_encodings.mean(dim=[0, 1, 2])
            self.assertTrue(torch.equal(channel_encodings[0, 0, 0], channel_mean))

        for i in range(t):
            time_encodings = d_encodings[:, :, :, i, :, channel_dims:]  # [b, h, w, c_g, d]
            time_mean = time_encodings.mean(dim=[0, 1, 2, 3])
            self.assertTrue(torch.allclose(time_encodings[0, 0, 0, 0], time_mean))
            # also make sure they are different
            if i > 0:
                prev_time_encodings = d_encodings[:, :, :, i - 1, :, channel_dims:]
                prev_time_mean = prev_time_encodings.mean(dim=[0, 1, 2, 3])
                self.assertFalse((prev_time_mean == time_mean).any())
