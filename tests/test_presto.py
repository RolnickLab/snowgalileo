import unittest
from pathlib import Path

import torch

from src.data import DYNAMIC_BANDS_GROUPS_IDX
from src.data.config import NUM_TIMESTEPS, PRESTO_INPUT_SIZE
from src.masked_datasets import (
    STATIC_BAND_GROUPS_IDX,
    MaskedOutput,
    PrestoToPrestoMaskedDataset,
)
from src.presto import Encoder, PrestoDecoder

DATA_FOLDER = Path(__file__).parents[1] / "data"
TEST_FILE = (
    DATA_FOLDER
    / "tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2022-01-01_2023-12-31.tiff"
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

    def test_presto_encoder(self):
        model = Encoder(embedding_size=8, num_heads=1)
        ds = PrestoToPrestoMaskedDataset(DATA_FOLDER, 0.25, False)
        output = ds[0]
        with torch.no_grad():
            # for now, we just make sure it all runs
            model(*self.to_tensor_with_batch_d(output))

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
            output = encoder(*self.to_tensor_with_batch_d(output))
            output = decoder(*output)
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
