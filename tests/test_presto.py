import unittest
from pathlib import Path

import torch

from src.data import DYNAMIC_BANDS_GROUPS_IDX, Dataset
from src.data.masking import mask_by_presto_pixels_time
from src.presto import Encoder, PrestoDecoder

TEST_FILE = (
    Path(__file__).parents[1]
    / "data/tifs_min_lat=19.2005_min_lon=-155.6227_max_lat=19.2132_max_lon=-155.6094_dates=2022-01-01_2023-12-31.tiff"
)


class TestPresto(unittest.TestCase):
    def test_presto_encoder(self):
        model = Encoder(embedding_size=2, num_heads=1)
        dynamic_data, static_data = Dataset.tif_to_array(TEST_FILE)
        output = mask_by_presto_pixels_time(dynamic_data, static_data, mask_ratio=0.25)
        # unsqueeze to add the batch dimension
        output_t = [torch.from_numpy(x).float().unsqueeze(0) for x in output]
        with torch.no_grad():
            # for now, we just make sure it all runs
            model(*output_t)

    def test_presto_end_to_end(self):
        embedding_size = 2
        encoder = Encoder(embedding_size=embedding_size, num_heads=1)
        decoder = PrestoDecoder(embedding_size=embedding_size, num_heads=1)
        dynamic_data, static_data = Dataset.tif_to_array(TEST_FILE)
        output = mask_by_presto_pixels_time(dynamic_data, static_data, mask_ratio=0.25)
        # unsqueeze to add the batch dimension
        output_t = [torch.from_numpy(x).float().unsqueeze(0) for x in output]
        with torch.no_grad():
            # for now, we just make sure it all runs
            output = encoder(*output_t)
            decoder(*output)

    def test_presto_decoder_add_masks(self):
        embedding_size = 2
        decoder = PrestoDecoder(embedding_size=embedding_size, num_heads=1)
        b, h, w, t = 5, 6, 7, 8
        d_x = torch.ones(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX), embedding_size)
        d_m = torch.zeros(b, h, w, t, len(DYNAMIC_BANDS_GROUPS_IDX))
        d_m[:, :, :, 0] = 1  # mask the first timestep
        with torch.no_grad():
            o, o_m = decoder.add_masks(d_x, d_m)
        self.assertTrue((o_m == 0).all())
        self.assertTrue((o[:, :, :, 0] == 0).all())
        self.assertTrue((o[:, :, :, 1:] == 1).all())
