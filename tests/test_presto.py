import unittest
from pathlib import Path

import torch

from src.data import Dataset
from src.data.masking import mask_by_presto_pixels_time
from src.presto import Encoder

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
