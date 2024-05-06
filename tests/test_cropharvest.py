import unittest

import numpy as np

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
)
from src.eval.cropharvest_eval import BANDS, BinaryCropHarvestEval


class TestCropHarvest(unittest.TestCase):
    def test_to_presto_arrays(self):
        b, t = 8, 12
        array = np.ones((b, t, len(BANDS)))
        (
            s_t_x,
            s_x,
            t_x,
            s_t_m,
            s_m,
            t_m,
            months,
        ) = BinaryCropHarvestEval.cropharvest_array_to_normalized_presto(array, start_month=1)

        self.assertEqual(s_t_x.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS)))
        self.assertEqual(s_t_m.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS_GROUPS_IDX)))
        self.assertTrue((s_t_m == 0).all())
        self.assertEqual(s_x.shape, (b, 1, 1, len(SPACE_BANDS)))
        self.assertEqual(s_m.shape, (b, 1, 1, len(SPACE_BAND_GROUPS_IDX)))
        self.assertTrue(
            (s_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("SRTM")] == 0).all()
        )
        self.assertTrue((s_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("DW")] == 1).all())
        self.assertEqual(t_x.shape, (b, t, len(TIME_BANDS)))
        self.assertEqual(t_m.shape, (b, t, len(TIME_BAND_GROUPS_IDX)))
        self.assertTrue((t_m == 0).all())
        self.assertEqual(months.shape, (b, t))
