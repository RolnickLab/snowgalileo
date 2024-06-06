import unittest

import numpy as np

from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
)
from src.eval.cropharvest_eval import BANDS, BinaryCropHarvestEval


class TestCropHarvest(unittest.TestCase):
    def test_to_presto_arrays(self):
        for include_latlons in [True, False]:

            class BinaryCropHarvestEvalNoDownload(BinaryCropHarvestEval):
                def __init__(self, include_latlons: bool = True):
                    self.include_latlons = include_latlons

            eval = BinaryCropHarvestEvalNoDownload(include_latlons)
            b, t = 8, 12
            array = np.ones((b, t, len(BANDS)))
            latlons = np.ones((b, 2))
            (
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = eval.cropharvest_array_to_normalized_presto(array, latlons, start_month=1)

            self.assertEqual(s_t_x.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS)))
            self.assertEqual(s_t_m.shape, (b, 1, 1, t, len(SPACE_TIME_BANDS_GROUPS_IDX)))
            self.assertTrue((s_t_m == 0).all())
            self.assertEqual(sp_x.shape, (b, 1, 1, len(SPACE_BANDS)))
            self.assertEqual(sp_m.shape, (b, 1, 1, len(SPACE_BAND_GROUPS_IDX)))
            self.assertTrue(
                (sp_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("SRTM")] == 0).all()
            )
            self.assertTrue(
                (sp_m[:, :, :, list(SPACE_BAND_GROUPS_IDX.keys()).index("DW")] == 1).all()
            )
            self.assertEqual(t_x.shape, (b, t, len(TIME_BANDS)))
            self.assertEqual(t_m.shape, (b, t, len(TIME_BAND_GROUPS_IDX)))
            self.assertEqual(st_x.shape, (b, len(STATIC_BANDS)))
            self.assertEqual(st_m.shape, (b, len(STATIC_BAND_GROUPS_IDX)))
            if not include_latlons:
                self.assertTrue((st_m == 1).all())
            self.assertEqual(months.shape, (b, t))
