import json
import unittest
from pathlib import Path

import torch

from src.data.config import (
    NORMALIZATION_DICT_FILENAME,
)
from src.data.earthengine.eo import (
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_LOW_RES_BANDS,
    SPACE_BANDS,
    TIME_BANDS,
    STATIC_BANDS,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
)
from src.eval.landsat_eval import LandsatEvalDataset
from src.utils import config_dir
from src.masking import _aggregate_mask_per_channel_group

class TestEval(unittest.TestCase):
    def test_create_masks(self):
        # create a test invalid data masks
        s_t_h_m = torch.ones((10, 10, 3, len(SPACE_TIME_HIGH_RES_BANDS)))
        s_t_h_m[2:5, 2:5, :, 1] = 0  # valid data in band 1
        s_t_h_m[4:7, 4:7, :, 2] = 0  # valid data in band 2
        s_t_h_m[6:9, 6:9, 2, 6] = 0  # valid data in band 6

        s_t_m_m = torch.ones((10, 10, 3, len(SPACE_TIME_MED_RES_BANDS)))
        s_t_m_m[1:4, 1:4, 1, 0] = 0
        s_t_m_m[1:4, 1:4, 1, 1] = 0
        s_t_m_m[5:8, 5:8, :, 1] = 0

        s_t_l_m = torch.ones((10, 10, 3, len(SPACE_TIME_LOW_RES_BANDS)))
        s_t_l_m[8:10, 8:10, :, 0] = 0
        s_t_l_m[4:10, 4:10, :, 1] = 0
        s_t_l_m[8:10, 8:10, 2, 2] = 0

        sp_m = torch.ones((10, 10, len(SPACE_BANDS)))
        sp_m[3:6, 3:6, 0] = 0
        sp_m[5:8, 5:8, 1] = 0
        sp_m[1:10, 1:10, 2] = 0

        t_m = torch.ones((3, len(TIME_BANDS)))
        t_m[0, 0] = 0
        t_m[1, 0] = 0

        st_m = torch.ones((len(STATIC_BANDS),))

        # apply the channel aggregation function
        (s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m) = _aggregate_mask_per_channel_group(
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
        )

        assert s_t_h_m.shape == (10, 10, 3, len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX))
        assert s_t_m_m.shape == (10, 10, 3, len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX))
        assert s_t_l_m.shape == (10, 10, 3, len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX))
        assert sp_m.shape == (10, 10, len(SPACE_BAND_GROUPS_IDX))
        assert t_m.shape == (3, len(TIME_BANDS_GROUPS_IDX))
        assert st_m.shape == (len(STATIC_BAND_GROUPS_IDX),)

        s_t_h_m_valid = torch.zeros_like(s_t_h_m, dtype=torch.bool)
        s_t_h_m_valid[6:9, 6:9, 2, 2] = True
        assert s_t_h_m_valid.all() == 0  # group 3 should be valid
        assert s_t_h_m[~s_t_h_m_valid].all() == 1  # other groups should be invalid

        s_t_m_m_valid = torch.zeros_like(s_t_m_m, dtype=torch.bool)
        s_t_m_m_valid[1:4, 1:4, 1, 0] = True
        assert s_t_m_m_valid.all() == 0
        assert s_t_m_m[~s_t_m_m_valid].all() == 1

        s_t_l_m_valid = torch.zeros_like(s_t_l_m, dtype=torch.bool)
        s_t_l_m_valid[8:10, 8:10, 2, 0] = True
        assert s_t_l_m_valid.all() == 0
        assert s_t_l_m[~s_t_l_m_valid].all() == 1

        sp_m_valid = torch.zeros_like(sp_m, dtype=torch.bool)
        sp_m_valid[5:7, 5:7, 0] = True
        assert sp_m_valid.all() == 0
        assert sp_m[~sp_m_valid].all() == 1

        assert t_m.all() == 1  # all time groups should be invalid since valid data comes from different channel groups
        assert st_m.all() == 1  # static bands should remain unchanged        

if __name__ == "__main__":
    unittest.main()