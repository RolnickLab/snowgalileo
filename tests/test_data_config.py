import unittest

from src.data.config import CHANNEL_WISE_INVALID_DATA_THRESHOLDS, MODALITIES
from src.data.earthengine.eo import (
    CLOUD_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_DIV_VALUES,
    SPACE_IMAGE_FUNCTIONS,
    SPACE_SHIFT_VALUES,
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_DIV_VALUES,
    SPACE_TIME_HIGH_RES_SHIFT_VALUES,
    SPACE_TIME_LOW_RES_BANDS,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_DIV_VALUES_NP,
    SPACE_TIME_LOW_RES_SHIFT_VALUES_NP,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_DIV_VALUES,
    SPACE_TIME_MED_RES_SHIFT_VALUES,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    STATIC_DIV_VALUES_NP,
    STATIC_SHIFT_VALUES_NP,
    TIME_BANDS,
    TIME_BANDS_GROUPS_IDX,
    TIME_DIV_VALUES,
    TIME_IMAGE_FUNCTIONS,
    TIME_SHIFT_VALUES,
)

array_types = {
    "s_t_h_x": SPACE_TIME_HIGH_RES_BANDS,
    "s_t_m_x": SPACE_TIME_MED_RES_BANDS,
    "s_t_l_x": SPACE_TIME_LOW_RES_BANDS,
    "sp_x": SPACE_BANDS,
    "t_x": TIME_BANDS,
    "st_x": STATIC_BANDS,
    "clouds": CLOUD_BANDS,
}


class TestConfig(unittest.TestCase):
    def test_config(self):
        # Flatten all index lists and collect unique indices
        s_t_h_bands_from_idx = set()
        for key, indices in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.items():
            s_t_h_bands_from_idx.update(indices)

        s_t_h_bands_from_idx = len(s_t_h_bands_from_idx)

        s_t_m_bands_from_idx = set()
        for key, indices in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.items():
            s_t_m_bands_from_idx.update(indices)

        s_t_m_bands_from_idx = len(s_t_m_bands_from_idx)

        s_t_l_bands_from_idx = set()
        for key, indices in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.items():
            s_t_l_bands_from_idx.update(indices)

        s_t_l_bands_from_idx = len(s_t_l_bands_from_idx)

        space_bands_from_idx = set()
        for key, indices in SPACE_BAND_GROUPS_IDX.items():
            space_bands_from_idx.update(indices)

        space_bands_from_idx = len(space_bands_from_idx)

        time_bands_from_idx = set()
        for key, indices in TIME_BANDS_GROUPS_IDX.items():
            if key != "NDVI" and key != "NDSI":
                time_bands_from_idx.update(indices)

        time_bands_from_idx = len(time_bands_from_idx)

        static_bands_from_idx = set()
        for key, indices in STATIC_BAND_GROUPS_IDX.items():
            static_bands_from_idx.update(indices)

        static_bands_from_idx = len(static_bands_from_idx)

        assert (
            len(SPACE_TIME_HIGH_RES_BANDS)
            == len(SPACE_TIME_HIGH_RES_SHIFT_VALUES)
            == len(SPACE_TIME_HIGH_RES_DIV_VALUES)
            == s_t_h_bands_from_idx
        )
        assert (
            len(SPACE_TIME_MED_RES_BANDS)
            == len(SPACE_TIME_MED_RES_SHIFT_VALUES)
            == len(SPACE_TIME_MED_RES_DIV_VALUES)
            == s_t_m_bands_from_idx
        )
        assert (
            len(SPACE_TIME_LOW_RES_BANDS)
            == len(SPACE_TIME_LOW_RES_SHIFT_VALUES_NP)
            == len(SPACE_TIME_LOW_RES_DIV_VALUES_NP)
            == s_t_l_bands_from_idx
        )
        assert (
            len(SPACE_BANDS)
            == len(SPACE_SHIFT_VALUES)
            == len(SPACE_DIV_VALUES)
            == space_bands_from_idx
        )
        assert (
            len(TIME_BANDS)
            == len(TIME_SHIFT_VALUES)
            == len(TIME_DIV_VALUES)
            == time_bands_from_idx
        )
        assert (
            len(STATIC_BANDS)
            == len(STATIC_SHIFT_VALUES_NP)
            == len(STATIC_DIV_VALUES_NP)
            == static_bands_from_idx
        )

        for array_type, bands in array_types.items():
            # Check length of the thresholds
            if array_type == "clouds":
                continue
            assert len(CHANNEL_WISE_INVALID_DATA_THRESHOLDS[array_type]) == len(bands)

        num_sp_mod = 0
        num_t_mod = 0
        num_st_mod = 0
        num_s_t_h_mod = 0
        num_s_t_m_mod = 0
        num_s_t_l_mod = 0
        num_cloud_mod = 0

        for key, modality in MODALITIES.items():
            assert modality["shape_type"] in array_types.keys(), (
                f"Unknown shape type: {modality['shape_type']}"
            )
            if key != "location":
                assert modality["original_resolution"] is not None, (
                    f"Original resolution is None for {key}"
                )
            assert modality["active"] is not None, f"Active is None for {key}"
            assert modality["export"] is not None, f"Export is None for {key}"

            if key != "ndvi" and key != "ndsi":
                if modality["shape_type"] == "sp_x":
                    num_sp_mod += 1
                elif modality["shape_type"] == "t_x":
                    num_t_mod += 1
                elif modality["shape_type"] == "st_x":
                    num_st_mod += 1
                elif modality["shape_type"] == "s_t_h_x":
                    num_s_t_h_mod += 1
                elif modality["shape_type"] == "clouds":
                    num_cloud_mod += 1
                elif modality["shape_type"] == "s_t_m_x":
                    num_s_t_m_mod += 1
                elif modality["shape_type"] == "s_t_l_x":
                    num_s_t_l_mod += 1

        assert num_sp_mod == len(SPACE_IMAGE_FUNCTIONS)
        assert num_t_mod + num_s_t_h_mod + num_s_t_m_mod + num_s_t_l_mod + num_cloud_mod == len(
            TIME_IMAGE_FUNCTIONS
        )
