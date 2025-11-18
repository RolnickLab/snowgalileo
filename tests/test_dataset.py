import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.config import (
    NORMALIZATION_DICT_FILENAME,
    NUM_HIGH_RES_PIXELS_PER_DIM,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_TIMESTEPS,
)
from src.data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_LOW_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    Dataset,
    Normalizer,
    to_cartesian,
)
from src.utils import config_dir

TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs"
BROKEN_TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs_broken"
UNBROKEN_TEST_FILES = [TIFS_FOLDER / x for x in TIFS_FOLDER.glob("*.tif")]
BROKEN_TEST_FILE = list(BROKEN_TIFS_FOLDER.glob("*.tif"))


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        for test_file in UNBROKEN_TEST_FILES:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = Dataset._tif_to_array(test_file)
            self.assertFalse(np.isnan(s_t_h_x).any())
            self.assertFalse(np.isnan(s_t_m_x).any())
            self.assertFalse(np.isnan(s_t_l_x).any())
            self.assertFalse(np.isnan(sp_x).any())
            self.assertFalse(np.isnan(t_x).any())
            self.assertFalse(np.isnan(st_x).any())
            self.assertFalse(np.isinf(s_t_h_x).any())
            self.assertFalse(np.isinf(s_t_m_x).any())
            self.assertFalse(np.isinf(s_t_l_x).any())
            self.assertFalse(np.isinf(sp_x).any())
            self.assertFalse(np.isinf(t_x).any())
            self.assertFalse(np.isinf(st_x).any())
            self.assertEqual(
                sp_x.shape[0],
                s_t_h_x.shape[0],
            )
            self.assertEqual(
                valid_data_mask_s_t_h.shape[0],
                valid_data_mask_s_t_h.shape[1],
                valid_data_mask_sp.shape[0],
            )
            self.assertEqual(
                sp_x.shape[0],
                valid_data_mask_sp.shape[1],
                NUM_HIGH_RES_PIXELS_PER_DIM,
            )
            self.assertEqual(
                s_t_m_x.shape[0],
                s_t_m_x.shape[1],
                valid_data_mask_s_t_m.shape[0],
            )
            self.assertEqual(
                valid_data_mask_s_t_m.shape[1],
                NUM_MED_RES_PIXELS_PER_DIM,
            )
            self.assertEqual(
                s_t_l_x.shape[0],
                valid_data_mask_s_t_l.shape[0],
                NUM_LOW_RES_PIXELS_PER_DIM,
            )
            self.assertEqual(t_x.shape[0], s_t_m_x.shape[2], s_t_l_x.shape[2])
            self.assertEqual(
                t_x.shape[0],
                valid_data_mask_s_t_h.shape[2],
                valid_data_mask_s_t_l.shape[2],
            )
            self.assertEqual(
                t_x.shape[0],
                valid_data_mask_t.shape[0],
                NUM_TIMESTEPS,
            )
            self.assertEqual(
                len(SPACE_TIME_HIGH_RES_BANDS), s_t_h_x.shape[-1], valid_data_mask_s_t_h.shape[-1]
            )
            self.assertEqual(
                len(SPACE_TIME_MED_RES_BANDS), s_t_m_x.shape[-1], valid_data_mask_s_t_m.shape[-1]
            )
            self.assertEqual(
                len(SPACE_TIME_LOW_RES_BANDS), s_t_l_x.shape[-1], valid_data_mask_s_t_l.shape[-1]
            )
            self.assertEqual(len(SPACE_BANDS), sp_x.shape[-1], valid_data_mask_sp.shape[-1])
            self.assertEqual(len(TIME_BANDS), t_x.shape[-1], valid_data_mask_t.shape[-1])
            self.assertEqual(len(STATIC_BANDS), st_x.shape[-1], valid_data_mask_st.shape[-1])

    def test_files_are_replaced(self):
        ds = Dataset(BROKEN_TIFS_FOLDER, download=False)

        for b in ds:
            assert len(b) == 13

        # for file in BROKEN_TEST_FILE:
        # assert (BROKEN_TIFS_FOLDER / file) not in ds.tifs

    def test_normalization(self):
        ds = Dataset(TIFS_FOLDER, download=False)
        o = ds.load_normalization_values(path=Path(config_dir / NORMALIZATION_DICT_FILENAME))
        for t in [
            "space_time_high_res",
            "space_time_med_res",
            "space_time_low_res",
            "space",
            "time",
            "static",
        ]:
            subdict = o[t]
            self.assertTrue("mean" in subdict)
            self.assertTrue("std" in subdict)
            self.assertTrue(len(subdict["mean"]) == len(subdict["std"]))
        normalizer = Normalizer(normalizing_dicts=o)
        ds.normalizer = normalizer
        for b in ds:
            for t in b:
                self.assertFalse(np.isnan(t).any())

    def test_subset_image_with_minimum_size(self):
        input = np.ones((3, 3, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, input, input, months, static, months, 3, 1)
        self.assertTrue(np.equal(input, output[0]).all())
        self.assertTrue(np.equal(input, output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())
        print("Test ok :-)")

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        months = static = np.ones(1)
        self.assertRaises(
            AssertionError,
            Dataset.subset_image,
            input,
            input,
            input,
            input,
            months,
            static,
            months,
            3,
            1,
        )

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, input, input, months, static, months, 3, 1)
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[0]).all())
        self.assertTrue(np.equal(np.ones((3, 3, 1)), output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())

    def test_latlon_checks_float(self):
        # just checking it runs
        _ = to_cartesian(
            30.0,
            40.0,
        )
        with self.assertRaises(AssertionError):
            to_cartesian(1000.0, 1000.0)

    def test_latlon_checks_np(self):
        # just checking it runs
        _ = to_cartesian(np.array([30.0]), np.array([40.0]))
        with self.assertRaises(AssertionError):
            to_cartesian(np.array([1000.0]), np.array([1000.0]))

    def test_process_h5pys(self):
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            dataset = Dataset(
                TIFS_FOLDER,
                download=False,
                h5py_folder=tempdir,
                h5pys_only=False,
            )
            dataset.process_h5pys()

            h5py_files = list(tempdir.glob("*.h5"))
            self.assertEqual(len(h5py_files), 2)
            for h5_file in h5py_files:
                with h5py.File(h5_file, "r") as f:
                    # mostly checking it can be read
                    self.assertEqual(f["t_x"].shape[0], NUM_TIMESTEPS)


if __name__ == "__main__":
    unittest.main()
