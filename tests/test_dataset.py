import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_HIGH_RES_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    Dataset,
    Normalizer,
    to_cartesian,
)

BROKEN_FILE = "min_lat=42.0017_min_lon=42.8257_max_lat=42.0108_max_lon=42.8378_season=late_dates=2019-04-05_2019-04-20.tif"
TEST_FILENAMES = [
    "min_lat=43.5142_min_lon=6.685_max_lat=43.5233_max_lon=6.6973_season=early_dates=2017-11-30_2017-12-15.tif",
    "min_lat=42.0017_min_lon=42.8257_max_lat=42.0108_max_lon=42.8378_season=mid_dates=2019-01-02_2019-01-17.tif",
]
TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs"
TEST_FILES = [TIFS_FOLDER / x for x in TEST_FILENAMES]


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        for test_file in TEST_FILES:
            s_t_h_x, s_t_m_x, s_t_l_x, sp_x, t_x, st_x, months = Dataset._tif_to_array(test_file)
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
            self.assertEqual(sp_x.shape[0], s_t_h_x.shape[0], s_t_m_x.shape[0], s_t_l_x.shape[0])
            self.assertEqual(sp_x.shape[1], s_t_h_x.shape[1], s_t_m_x.shape[1], s_t_l_x.shape[1])
            self.assertEqual(t_x.shape[0], s_t_h_x.shape[2], s_t_m_x.shape[2], s_t_l_x.shape[2])
            self.assertEqual(len(SPACE_TIME_HIGH_RES_BANDS), s_t_h_x.shape[-1])
            self.assertEqual(sp_x.shape[0], s_t_x.shape[0])
            self.assertEqual(sp_x.shape[1], s_t_x.shape[1])
            self.assertEqual(t_x.shape[0], s_t_x.shape[2])
            self.assertEqual(len(SPACE_TIME_HIGH_RES_BANDS), s_t_x.shape[-1])
            self.assertEqual(len(SPACE_BANDS), sp_x.shape[-1])
            self.assertEqual(len(TIME_BANDS), t_x.shape[-1])
            self.assertEqual(len(STATIC_BANDS), st_x.shape[-1])
            self.assertEqual(months[0], 0)

    def test_files_are_replaced(self):
        ds = Dataset(TIFS_FOLDER, download=False)
        assert TIFS_FOLDER / BROKEN_FILE in ds.tifs

        for b in ds:
            assert len(b) == 5
        assert TIFS_FOLDER / BROKEN_FILE not in ds.tifs

    def test_normalization(self):
        ds = Dataset(TIFS_FOLDER, download=False)
        o = ds.load_normalization_values(path=Path("config/normalizing_dict_500m.json"))
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
        output = Dataset.subset_image_and_mask(
            input, input, months, static, months, input, input, months, static, 3, 1
        )
        self.assertTrue(np.equal(input, output[0]).all())
        self.assertTrue(np.equal(input, output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())
        print("Test ok :-)")

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        months = static = np.ones(1)
        self.assertRaises(
            AssertionError,
            Dataset.subset_image_and_mask,
            input,
            input,
            months,
            static,
            months,
            input,
            input,
            months,
            static,
            3,
            1,
        )

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image_and_mask(
            input, input, months, static, months, input, input, months, static, 3, 1
        )
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
                    self.assertEqual(f["t_x"].shape[0], 24)


if __name__ == "__main__":
    unittest.main()
