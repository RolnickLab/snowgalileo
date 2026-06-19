import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.config import (
    NO_DATA_VALUE,
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

TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs_test"
BROKEN_TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs_broken_test"
UNBROKEN_TEST_FILES = [TIFS_FOLDER / x for x in TIFS_FOLDER.glob("*.tif")]
BROKEN_TEST_FILE = list(BROKEN_TIFS_FOLDER.glob("*.tif"))


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        ds = Dataset(data_folder=TIFS_FOLDER)
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
            ) = ds._tif_to_array(tif_path=test_file)
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

    def test_one_hot_encoding(self):
        ds = Dataset(TIFS_FOLDER, download=False)
        for b in ds:
            sp_x = b[3]
            self.assertEqual(sp_x.shape[-1], len(SPACE_BANDS))
            # check one hot encoding of categorical variables
            # landcover (11 classes + 1 no data)
            # starting from 3rd index
            self.assertTrue(np.all(np.isin(sp_x[..., 3:14], [0, 1, NO_DATA_VALUE])))

        no_data_test = np.array([[0], [30], [90]])
        expected_output = np.array(
            [
                [[NO_DATA_VALUE] * 11],
                [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
            ]
        )
        output = Dataset.one_hot_encode_esa_worldcover(no_data_test)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_create_valid_masks(self):
        ds = Dataset(TIFS_FOLDER, download=False)

        NO_DATA_VALUE = -9999
        rng = np.random.default_rng(42)

        def insert_invalid(x, frac=0.1):
            mask = rng.random(x.shape) < frac
            x = x.copy()
            x[mask] = NO_DATA_VALUE
            return x, mask

        h, w, t, c_sth, c_stm, c_stl, c_sp, c_t, c_st = 10, 10, 8, 15, 2, 11, 14, 9, 3

        s_t_h_x = np.random.randint(1, 1000, size=(h, w, t, c_sth)).astype(float)
        s_t_m_x = np.random.randint(0, 1000, size=(h, w, t, c_stm)).astype(float)
        s_t_l_x = np.random.randint(0, 1000, size=(h, w, t, c_stl)).astype(float)
        sp_x = np.random.randint(1, 1000, size=(h, w, c_sp)).astype(float)
        t_x = np.random.randint(200, 1000, size=(t, c_t)).astype(float)
        st_x = np.random.randint(0, 1000, size=(c_st,)).astype(float)

        # insert invalid data values at random positions
        s_t_h_x, invalid_sth = insert_invalid(s_t_h_x)
        s_t_m_x, invalid_stm = insert_invalid(s_t_m_x)
        s_t_l_x, invalid_stl = insert_invalid(s_t_l_x)
        sp_x, invalid_sp = insert_invalid(sp_x)
        t_x, invalid_t = insert_invalid(t_x)
        st_x, invalid_st = insert_invalid(st_x)

        (
            valid_data_sth,
            valid_data_stm,
            valid_data_stl,
            valid_data_sp,
            valid_data_t,
            valid_data_st,
        ) = ds.create_valid_mask(
            s_t_h_x=s_t_h_x, s_t_m_x=s_t_m_x, s_t_l_x=s_t_l_x, sp_x=sp_x, t_x=t_x, st_x=st_x
        )

        self.assertEqual(valid_data_sth.shape, s_t_h_x.shape)
        self.assertEqual(valid_data_stm.shape, s_t_m_x.shape)
        self.assertEqual(valid_data_stl.shape, s_t_l_x.shape)
        self.assertEqual(valid_data_sp.shape, sp_x.shape)
        self.assertEqual(valid_data_t.shape, t_x.shape)
        self.assertEqual(valid_data_st.shape, st_x.shape)

        np.testing.assert_array_equal(valid_data_sth, (~invalid_sth).astype(int))
        np.testing.assert_array_equal(valid_data_stm, (~invalid_stm).astype(int))
        np.testing.assert_array_equal(valid_data_stl, (~invalid_stl).astype(int))
        np.testing.assert_array_equal(valid_data_sp, (~invalid_sp).astype(int))
        np.testing.assert_array_equal(valid_data_t, (~invalid_t).astype(int))
        np.testing.assert_array_equal(valid_data_st, (~invalid_st).astype(int))

        for mask in [
            valid_data_sth,
            valid_data_stm,
            valid_data_stl,
            valid_data_sp,
            valid_data_t,
            valid_data_st,
        ]:
            assert set(np.unique(mask)).issubset({0, 1})

        # test if all invalid positions have value 0, and all others a value of 1
        self.assertTrue(np.all(valid_data_sth[invalid_sth] == 0))
        self.assertTrue(np.all(valid_data_stm[invalid_stm] == 0))
        self.assertTrue(np.all(valid_data_stl[invalid_stl] == 0))
        self.assertTrue(np.all(valid_data_sp[invalid_sp] == 0))
        self.assertTrue(np.all(valid_data_t[invalid_t] == 0))
        self.assertTrue(np.all(valid_data_st[invalid_st] == 0))

        self.assertTrue(np.all(valid_data_sth[~invalid_sth] == 1))
        self.assertTrue(np.all(valid_data_stm[~invalid_stm] == 1))
        self.assertTrue(np.all(valid_data_stl[~invalid_stl] == 1))
        self.assertTrue(np.all(valid_data_sp[~invalid_sp] == 1))
        self.assertTrue(np.all(valid_data_t[~invalid_t] == 1))
        self.assertTrue(np.all(valid_data_st[~invalid_st] == 1))


if __name__ == "__main__":
    unittest.main()
