import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    Dataset,
    to_cartesian,
)

BROKEN_FILE = "min_lat=24.7979_min_lon=-105.1508_max_lat=24.8069_max_lon=-105.141_dates=2022-01-01_2023-12-31.tif"
TEST_FILENAMES = [
    "min_lat=5.4427_min_lon=101.4016_max_lat=5.4518_max_lon=101.4107_dates=2022-01-01_2023-12-31.tif",
    "min_lat=-27.6721_min_lon=25.6796_max_lat=-27.663_max_lon=25.6897_dates=2022-01-01_2023-12-31.tif",
]
TIFS_FOLDER = Path(__file__).parents[1] / "data/tifs"
TEST_FILES = [TIFS_FOLDER / x for x in TEST_FILENAMES]


class TestDataset(unittest.TestCase):
    def test_tif_to_array(self):
        for test_file in TEST_FILES:
            s_t_x, sp_x, t_x, st_x, months = Dataset._tif_to_array(test_file)
            self.assertFalse(np.isnan(s_t_x).any())
            self.assertFalse(np.isnan(sp_x).any())
            self.assertFalse(np.isnan(t_x).any())
            self.assertFalse(np.isnan(st_x).any())
            self.assertFalse(np.isinf(s_t_x).any())
            self.assertFalse(np.isinf(sp_x).any())
            self.assertFalse(np.isinf(t_x).any())
            self.assertFalse(np.isinf(st_x).any())
            self.assertEqual(sp_x.shape[0], s_t_x.shape[0])
            self.assertEqual(sp_x.shape[1], s_t_x.shape[1])
            self.assertEqual(t_x.shape[0], s_t_x.shape[2])
            self.assertEqual(len(SPACE_TIME_BANDS), s_t_x.shape[-1])
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

    def test_subset_image_with_minimum_size(self):
        input = np.ones((3, 3, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, months, static, months, 3, 1)
        self.assertTrue(np.equal(input, output[0]).all())
        self.assertTrue(np.equal(input, output[1]).all())
        self.assertTrue(np.equal(months, output[2]).all())

    def test_subset_with_too_small_image(self):
        input = np.ones((2, 2, 1))
        months = static = np.ones(1)
        self.assertRaises(
            AssertionError, Dataset.subset_image, input, input, months, static, months, 3, 1
        )

    def test_subset_with_larger_images(self):
        input = np.ones((5, 5, 1))
        months = static = np.ones(1)
        output = Dataset.subset_image(input, input, months, static, months, 3, 1)
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

    def test_latlon_checks_tensor(self):
        # just checking it runs
        _ = to_cartesian(torch.tensor([30.0]), torch.tensor([40.0]))
        with self.assertRaises(AssertionError):
            to_cartesian(torch.tensor([1000.0]), torch.tensor([1000.0]))

    def test_cache_in_ram_and_h5pys(self):
        with tempfile.TemporaryDirectory() as tempdir_str:
            tempdir = Path(tempdir_str)
            dataset = Dataset(
                TIFS_FOLDER,
                download=False,
                h5py_folder=tempdir,
                h5pys_only=False,
                cache_in_ram=True,
            )
            assert len(dataset) == 3
            for i in range(len(dataset)):
                _ = dataset[i]
            # the broken tif shouldn't get added
            assert len(dataset.dataset_outputs) == 2, len(dataset.dataset_outputs)

            # then with h5pys only
            dataset_h5pys = Dataset(
                TIFS_FOLDER,
                download=False,
                h5py_folder=tempdir,
                h5pys_only=True,
                cache_in_ram=True,
            )
            assert len(dataset_h5pys.tifs) == 0
            assert len(dataset_h5pys) == 2
            for i in range(len(dataset_h5pys)):
                _ = dataset_h5pys[i]
            assert len(dataset_h5pys.dataset_outputs) == 2, len(dataset.dataset_outputs)
