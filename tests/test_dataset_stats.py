import unittest

import numpy as np
from einops import rearrange

from src.data.dataset import Dataset
from src.data.dataset_stats import RunningStatistics


class TestDatasetStatistics(unittest.TestCase):
    def test_dataset_statistics(self):
        mean_1, std_1 = 0, 0.1
        mean_2, std_2 = -1, 1.5
        a = rearrange(
            np.random.normal(loc=mean_1, scale=std_1, size=10000),
            "(b h w) -> b h w",
            b=100,
            h=10,
            w=10,
        )
        b = rearrange(
            np.random.normal(loc=mean_2, scale=std_2, size=10000),
            "(b h w) -> b h w",
            b=100,
            h=10,
            w=10,
        )
        combined = np.stack((a, b), axis=-1)  # b, h, w, 2

        stats = RunningStatistics()
        for i in range(combined.shape[1]):
            stats.update(combined[i])

        mean, std = stats.mean, stats.std
        self.assertTrue(len(mean) == 2)
        self.assertTrue(np.isclose(mean[0], mean_1, atol=0.1))
        self.assertTrue(np.isclose(mean[1], mean_2, atol=0.1))
        self.assertTrue(np.isclose(std[0], std_1, atol=0.1))
        self.assertTrue(np.isclose(std[1], std_2, atol=0.1))

    def test_dataset_stats_v2(self):
        mean_1, std_1 = 0, 0.1
        mean_2, std_2 = -1, 1.5
        a = rearrange(
            np.random.normal(loc=mean_1, scale=std_1, size=10000),
            "(b h w) -> b h w",
            b=100,
            h=10,
            w=10,
        )
        b = rearrange(
            np.random.normal(loc=mean_2, scale=std_2, size=10000),
            "(b h w) -> b h w",
            b=100,
            h=10,
            w=10,
        )
        combined = np.stack((a, b), axis=-1)  # b, h, w, 2
        interim = {"n": 0, "mean": np.zeros(2), "M2": np.zeros(2)}
        for i in range(combined.shape[1]):
            interim = Dataset._update_normalizing_values(combined[i], interim)
        output = Dataset._calculate_normalizing_dict(interim)
        self.assertTrue(np.isclose(output["mean"][0], mean_1, atol=0.1))
        self.assertTrue(np.isclose(output["mean"][1], mean_2, atol=0.1))
        self.assertTrue(np.isclose(output["std"][0], std_1, atol=0.1))
        self.assertTrue(np.isclose(output["std"][1], std_2, atol=0.1))
