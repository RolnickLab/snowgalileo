import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import unittest
import numpy as np
import torch
from eval.so2sat_eval import So2SatDataset


class TestSo2Sat(unittest.TestCase):
    def test_so2sat_dataset(self):
        dataset = So2SatDataset(split="validation")
        sample = dataset[0]
        d_x, s_x, d_m, s_m, m = sample[0]
        label = sample[1]
        # input shape expected by Presto
        self.assertEqual(d_x.shape, (32,32,1,24))
        self.assertEqual(s_x.shape, (32,32,2))
        self.assertEqual(d_m.shape, (32,32,1,9))
        self.assertEqual(s_m.shape, (32,32,1))
        # sa2sat has only one timestep
        self.assertEqual(m.shape, (1,))
        self.assertEqual(label.shape, (17,))
        self.assertFalse(torch.any(torch.isnan(d_x)))
        # no month in so2sat so set to zero
        self.assertEqual(m[0], 0)
        self.assertTrue(torch.all(torch.logical_or(d_m == 0, s_m == 1)))
        # no static data in so2sat so added as zeros and masked out
        self.assertTrue(torch.all(s_x == 0))
        self.assertTrue(torch.all(s_m == 1))
        # labels are one-hot encoded
        self.assertTrue(torch.all(torch.logical_or(label == 0, label == 1)))