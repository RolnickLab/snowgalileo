import random
import unittest
import torch

from src.eval.landsat_baselines import (
    forward_filling_masked_data_per_channel_else_median,
)


class TestMasking(unittest.TestCase):
    def test_forward_filling_masked_data_per_channel_else_median(self):
        # test that filling works correctly
        # four dim data simulates s_t_h_x, s_t_m_x, s_t_l_x, and t_x
        four_dim_data_c1 = torch.tensor(
            [[[[1.0, float("nan"), float("nan"), 4.0, 5.0]]]]
        )  # shape (B=1, S=1, C=1, T=5)
        four_dim_data_c2 = torch.tensor([[[[float("nan"), float("nan"), 3.0, float("nan"), 5.0]]]])
        four_dim_data_c3 = torch.tensor(
            [[[[float("nan"), float("nan"), float("nan"), float("nan"), float("nan")]]]]
        )

        # three dim data simulates sp_x and st_x
        three_dim_data_c1 = torch.tensor([[[4.0]]])  # shape (B=1, S=1, C=1)
        three_dim_data_c2 = torch.tensor([[[3.0]]])
        three_dim_data_c3 = torch.tensor([[[2.0]]])

        # stack into a single tensor
        four_dim_data = torch.cat(
            [four_dim_data_c1, four_dim_data_c2, four_dim_data_c3], dim=2
        )  # shape (B=1, S=1, C=3, T=5)
        four_dim_mask = torch.where(four_dim_data == float("nan"), 1, 0)
        four_dim_time = torch.zeros_like(four_dim_data)

        three_dim_data = torch.cat(
            [three_dim_data_c1, three_dim_data_c2, three_dim_data_c3], dim=2
        )  # shape (B=1, S=1, C=3)
        three_dim_mask = torch.where(three_dim_data == float("nan"), 1, 0)
        three_dim_time = torch.zeros_like(three_dim_data)

        filled_four_dim_data, _ = forward_filling_masked_data_per_channel_else_median(
            four_dim_data, four_dim_mask, four_dim_time
        )
        filled_three_dim_data, _ = forward_filling_masked_data_per_channel_else_median(
            three_dim_data, three_dim_mask, three_dim_time
        )
        import pdb

        pdb.set_trace()

if __name__ == '__main__':
    unittest.main()