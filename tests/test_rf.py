import json
import unittest
from pathlib import Path

import torch

from src.eval.landsat_baselines import (
    LandsatEvalRandomForest,
)


class TestMasking(unittest.TestCase):
    def test_median_replace(self):
        # create data with NaNs
        data_test1 = torch.tensor(
            [
                [
                    [float("nan"), 1.0, 2.0],
                    [3.0, float("nan"), 5.0],
                    [6.0, 7.0, float("nan")],
                    [float("nan"), float("nan"), float("nan")],
                ]
            ]
        )

        # expected result after median replacement: if same number of values below and above median,
        # the lower of the two is chosen
        expected_test1 = torch.tensor(
            [[[1.0, 1.0, 2.0], [3.0, 3.0, 5.0], [6.0, 7.0, 6.0], [3.0, 3.0, 5.0]]]
        )

        result_test1 = LandsatEvalRandomForest.replace_masked_data_with_median_per_dimension(
            data_test1, torch.where(torch.isnan(data_test1), 1, 0)
        )

        self.assertTrue(torch.equal(result_test1, expected_test1))

        data_test2 = torch.tensor(
            [
                [
                    [[float("nan"), float("nan")], [float("nan"), float("nan")]],
                    [[float("nan"), float("nan")], [float("nan"), float("nan")]],
                    [[1.0, 2.0], [3.0, 4.0]],
                ]
            ]
        )

        expected_test2 = torch.tensor(
            [[[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]]]]
        )

        result_test2 = LandsatEvalRandomForest.replace_masked_data_with_median_per_dimension(
            data_test2, torch.where(torch.isnan(data_test2), 1, 0)
        )

        self.assertTrue(torch.equal(result_test2, expected_test2))

        data_test3 = torch.tensor(
            [
                [
                    [[float("nan"), 3.0, float("nan")], [1.0, 8.0, 10.0]],
                    [
                        [float("nan"), float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ],
                    [[1.0, 6.0, 6.0], [3.0, 2.0, 4.0]],
                ],
                [
                    [[7.0, 3.0, float("nan")], [1.0, 8.0, 10.0]],
                    [
                        [7.0, float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ],
                    [[1.0, 2.0, float("nan")], [3.0, float("nan"), 4.0]],
                ],
            ]
        )

        expected_test3 = torch.tensor(
            [
                [
                    [[3.0, 3.0, 3.0], [1.0, 8.0, 10.0]],
                    [[7.0, 3.0, 6.0], [1.0, 2.0, 4.0]],
                    [[1.0, 6.0, 6.0], [3.0, 2.0, 4.0]],
                ],
                [
                    [[7.0, 3.0, 3.0], [1.0, 8.0, 10.0]],
                    [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
                    [[1.0, 2.0, 1.0], [3.0, 3.0, 4.0]],
                ],
            ]
        )

        result_test3 = LandsatEvalRandomForest.replace_masked_data_with_median_per_dimension(
            data_test3, torch.where(torch.isnan(data_test3), 1, 0)
        )

        self.assertTrue(torch.equal(result_test3, expected_test3))

    def test_forward_filling(self):
        data_test1 = torch.tensor(
            [
                [
                    [[float("nan"), 3.0, float("nan")], [1.0, 8.0, 10.0]],
                    [
                        [float("nan"), float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ],
                    [[1.0, 6.0, 6.0], [3.0, 2.0, 4.0]],
                ],
                [
                    [[7.0, 3.0, float("nan")], [1.0, 8.0, 10.0]],
                    [
                        [7.0, float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ],
                    [[1.0, 2.0, float("nan")], [10.0, float("nan"), 1.0]],
                ],
            ]
        )

        expected_test1 = torch.tensor(
            [
                [
                    [[3.0, 3.0, 3.0], [1.0, 8.0, 10.0]],
                    [[7.0, 3.0, 6.0], [1.0, 2.0, 4.0]],
                    [[1.0, 6.0, 6.0], [3.0, 2.0, 4.0]],
                ],
                [
                    [[7.0, 3.0, 3.0], [1.0, 8.0, 10.0]],
                    [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]],
                    [[1.0, 2.0, 2.0], [10.0, 10.0, 1.0]],
                ],
            ]
        )

        result_test3 = LandsatEvalRandomForest.replace_masked_data_with_median_per_dimension(
            data_test1, torch.where(torch.isnan(data_test1), 1, 0)
        )

        self.assertTrue(torch.equal(result_test3, expected_test1))

    def _test_aggregation_patch_size_2(
        self,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        s_t_m_m,
        s_t_l_m,
        sp_m,
        t_m,
        st_m,
        month,
    ):
        with (
            Path(__file__).parents[1]
            / Path("src/eval/eval_configs")
            / Path("landsat_eval_5_95.json")
        ).open("r") as f:
            config = json.load(f)

        Eval = LandsatEvalRandomForest(
            patch_size_high_res=2,
            eval_config=config,
        )
        expected = (
            torch.tensor([[[3.5, 5.5, 11.5, 13.5]]]).view(1, 4, 1, 1),
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[6.0, 6.0, 6.0, 6.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[4.5, 6.5, 12.5, 14.5]]]).view(1, 4, 1),
            torch.tensor([[[9.0, 9.0, 9.0, 9.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[19.0, 19.0, 19.0, 19.0]]]).view(1, 4, 1),
            torch.tensor([[[1.0, 0.0, 0.0, 1.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[1.0, 1.0, 0.0, 1.0]]]).view(1, 4, 1),
            torch.tensor([[[0.0, 0.0, 0.0, 0.0]]]).view(1, 4, 1, 1),
            torch.tensor([[[1.0, 1.0, 1.0, 1.0]]]).view(1, 4, 1),
        )

        results = Eval.aggregate_data_per_output_pixel(
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            month,
        )
        return expected, results

    def _test_aggregation_patch_size_4(
        self,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        s_t_m_m,
        s_t_l_m,
        sp_m,
        t_m,
        st_m,
        month,
    ):
        with (
            Path(__file__).parents[1]
            / Path("src/eval/eval_configs")
            / Path("landsat_eval_5_95.json")
        ).open("r") as f:
            config = json.load(f)

        Eval = LandsatEvalRandomForest(
            patch_size_high_res=4,
            eval_config=config,
        )
        expected = (
            torch.tensor(
                [
                    [
                        1.0, 2.0, 3.0, 4.0,
                        5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0,
                        13.0, 14.0, 15.0, 16.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        1.0, 1.0, 2.0, 2.0,
                        1.0, 1.0, 2.0, 2.0,
                        3.0, 3.0, 4.0, 4.0,
                        3.0, 3.0, 4.0, 4.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        6.0, 6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0, 6.0,
                        6.0, 6.0, 6.0, 6.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        2.0, 3.0, 4.0, 5.0,
                        6.0, 7.0, 8.0, 9.0,
                        9.0, 10.0, 11.0, 12.0,
                        13.0, 14.0, 15.0, 16.0,
                    ]
                ]
            ).view(1, 16, 1),
            torch.tensor(
                [
                    [
                        9.0, 9.0, 9.0, 9.0,
                        9.0, 9.0, 9.0, 9.0,
                        9.0, 9.0, 9.0, 9.0,
                        9.0, 9.0, 9.0, 9.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        19.0, 19.0, 19.0, 19.0,
                        19.0, 19.0, 19.0, 19.0,
                        19.0, 19.0, 19.0, 19.0,
                        19.0, 19.0, 19.0, 19.0,
                    ]
                ]
            ).view(1, 16, 1),
            torch.tensor(
                [
                    [
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                        1.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 1.0,
                        0.0, 0.0, 1.0, 1.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ]
                ]
            ).view(1, 16, 1),
            torch.tensor(
                [
                    [
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0,
                    ]
                ]
            ).view(1, 16, 1, 1),
            torch.tensor(
                [
                    [
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0,
                    ]
                ]
            ).view(1, 16, 1),
        )

        results = Eval.aggregate_data_per_output_pixel(
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            month,
        )
        return expected, results

    def test_aggregation(self):
        s_t_h_x = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ).view(1, 4, 4, 1, 1)
        s_t_m_x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]).view(1, 2, 2, 1, 1)
        s_t_l_x = torch.tensor([[[6.0]]]).view(1, 1, 1, 1, 1)
        sp_x = torch.tensor(
            [
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ).view(1, 4, 4, 1)
        t_x = torch.tensor([[[9.0]]]).view(1, 1, 1)
        st_x = torch.tensor([[[19.0]]]).view(1, 1)

        s_t_h_m = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ).view(1, 4, 4, 1, 1)
        s_t_m_m = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]).view(1, 2, 2, 1, 1)
        s_t_l_m = torch.tensor([[[0.0]]]).view(1, 1, 1, 1, 1)
        sp_m = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        ).view(1, 4, 4, 1)
        t_m = torch.tensor([[[0.0]]]).view(1, 1, 1)
        st_m = torch.tensor([[[1.0]]]).view(1, 1)
        month = torch.tensor(6).view(1, 1)

        expected_ps_2, results_ps_2 = self._test_aggregation_patch_size_2(
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            month,
        )
        for expected_ps_2, results_ps_2 in zip(expected_ps_2, results_ps_2):
            self.assertTrue(torch.equal(expected_ps_2, results_ps_2))

        expected_ps_4, results_ps_4 = self._test_aggregation_patch_size_4(
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m,
            month,
        )
        for expected_ps_4, results_ps_4 in zip(expected_ps_4, results_ps_4):
            self.assertTrue(torch.equal(expected_ps_4, results_ps_4))


if __name__ == "__main__":
    unittest.main()
