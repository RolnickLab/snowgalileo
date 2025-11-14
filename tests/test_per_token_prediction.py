import unittest

import numpy as np

from src.eval.eval import EvalTask


class TestEval(unittest.TestCase):
    def check_reduce_targets_per_token(self, output_mode):
        nr_tokens = 3
        nr_outputs = 10

        eval = EvalTask(patch_size=3, output_mode=output_mode)
        eval.__setattr__("num_outputs", nr_outputs)

        # test array of with 3 tokens and 4 pixels per token
        tar_np = np.array([[1, 1, 5, 1], [0, 9, 9, 1], [2, 5, 4, 5]])

        tar_np = eval.reduce_targets_per_token(tar_np)

        if output_mode == "mode":
            assert tar_np.shape == (nr_tokens,)
            assert tar_np[0] == 1
            assert tar_np[2] == 5
        else:
            assert tar_np.shape == (nr_tokens, nr_outputs)
            # each token's values should be summed to 1
            assert np.all(np.sum(tar_np, axis=1) == 1)
            # third token has 4 pixels, half of them are 5
            assert tar_np[2][5] == 0.5
            # first token has 4 pixels, 3 of them are 1
            assert tar_np[0][1] == 0.75
            assert tar_np[0][~1 and ~5] == 0


# TODO: Use as test for our evaluation task later
"""
    def test_pastis_patch_eval(self):
        # create test arrays
        batch_size = 2
        height = 6
        width = 6
        nr_tokens = 8
        nr_pixels_per_token = 9
        model_dim = 8
        nr_timesteps = 2

        model = Encoder(embedding_size=model_dim)
        eval = EvalTask(patch_size=3)

        assert batch_size * height * width == nr_tokens * nr_pixels_per_token

        # check group per token
        tar_torch = torch.zeros((batch_size, height, width))
        s_t_x = torch.zeros(
            (
                batch_size,
                height // int(sqrt(nr_pixels_per_token)),
                width // int(sqrt(nr_pixels_per_token)),
                nr_timesteps,
                2,
                model_dim,
            )
        )
        sp_x = torch.zeros(
            (
                batch_size,
                height // int(sqrt(nr_pixels_per_token)),
                width // int(sqrt(nr_pixels_per_token)),
                3,
                model_dim,
            )
        )
        t_x = torch.zeros((batch_size, nr_timesteps, 4, model_dim))
        st_x = torch.zeros((batch_size, 2, model_dim))
        s_t_m = torch.zeros(
            (
                batch_size,
                height // int(sqrt(nr_pixels_per_token)),
                width // int(sqrt(nr_pixels_per_token)),
                nr_timesteps,
                2,
            )
        )
        sp_m = torch.zeros(
            (
                batch_size,
                height // int(sqrt(nr_pixels_per_token)),
                width // int(sqrt(nr_pixels_per_token)),
                3,
            )
        )
        t_m = torch.zeros((batch_size, nr_timesteps, 4))
        st_m = torch.zeros((batch_size, 2))

        # insert ones to check if they are at the right place
        tar_torch[0][0][0] = 1
        tar_torch[1][5][2] = 1
        s_t_x[0][0][0] = 1
        s_t_x[1][1][0] = 1
        sp_x[0][0][0] = 1
        sp_x[1][1][0] = 1
        t_x[0] = 1
        t_x[1] = 1
        st_x[0] = 1
        st_x[1] = 1

        tar_torch = eval.group_targets_per_token(tar_torch)
        enc_torch = eval.group_encodings_per_token(
            model, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
        )

        assert tar_torch.shape == (nr_tokens, nr_pixels_per_token)
        assert 1 in np.nditer(tar_torch[0])
        assert 1 in np.nditer(tar_torch[6])
        assert 1 in np.nditer(enc_torch[0])
        assert 1 in np.nditer(enc_torch[6])
        assert 1 not in np.nditer(enc_torch[1])

        enc_np, tar_np = (
            np.zeros((nr_tokens, model_dim)),
            np.zeros((nr_tokens, nr_pixels_per_token)),
        )

        self.check_remove_void(enc_np, tar_np)
        self.check_reduce_targets_per_token(output_mode="mode")
        self.check_reduce_targets_per_token(output_mode="norm_counts")
"""
