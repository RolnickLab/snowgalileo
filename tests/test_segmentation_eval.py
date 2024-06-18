import unittest
from math import sqrt

import numpy as np
import torch

from src.eval.eval import EvalTask
from src.flexipresto import Encoder


class TestEval(unittest.TestCase):
    def check_remove_void(self, enc_np, tar_np):
        nr_tokens = tar_np.shape[0]

        # insert void labels
        tar_np[0][3] = 19
        tar_np[1][2] = 19

        # insert ones in encodings to check if they are removed
        enc_np[0] = np.ones(enc_np.shape[1])
        enc_np[1] = np.ones(enc_np.shape[1])

        # create void mask as in eval.py
        void_mask = np.any(tar_np == 19, axis=1)
        tar_np = tar_np[~void_mask]
        enc_np = enc_np[~void_mask]

        # resulting arrays should have 2 less tokens
        assert enc_np.shape[0] == nr_tokens - 2
        assert tar_np.shape[0] == nr_tokens - 2
        assert np.all(enc_np == 0)
        assert np.all(enc_np == 0)

    def check_reduce_targets_per_token(self, nr_outputs):
        nr_tokens = 8
        nr_pixels_per_token = 9
        max_class = 8

        eval = EvalTask(patch_size=3, num_outputs=nr_outputs)

        # each token should contain a unique class value
        values = np.arange(0, max_class, dtype=np.uint8)
        tar_np = np.repeat(values, nr_pixels_per_token).reshape(nr_tokens, nr_pixels_per_token)

        tar_np = eval.reduce_targets_per_token(tar_np)

        if nr_outputs == 1:
            assert tar_np.shape == (nr_tokens,)
            assert tar_np[1] == 1
            assert tar_np[7] == 7
        else:
            assert tar_np.shape == (nr_tokens, nr_outputs)
            # each token should contain a unique class value
            assert np.all(np.sum(tar_np, axis=1) == 1)
            # second token should contain class 1
            assert tar_np[1][1] == 1
            assert tar_np[1][~1] == 0

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
        self.check_reduce_targets_per_token(nr_outputs=9)
        self.check_reduce_targets_per_token(nr_outputs=1)
