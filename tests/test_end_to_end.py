import unittest
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.collate_fns import mae_collate_fn
from src.data import Dataset
from src.flexipresto import Encoder, PrestoPixelDecoder
from src.loss import LOSS_TYPES, masked_autoencoder_loss
from src.utils import device

DATA_FOLDER = Path(__file__).parents[1] / "data/tifs"


class TestEndtoEnd(unittest.TestCase):
    def test_end_to_end(self):
        for loss_type in LOSS_TYPES:
            self._test_end_to_end(loss_type)

    def _test_end_to_end(self, loss_type: str):
        embedding_size = 32

        dataset = Dataset(DATA_FOLDER, download=False, h5py_folder=None)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=partial(
                mae_collate_fn,
                patch_sizes=[1, 2, 3, 4, 5, 6, 7, 8],
                shape_time_combinations=[
                    {"size": 4, "timesteps": 12},
                    {"size": 5, "timesteps": 6},
                    {"size": 6, "timesteps": 4},
                    {"size": 7, "timesteps": 3},
                    {"size": 9, "timesteps": 3},
                    {"size": 12, "timesteps": 3},
                ],
                mask_ratio=0.25,
                decoder_unmask_ratio=0.25,
            ),
            pin_memory=True,
        )

        encoder = Encoder(embedding_size=embedding_size, num_heads=1).to(device)
        predictor = PrestoPixelDecoder(
            encoder_embedding_size=embedding_size,
            decoder_embedding_size=embedding_size,
            num_heads=1,
        ).to(device)
        param_groups = [{"params": encoder.parameters()}, {"params": predictor.parameters()}]
        optimizer = torch.optim.AdamW(param_groups, lr=3e-4)  # type: ignore

        # let's just consider one of the augmentations
        for _, (b, _, _) in enumerate(dataloader):
            for x in b:
                if isinstance(x, torch.Tensor):
                    self.assertFalse(torch.isnan(x).any())
            b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
            (
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                expanded_s_t_x,
                expanded_sp_x,
                s_t_m_p,
                sp_m_p,
                t_m_p,
                st_m_p,
                patch_size,
                _,
            ) = b
            # no autocast since its poorly supported on CPU
            (p_s_t, p_sp, p_t, p_st) = predictor(
                *encoder(
                    s_t_x=s_t_x.float(),
                    sp_x=sp_x.float(),
                    t_x=t_x.float(),
                    st_x=st_x.float(),
                    s_t_m=s_t_m.int(),
                    sp_m=sp_m.int(),
                    t_m=t_m.int(),
                    st_m=st_m.int(),
                    months=months.long(),
                    patch_size=patch_size,
                ),
                patch_size=patch_size,
            )
            self.assertFalse(torch.isnan(p_s_t[s_t_m_p == 2]).any())
            self.assertFalse(torch.isnan(p_sp[sp_m_p == 2]).any())
            self.assertFalse(torch.isnan(p_t[t_m_p == 2]).any())
            self.assertFalse(torch.isnan(p_st[st_m_p == 2]).any())

            loss = masked_autoencoder_loss(
                expanded_s_t_x,
                expanded_sp_x,
                t_x,
                st_x,
                p_s_t,
                p_sp,
                p_t,
                p_st,
                s_t_m_p,
                sp_m_p,
                t_m_p,
                st_m_p,
                patch_size=8,
                loss_type=loss_type,
            )
            self.assertFalse(torch.isnan(loss).any())
            loss.backward()
            optimizer.step()
