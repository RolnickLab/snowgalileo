import random
from collections import namedtuple

import numpy as np
import torch
from einops import rearrange, repeat

from .data.config import NUM_TIMESTEPS
from .data.dataset import DYNAMIC_BANDS_GROUPS_IDX, STATIC_BAND_GROUPS_IDX

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
DYNAMIC_BAND_EXPANSION = [len(x) for x in DYNAMIC_BANDS_GROUPS_IDX.values()]
STATIC_BAND_EXPANSION = [len(x) for x in STATIC_BAND_GROUPS_IDX.values()]


MaskedOutput = namedtuple(
    "MaskedOutput", ["dynamic_x", "static_x", "dynamic_mask", "static_mask", "months"]
)


def subset_batch_of_images(
    dynamic_x: torch.Tensor,
    static_x: torch.Tensor,
    size: int,
):
    assert (dynamic_x.shape[1] == static_x.shape[1]) & (dynamic_x.shape[2] == static_x.shape[2])
    possible_h = dynamic_x.shape[1] - size
    possible_w = dynamic_x.shape[2] - size
    assert (possible_h >= 0) & (possible_w >= 0)

    if possible_h > 0:
        start_h = np.random.choice(possible_h)
    else:
        start_h = possible_h

    if possible_w > 0:
        start_w = np.random.choice(possible_w)
    else:
        start_w = possible_w
    return (
        dynamic_x[:, start_h : start_h + size, start_w : start_w + size],
        static_x[:, start_h : start_h + size, start_w : start_w + size],
    )


def batch_mask_presto(
    dynamic_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    time_ratio: float,
) -> MaskedOutput:
    b = dynamic_x.shape[0]
    s = int(b * time_ratio)
    d_x_t, s_x_t, d_m_t, s_m_t, m_t = batch_mask_time(
        dynamic_x[:s], static_x[:s], months[:s], mask_ratio
    )
    d_x_s, s_x_s, d_m_s, s_m_s, m_s = batch_mask_space(
        dynamic_x[s:], static_x[s:], months[s:], mask_ratio, patch_size
    )

    return MaskedOutput(
        torch.cat((d_x_t, d_x_s), 0),
        torch.cat((s_x_t, s_x_s), 0),
        torch.cat((d_m_t, d_m_s), 0),
        torch.cat((s_m_t, s_m_s), 0),
        torch.cat((m_t, m_s), 0),
    )


def batch_mask_time(
    dynamic_x: torch.Tensor, static_x: torch.Tensor, months: torch.Tensor, mask_ratio: float
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timesteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    b, h, w, t, _ = dynamic_x.shape
    assert t == NUM_TIMESTEPS
    num_timesteps_to_mask = int(t * mask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_timesteps = np.concatenate(
        (
            np.ones(num_timesteps_to_mask),
            np.zeros(t - num_timesteps_to_mask),
        )
    )
    b_flat_timesteps = repeat(flat_timesteps, "t -> b t", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_timesteps_t = torch.from_numpy(rng.permuted(b_flat_timesteps, axis=1)).to(
        dynamic_x.device
    )
    dynamic_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h,
        w=w,
        c_g=len(DYNAMIC_BANDS_GROUPS_IDX),
    )

    static_mask = torch.rand(b, device=static_x.device) <= mask_ratio
    static_mask = repeat(static_mask, "b -> b h w s", h=h, w=w, s=len(STATIC_BAND_GROUPS_IDX))

    return MaskedOutput(dynamic_x, static_x, dynamic_mask, static_mask, months)


def batch_mask_space(
    dynamic_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timesteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    b, h, w, t, _ = dynamic_x.shape
    assert (h % patch_size == 0) and (w % patch_size == 0)
    assert t == NUM_TIMESTEPS
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)
    total_patches = h_p * w_p
    num_patches_to_mask = int(total_patches * mask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_patches = np.concatenate(
        (
            np.ones(num_patches_to_mask),
            np.zeros(total_patches - num_patches_to_mask),
        )
    )
    b_flat_patches = repeat(flat_patches, "p -> b p", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_patches = rng.permuted(b_flat_patches, axis=1)
    two_d_patch_mask = rearrange(b_flat_patches, "b (h w) -> b h w", h=h_p, w=w_p)
    two_d_mask = np.repeat(
        np.repeat(two_d_patch_mask, repeats=patch_size, axis=1), repeats=patch_size, axis=2
    )
    dynamic_mask = torch.from_numpy(
        repeat(
            two_d_mask,
            "b h w -> b h w t c_g",
            t=t,
            c_g=len(DYNAMIC_BANDS_GROUPS_IDX),
        )
    ).to(dynamic_x.device)

    static_mask = torch.from_numpy(
        repeat(
            two_d_mask,
            "b h w -> b h w c_g",
            c_g=len(STATIC_BAND_GROUPS_IDX),
        )
    ).to(static_x.device)

    return MaskedOutput(dynamic_x, static_x, dynamic_mask, static_mask, months)
