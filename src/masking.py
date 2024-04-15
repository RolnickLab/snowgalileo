import random
from collections import namedtuple

import numpy as np
import torch
from einops import rearrange, repeat

from .data.config import NUM_TIMESTEPS
from .data.dataset import SPACE_BAND_GROUPS_IDX, SPACE_TIME_BANDS_GROUPS_IDX, TIME_BAND_GROUPS_IDX

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
SPACE_TIME_BAND_EXPANSION = [len(x) for x in SPACE_TIME_BANDS_GROUPS_IDX.values()]
SPACE_BAND_EXPANSION = [len(x) for x in SPACE_BAND_GROUPS_IDX.values()]
TIME_BAND_GROUPS_IDX = [len(x) for x in TIME_BAND_GROUPS_IDX.values()]


MaskedOutput = namedtuple(
    "MaskedOutput",
    ["space_time_x", "space_x", "time_x", "space_time_mask", "space_mask", "time_mask", "months"],
)


def subset_batch_of_images(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    size: int,
):
    assert (space_time_x.shape[1] == space_x.shape[1]) & (
        space_time_x.shape[2] == space_x.shape[2]
    )
    possible_h = space_time_x.shape[1] - size
    possible_w = space_time_x.shape[2] - size
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
        space_time_x[:, start_h : start_h + size, start_w : start_w + size],
        space_x[:, start_h : start_h + size, start_w : start_w + size],
    )


def batch_mask_presto(
    s_t_x: torch.Tensor,
    s_x: torch.Tensor,
    t_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    time_ratio: float,
    space_ratio: float,
) -> MaskedOutput:
    b = s_t_x.shape[0]
    t_r = int(b * time_ratio)
    s_r = int(b * space_ratio)
    o_t = batch_mask_time(s_t_x[:t_r], s_x[:t_r], t_x[:t_r], months[:t_r], mask_ratio)
    o_s = batch_mask_space(
        s_t_x[t_r : t_r + s_r],
        s_x[t_r : t_r + s_r],
        t_x[t_r : t_r + s_r],
        months[t_r : t_r + s_r],
        mask_ratio,
        patch_size,
    )
    o_r = batch_mask_random(
        s_t_x[t_r + s_r :],
        s_x[t_r + s_r :],
        t_x[t_r + s_r :],
        months[t_r + s_r :],
        mask_ratio,
        patch_size,
    )
    return MaskedOutput(
        torch.cat((o_t[0], o_s[0], o_r[0]), 0),
        torch.cat((o_t[1], o_s[1], o_r[1]), 0),
        torch.cat((o_t[2], o_s[2], o_r[2]), 0),
        torch.cat((o_t[3], o_s[3], o_r[3]), 0),
        torch.cat((o_t[4], o_s[4], o_r[4]), 0),
        torch.cat((o_t[5], o_s[5], o_r[5]), 0),
        torch.cat((o_t[6], o_s[6], o_r[6]), 0),
    )


def batch_mask_time(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timesteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    b, h, w, t, _ = space_time_x.shape
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
        space_time_x.device
    )
    space_time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h,
        w=w,
        c_g=len(SPACE_TIME_BANDS_GROUPS_IDX),
    )
    time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b t c_g",
        c_g=len(TIME_BAND_GROUPS_IDX),
    )

    space_mask = torch.rand(b, device=space_x.device) <= mask_ratio
    space_mask = repeat(space_mask, "b -> b h w s", h=h, w=w, s=len(SPACE_BAND_GROUPS_IDX))

    return MaskedOutput(
        space_time_x, space_x, time_x, space_time_mask, space_mask, time_mask, months
    )


def batch_mask_space(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
):
    """
    Masks out patches (blocks of of pxpxtxBAND_GROUPs).
    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a mask might be:
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    repeated over all dynamic timesteps + channel groups and static channel groups
    Operates over batches where each item in the batch is independently masked
    """
    b, h, w, t, _ = space_time_x.shape
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
    space_time_mask = torch.from_numpy(
        repeat(
            two_d_mask,
            "b h w -> b h w t c_g",
            t=t,
            c_g=len(SPACE_TIME_BANDS_GROUPS_IDX),
        )
    ).to(space_time_x.device)

    space_mask = torch.from_numpy(
        repeat(
            two_d_mask,
            "b h w -> b h w c_g",
            c_g=len(SPACE_BAND_GROUPS_IDX),
        )
    ).to(space_x.device)

    time_mask = torch.rand(b, device=time_x.device) <= mask_ratio
    time_mask = repeat(time_mask, "b -> b t c_g", t=t, c_g=len(TIME_BAND_GROUPS_IDX))

    return MaskedOutput(
        space_time_x, space_x, time_x, space_time_mask, space_mask, time_mask, months
    )


def batch_mask_random(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
):
    """
    Masks out random tokens (blocks of of pxpx1x1).
    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a mask (for one timestep)
    and channel group) might be
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    Operates over batches where each item in the batch is independently masked
    """
    b, h, w, t, _ = space_time_x.shape
    c_s_t = len(SPACE_TIME_BANDS_GROUPS_IDX)
    c_s = len(SPACE_BAND_GROUPS_IDX)
    c_t = len(TIME_BAND_GROUPS_IDX)
    assert (h % patch_size == 0) and (w % patch_size == 0)
    assert t == NUM_TIMESTEPS
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)

    num_space_time_tokens = h_p * w_p * t * c_s_t
    num_space_tokens = h_p * w_p * c_s
    num_time_tokens = t * c_t

    total_tokens = num_space_time_tokens + num_space_tokens + num_time_tokens
    num_tokens_to_mask = int(total_tokens * mask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_tokens = np.concatenate(
        (
            np.ones(num_tokens_to_mask),
            np.zeros(total_tokens - num_tokens_to_mask),
        )
    )
    b_flat_tokens = repeat(flat_tokens, "t -> b t", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_tokens = rng.permuted(b_flat_tokens, axis=1)

    s_t_tokens = b_flat_tokens[:, :num_space_time_tokens]
    s_t_tokens = rearrange(s_t_tokens, "b (h w t c) -> b h w t c", h=h_p, w=w_p, t=t, c=c_s_t)
    space_time_mask = torch.from_numpy(
        np.repeat(np.repeat(s_t_tokens, repeats=patch_size, axis=1), repeats=patch_size, axis=2)
    ).to(space_time_x.device)

    space_tokens = b_flat_tokens[:, num_space_time_tokens:-num_time_tokens]
    space_tokens = rearrange(space_tokens, "b (h w c) -> b h w c", h=h_p, w=w_p, c=c_s)
    space_mask = torch.from_numpy(
        np.repeat(np.repeat(space_tokens, repeats=patch_size, axis=1), repeats=patch_size, axis=2)
    ).to(space_x.device)

    time_tokens = b_flat_tokens[:, -num_time_tokens:]
    time_mask = torch.from_numpy(rearrange(time_tokens, "b (t c) -> b t c", t=t, c=c_t)).to(
        time_x.device
    )

    return MaskedOutput(
        space_time_x, space_x, time_x, space_time_mask, space_mask, time_mask, months
    )
