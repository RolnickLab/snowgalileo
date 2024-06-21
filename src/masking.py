import random
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat

from .data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
SPACE_TIME_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_BANDS_GROUPS_IDX.values()]
).long()
SPACE_BAND_EXPANSION = torch.tensor([len(x) for x in SPACE_BAND_GROUPS_IDX.values()]).long()
TIME_BAND_EXPANSION = torch.tensor([len(x) for x in TIME_BAND_GROUPS_IDX.values()]).long()
STATIC_BAND_EXPANSION = torch.tensor([len(x) for x in STATIC_BAND_GROUPS_IDX.values()]).long()

NON_S2_RGB_BANDS = [
    idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if val != "S2_RGB"
]
S2_RGB_BANDS = [
    idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if val == "S2_RGB"
]

NON_S2_BANDS = [
    idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if "S2" not in val
]
S2_BANDS = [idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if "S2" in val]
NON_S1_BANDS = [
    idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if "S1" not in val
]
S1_BANDS = [idx for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys())) if "S1" in val]
NON_S1_S2_BANDS = [
    idx
    for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys()))
    if (("S1" not in val) and ("S2" not in val))
]
S1_S2_BANDS = [
    idx
    for idx, val in enumerate(list(SPACE_TIME_BANDS_GROUPS_IDX.keys()))
    if (("S1" in val) or ("S2" in val))
]

MASKING_MODES = [None, "s2", "s2rgb", "s1", "s1+s2"]
# we divide the dataloader's batch size by 8 because the
# masking function (batch_subset_mask_presto_8x) will augment
# each instance in the batch 8 times (with different subsetting and
# masking).
MASKING_MULTIPLIER = 4


class MaskedOutput(NamedTuple):
    """
    A mask can take 3 values:
    0: seen by the encoder (i.e. makes the key and value tokens in the decoder)
    1: not seen by the encoder, and ignored by the decoder
    2: not seen by the encoder, and processed by the decoder (the decoder's query values)
    """

    space_time_x: torch.Tensor
    space_x: torch.Tensor
    time_x: torch.Tensor
    static_x: torch.Tensor
    space_time_mask: torch.Tensor
    space_mask: torch.Tensor
    time_mask: torch.Tensor
    static_mask: torch.Tensor
    months: torch.Tensor

    @staticmethod
    def concatenate(x: List["MaskedOutput"]) -> "MaskedOutput":
        return MaskedOutput(
            torch.cat([x_i.space_time_x for x_i in x], 0),
            torch.cat([x_i.space_x for x_i in x], 0),
            torch.cat([x_i.time_x for x_i in x], 0),
            torch.cat([x_i.static_x for x_i in x], 0),
            torch.cat([x_i.space_time_mask for x_i in x], 0),
            torch.cat([x_i.space_mask for x_i in x], 0),
            torch.cat([x_i.time_mask for x_i in x], 0),
            torch.cat([x_i.static_mask for x_i in x], 0),
            torch.cat([x_i.months for x_i in x], 0),
        )


def check_mode_and_return_channels(
    mode: Optional[str],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    assert mode in MASKING_MODES
    if mode is None:
        return None, None
    elif mode == "s2rgb":
        return S2_RGB_BANDS, NON_S2_RGB_BANDS
    elif mode == "s2":
        return S2_BANDS, NON_S2_BANDS
    elif mode == "s1":
        return S1_BANDS, NON_S1_BANDS
    else:
        return S1_S2_BANDS, NON_S1_S2_BANDS


def subset_batch_of_images(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    size: int,
    num_timesteps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert (space_time_x.shape[1] == space_x.shape[1]) & (
        space_time_x.shape[2] == space_x.shape[2]
    )
    assert time_x.shape[1] == space_time_x.shape[3] == months.shape[1]
    possible_h = space_time_x.shape[1] - size
    possible_w = space_time_x.shape[2] - size
    possible_t = space_time_x.shape[3] - num_timesteps
    assert (possible_h >= 0) & (possible_w >= 0) & (possible_t >= 0)

    if possible_h > 0:
        start_h = np.random.choice(possible_h)
    else:
        start_h = possible_h

    if possible_w > 0:
        start_w = np.random.choice(possible_w)
    else:
        start_w = possible_w

    if possible_t > 0:
        start_t = np.random.choice(possible_t)
    else:
        start_t = possible_t

    return (
        space_time_x[
            :,
            start_h : start_h + size,
            start_w : start_w + size,
            start_t : start_t + num_timesteps,
        ],
        space_x[:, start_h : start_h + size, start_w : start_w + size],
        time_x[:, start_t : start_t + num_timesteps],
        static_x,
        months[:, start_t : start_t + num_timesteps],
    )


def batch_subset_mask_presto_augmented(
    s_t_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    image_size: int,
    num_timesteps: int,
) -> MaskedOutput:
    """
    Given an input batch size of x, this function will
    return 8x as many points (e.g. 16 -> 128)
    """
    maskedoutputs: List[MaskedOutput] = []

    for mode in random.sample(MASKING_MODES, k=1):
        maskedoutputs.append(
            batch_mask_time(
                *subset_batch_of_images(
                    s_t_x, sp_x, t_x, st_x, months, size=image_size, num_timesteps=num_timesteps
                ),
                mask_ratio,
                mode=mode,
            )
        )
    for mode in random.sample(MASKING_MODES, k=1):
        maskedoutputs.append(
            batch_mask_space(
                *subset_batch_of_images(
                    s_t_x, sp_x, t_x, st_x, months, size=image_size, num_timesteps=num_timesteps
                ),
                mask_ratio,
                patch_size,
                mode=mode,
            )
        )

    maskedoutputs.append(
        batch_mask_channels(
            *subset_batch_of_images(
                s_t_x, sp_x, t_x, st_x, months, size=image_size, num_timesteps=num_timesteps
            ),
            mask_ratio,
        )
    )
    maskedoutputs.append(
        batch_mask_random(
            *subset_batch_of_images(
                s_t_x, sp_x, t_x, st_x, months, size=image_size, num_timesteps=num_timesteps
            ),
            mask_ratio,
            patch_size,
        )
    )
    return MaskedOutput.concatenate(maskedoutputs)


def batch_mask_time(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    mode: Optional[str] = None,
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timesteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    bands_to_keep, bands_to_mask = check_mode_and_return_channels(mode)
    b, h, w, t, _ = space_time_x.shape
    # if there is only a single timestep, mask it
    num_timesteps_to_mask = int(t * mask_ratio) if t > 1 else 1
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_timesteps = np.concatenate(
        (
            np.ones(num_timesteps_to_mask, dtype=np.int_),
            np.zeros(t - num_timesteps_to_mask, dtype=np.int_),
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
    ).clone()
    if bands_to_mask is None:
        time_mask = repeat(
            b_flat_timesteps_t,
            "b t-> b t c_g",
            c_g=len(TIME_BAND_GROUPS_IDX),
        )
        space_mask = torch.rand(b, device=space_x.device) <= mask_ratio
        if t == 1:
            # can't mask out everything if t == 1, so we make sure the
            # space only mask remains unmasked
            space_mask = space_mask * 0
        space_mask = repeat(space_mask, "b -> b h w c_g", h=h, w=w, c_g=len(SPACE_BAND_GROUPS_IDX))
        static_mask = torch.rand(b, device=static_x.device) <= mask_ratio
        static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX))
    else:
        space_time_mask[:, :, :, :, bands_to_mask] = 1
        if t == 1:
            assert bands_to_keep is not None
            space_time_mask[:, :, :, :, bands_to_keep] = 0
        space_mask = torch.ones((b, h, w, len(SPACE_BAND_GROUPS_IDX))).to(space_x.device)
        time_mask = torch.ones((b, t, len(TIME_BAND_GROUPS_IDX))).to(time_x.device)
        static_mask = torch.ones((b, len(STATIC_BAND_GROUPS_IDX))).to(static_x.device)

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def batch_mask_space(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    mode: Optional[str] = None,
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
    _, bands_to_mask = check_mode_and_return_channels(mode)
    b, h, w, t, _ = space_time_x.shape
    assert (h % patch_size == 0) and (w % patch_size == 0)
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)
    total_patches = h_p * w_p
    num_patches_to_mask = int(total_patches * mask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_patches = np.concatenate(
        (
            np.ones(num_patches_to_mask, dtype=np.int_),
            np.zeros(total_patches - num_patches_to_mask, dtype=np.int_),
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

    if bands_to_mask is None:
        space_mask = torch.from_numpy(
            repeat(
                two_d_mask,
                "b h w -> b h w c_g",
                c_g=len(SPACE_BAND_GROUPS_IDX),
            )
        ).to(space_x.device)

        time_mask = torch.rand(b, device=time_x.device) <= mask_ratio
        time_mask = repeat(time_mask, "b -> b t c_g", t=t, c_g=len(TIME_BAND_GROUPS_IDX))
        static_mask = torch.rand(b, device=static_x.device) <= mask_ratio
        static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX))
    else:
        space_time_mask[:, :, :, :, bands_to_mask] = 1
        # we only want S2 and / or S1, so we mask everything else
        space_mask = torch.ones((b, h, w, len(SPACE_BAND_GROUPS_IDX))).to(space_x.device)
        time_mask = torch.ones((b, t, len(TIME_BAND_GROUPS_IDX))).to(time_x.device)
        static_mask = torch.ones((b, len(STATIC_BAND_GROUPS_IDX))).to(static_x.device)

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def batch_mask_channels(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
):
    """
    Masks out channels. All channels are masked out
    with probability mask_ratio
    """

    def channel_mask(b: int, num_channels: int, mask_ratio: float, device: torch.device):
        if num_channels == 1:
            return (torch.rand(b, device=device) <= mask_ratio).unsqueeze(-1)
        else:
            num_channels_to_mask = int(num_channels * mask_ratio)
            flat_channels = np.concatenate(
                (
                    np.ones(num_channels_to_mask, dtype=np.int_),
                    np.zeros(num_channels - num_channels_to_mask, dtype=np.int_),
                )
            )
            b_flat_channels = repeat(flat_channels, "c -> b c", b=b)
            # hopefully this will allow for reproducibility, since random is seeded
            rng = np.random.default_rng(random.randint(0, 100))
            b_flat_channels_t = torch.from_numpy(rng.permuted(b_flat_channels, axis=1)).to(device)
            return b_flat_channels_t

    b, h, w, t, _ = space_time_x.shape
    space_time_channel_mask = channel_mask(
        b, len(SPACE_TIME_BANDS_GROUPS_IDX), mask_ratio, space_time_x.device
    )
    space_channel_mask = channel_mask(b, len(SPACE_BAND_GROUPS_IDX), mask_ratio, space_x.device)
    time_channel_mask = channel_mask(b, len(TIME_BAND_GROUPS_IDX), mask_ratio, time_x.device)
    static_mask = channel_mask(b, len(STATIC_BAND_GROUPS_IDX), mask_ratio, static_x.device)

    space_time_mask = repeat(space_time_channel_mask, "b c_g -> b h w t c_g", h=h, w=w, t=t)
    space_mask = repeat(space_channel_mask, "b c_g -> b h w c_g", h=h, w=w)
    time_mask = repeat(time_channel_mask, "b c_g -> b t c_g", t=t)

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def batch_mask_random(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    decoder_unmask_ratio: float,
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
    c_sp = len(SPACE_BAND_GROUPS_IDX)
    c_t = len(TIME_BAND_GROUPS_IDX)
    c_st = len(STATIC_BAND_GROUPS_IDX)
    assert (h % patch_size == 0) and (w % patch_size == 0)
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)

    num_space_time_tokens = h_p * w_p * t * c_s_t
    num_space_tokens = h_p * w_p * c_sp
    num_time_tokens = t * c_t
    num_static_tokens = c_st

    total_tokens = num_space_time_tokens + num_space_tokens + num_time_tokens + num_static_tokens
    tokens_the_decoder_will_unmask = int(total_tokens * decoder_unmask_ratio)
    unused_tokens = int(total_tokens * mask_ratio) - tokens_the_decoder_will_unmask
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_tokens = np.concatenate(
        (
            np.ones(unused_tokens, dtype=np.int_),
            np.ones(tokens_the_decoder_will_unmask, dtype=np.int_) * 2,
            np.zeros(
                total_tokens - (tokens_the_decoder_will_unmask + tokens_the_decoder_will_unmask),
                dtype=np.int_,
            ),
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

    space_tokens = b_flat_tokens[:, num_space_time_tokens : -(num_time_tokens + num_static_tokens)]
    space_tokens = rearrange(space_tokens, "b (h w c) -> b h w c", h=h_p, w=w_p, c=c_sp)
    space_mask = torch.from_numpy(
        np.repeat(np.repeat(space_tokens, repeats=patch_size, axis=1), repeats=patch_size, axis=2)
    ).to(space_x.device)

    time_tokens = b_flat_tokens[:, -(num_time_tokens + num_static_tokens) : -num_static_tokens]
    time_mask = torch.from_numpy(rearrange(time_tokens, "b (t c) -> b t c", t=t, c=c_t)).to(
        time_x.device
    )

    static_tokens = b_flat_tokens[:, -num_static_tokens:]
    static_mask = torch.from_numpy(static_tokens).to(static_x.device)

    return MaskedOutput(
        space_time_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )
