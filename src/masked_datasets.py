import random
from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
from einops import repeat

from .config import (
    CROMA_INPUT_SIZE,
    NUM_TIMESTEPS,
    NUM_VIT_PATCHES_PER_CROMA_DIM,
    VIT_PATCH_SIZE,
)
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
    num_timesteps: int
):
    assert (dynamic_x.shape[1] == static_x.shape[1]) & (dynamic_x.shape[2] == static_x.shape[2])
    possible_h = dynamic_x.shape[1] - size
    possible_w = dynamic_x.shape[2] - size
    possible_t = dynamic_x.shape[3] - num_timesteps
    assert possible_t >= 0
    assert (possible_h >= 0) & (possible_w >= 0)

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
        dynamic_x[:, start_h : start_h + size, start_w : start_w + size, start_t + num_timesteps],
        static_x[:, start_h : start_h + size, start_w : start_w + size, start_t + num_timesteps],
    )


def subset_image(
    dynamic_input: np.ndarray,
    static_input: np.ndarray,
    months: np.ndarray,
    size: int,
    num_timesteps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    dynamic_input: array of shape [H, W, T, D]
    static_input: array of shape [H, W, D]

    size must be greater or equal to H & W
    """
    assert (dynamic_input.shape[0] == static_input.shape[0]) & (
        dynamic_input.shape[1] == static_input.shape[1]
    )
    possible_h = dynamic_input.shape[0] - size
    possible_w = dynamic_input.shape[1] - size
    assert (possible_h >= 0) & (possible_w >= 0)
    possible_t = dynamic_input.shape[2] - num_timesteps
    assert possible_t >= 0

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
        dynamic_input[
            start_h : start_h + size, start_w : start_w + size, start_t : start_t + num_timesteps
        ],
        static_input[start_h : start_h + size, start_w : start_w + size],
        months[start_t : start_t + num_timesteps],
    )


def mask_by_croma_spatial_blocks(
    dynamic_input: np.ndarray, static_input: np.ndarray, months: np.ndarray, mask_ratio: float
) -> MaskedOutput:
    """
    Given a H >= CROMA_INPUT_SIZE, W >= CROMA_INPUT_SIZE input:
    1. Crops to CROMA_INPUT_SIZExCROMA_INPUT_SIZE
    2. Masks out blocks of VIT_PATCH_SIZExVIT_PATCH_SIZExTimestepsxBands.
        e.g. if CROMA_INPUT_SIZE=4 and VIT_PATCH_SIZE=2 and mask_ratio=0.25,
        then a mask might look like
        [0 0 1 1]
        [0 0 1 1]
        [0 0 0 0]
        [0 0 0 0]
        Where the top VIT_PATCH (2x2 pixels) is masked out.
    3. This mask is then repeated along the channel and time dimensions to match
       the dynamic and static input shapes
    """
    assert mask_ratio % (1 / ((CROMA_INPUT_SIZE / VIT_PATCH_SIZE) ** 2)) == 0
    dynamic_input, static_input, months = subset_image(
        dynamic_input, static_input, months, CROMA_INPUT_SIZE, NUM_TIMESTEPS
    )
    # To begin with, we compute a flat "mask" of patches
    num_masked_patches = int((NUM_VIT_PATCHES_PER_CROMA_DIM**2) * mask_ratio)
    flat_spatial_mask = np.concatenate(
        (
            np.ones(num_masked_patches),
            np.zeros(NUM_VIT_PATCHES_PER_CROMA_DIM**2 - num_masked_patches),
        )
    )
    np.random.shuffle(flat_spatial_mask)
    spatial_mask = flat_spatial_mask.reshape(
        NUM_VIT_PATCHES_PER_CROMA_DIM, NUM_VIT_PATCHES_PER_CROMA_DIM
    )
    # then we go from CROMA token space (16x16 pixels) back to pixel space
    pixel_spatial_mask = np.repeat(
        np.repeat(spatial_mask, repeats=VIT_PATCH_SIZE, axis=0), repeats=VIT_PATCH_SIZE, axis=1
    )
    # expand the temporal and band dims so they match the dynamic and static input shapes
    dynamic_mask = repeat(
        pixel_spatial_mask,
        "h w -> h w t c",
        t=dynamic_input.shape[2],
        c=len(DYNAMIC_BANDS_GROUPS_IDX),
    )
    static_mask = repeat(pixel_spatial_mask, "h w -> h w c", c=len(STATIC_BAND_GROUPS_IDX))

    return MaskedOutput(dynamic_input, static_input, dynamic_mask, static_mask, months)


def mask_by_croma_blocks_random(
    dynamic_input: np.ndarray, static_input: np.ndarray, months: np.ndarray, mask_ratio: float
) -> MaskedOutput:
    """
    Given a H >= CROMA_INPUT_SIZE, W >= CROMA_INPUT_SIZE input:
    1. Crops to CROMA_INPUT_SIZExCROMA_INPUT_SIZE
    2. Masks out blocks of VIT_PATCH_SIZExVIT_PATCH_SIZEx1xBAND_GROUP.
        e.g. if CROMA_INPUT_SIZE=4 and VIT_PATCH_SIZE=2 and mask_ratio=0.25,
        then a mask might look like
        [0 0 1 1]
        [0 0 1 1]
        [0 0 0 0]
        [0 0 0 0]
        Where the top VIT_PATCH (2x2 pixels) is masked out.
    3. This mask is not applied to each mask and band group; its randomly
        applied along both of these dimensions
    """
    dynamic_input, static_input, months = subset_image(
        dynamic_input, static_input, months, CROMA_INPUT_SIZE, NUM_TIMESTEPS
    )
    num_timesteps = dynamic_input.shape[2]
    num_dynamic_tokens = (
        num_timesteps * (NUM_VIT_PATCHES_PER_CROMA_DIM**2) * len(DYNAMIC_BANDS_GROUPS_IDX)
    )
    num_static_tokens = (NUM_VIT_PATCHES_PER_CROMA_DIM**2) * len(STATIC_BAND_GROUPS_IDX)
    num_tokens_to_mask = int(mask_ratio * (num_dynamic_tokens + num_static_tokens))

    flat_spatial_mask = np.concatenate(
        (
            np.ones(num_tokens_to_mask),
            np.zeros(num_dynamic_tokens + num_static_tokens - num_tokens_to_mask),
        )
    )
    np.random.shuffle(flat_spatial_mask)
    static_mask = flat_spatial_mask[:num_static_tokens]
    dynamic_mask = flat_spatial_mask[num_static_tokens:]

    static_mask = static_mask.reshape(
        (NUM_VIT_PATCHES_PER_CROMA_DIM, NUM_VIT_PATCHES_PER_CROMA_DIM, len(STATIC_BAND_GROUPS_IDX))
    )
    dynamic_mask = dynamic_mask.reshape(
        (
            NUM_VIT_PATCHES_PER_CROMA_DIM,
            NUM_VIT_PATCHES_PER_CROMA_DIM,
            num_timesteps,
            len(DYNAMIC_BANDS_GROUPS_IDX),
        )
    )

    # then we go from CROMA token space (16x16 pixels) back to pixel space
    static_pixel_spatial_mask = np.repeat(
        np.repeat(static_mask, repeats=VIT_PATCH_SIZE, axis=0), repeats=VIT_PATCH_SIZE, axis=1
    )
    dynamic_pixel_spatial_mask = np.repeat(
        np.repeat(dynamic_mask, repeats=VIT_PATCH_SIZE, axis=0), repeats=VIT_PATCH_SIZE, axis=1
    )

    return MaskedOutput(
        dynamic_input, static_input, dynamic_pixel_spatial_mask, static_pixel_spatial_mask, months
    )


def batch_mask_presto(
    dynamic_x: torch.Tensor, static_x: torch.Tensor, months: torch.Tensor, mask_ratio: float
) -> MaskedOutput:
    # for now, we only have one implemented but spatial is on its way
    return batch_mask_presto_time(dynamic_x, static_x, months, mask_ratio)


def batch_mask_presto_time(
    dynamic_x: torch.Tensor, static_x: torch.Tensor, months: torch.Tensor, mask_ratio: float
):
    """
    Given a B > 1, H >= PRESTO_INPUT_SIZE, W >= PRESTO_INPUT_SIZE input:
    1. Masks out blocks of PRESTO_INPUT_SIZExPRESTO_INPUT_SIZEx1xBAND_GROUPs.
        e.g. if PRESTO_INPUT_SIZE=4 and mask_ratio=0.25, then 1/4 of the timesteps
        (and the static channel groups, with 1/4 probability) will be masked out
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
    b_flat_timesteps = torch.from_numpy(rng.permuted(b_flat_timesteps, axis=1)).to(
        dynamic_x.device
    )
    dynamic_mask = repeat(
        b_flat_timesteps,
        "b t-> b h w t c_g",
        h=h,
        w=w,
        c_g=len(DYNAMIC_BANDS_GROUPS_IDX),
    )

    static_mask = torch.rand(b, device=static_x.device) <= mask_ratio
    static_mask = repeat(static_mask, "b -> b h w s", h=h, w=w, s=len(STATIC_BAND_GROUPS_IDX))

    return MaskedOutput(dynamic_x, static_x, dynamic_mask, static_mask, months)
