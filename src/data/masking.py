from collections import namedtuple
from enum import Enum
from typing import Tuple

import numpy as np
from einops import repeat

from .config import CROMA_INPUT_SIZE, VIT_PATCH_SIZE


class MaskingStrategy(Enum):
    CROMA_TO_PRESTO = 0
    PRESTO_TO_CROMA = 1
    CROMA_TO_CROMA = 2
    PRESTO_TO_PRESTO = 3


MaskedOutput = namedtuple(
    "MaskedOutput", ["dynamic_input", "static_input", "dynamic_mask", "static_mask"]
)


def subset_image(
    dynamic_input: np.ndarray, static_input: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert (dynamic_input.shape[0] == static_input.shape[0]) & (
        dynamic_input.shape[1] == static_input.shape[1]
    )
    possible_h = dynamic_input.shape[0] - size
    possible_w = dynamic_input.shape[1] - size
    assert (possible_h >= 0) & (possible_w >= 0)

    if possible_h > 0:
        start_h = np.random.choice(possible_h)
    else:
        start_h = possible_h

    if possible_w > 0:
        start_w = np.random.choice(possible_w)
    else:
        start_w = possible_w

    return dynamic_input[start_h : start_h + size, start_w : start_w + size], static_input[
        start_h : start_h + size, start_w : start_w + size
    ]


def mask_by_croma_blocks(
    dynamic_input: np.ndarray, static_input: np.ndarray, mask_ratio: float
) -> MaskedOutput:
    """
    Given a >CROMA_INPUT_SIZE>CROMA_INPUT_SIZE input:
    1. Crops to CROMA_INPUT_SIZExCROMA_INPUT_SIZE
    2. Masks out blocks of VIT_PATCH_SIZExVIT_PATCH_SIZE.
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
    num_patches_per_dim = int(CROMA_INPUT_SIZE / VIT_PATCH_SIZE)
    dynamic_input, static_input = subset_image(dynamic_input, static_input, CROMA_INPUT_SIZE)
    # for the Presto to CROMA case, we will just remove blocks.
    # To begin with, we compute a flat "mask" of patches
    num_masked_patches = int((num_patches_per_dim**2) * mask_ratio)
    flat_spatial_mask = np.concatenate(
        (np.ones(num_masked_patches), np.zeros(num_patches_per_dim**2 - num_masked_patches))
    )
    np.random.shuffle(flat_spatial_mask)
    spatial_mask = np.reshape(flat_spatial_mask, (num_patches_per_dim, num_patches_per_dim))
    # then we go from CROMA token space (16x16 pixels) back to pixel space
    pixel_spatial_mask = np.repeat(
        np.repeat(spatial_mask, repeats=VIT_PATCH_SIZE, axis=0), repeats=VIT_PATCH_SIZE, axis=1
    )
    # expand the temporal and band dims so they match the dynamic and static input shapes
    dynamic_mask = repeat(
        pixel_spatial_mask, "h w -> h w t c", t=dynamic_input.shape[2], c=dynamic_input.shape[3]
    )
    static_mask = repeat(pixel_spatial_mask, "h w -> h w c", c=static_input.shape[2])

    return MaskedOutput(dynamic_input, static_input, dynamic_mask, static_mask)
