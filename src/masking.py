import random
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat

from .data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from .data_augmentation import Augmentation

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
SPACE_TIME_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_BANDS_GROUPS_IDX.values()]
).long()
SPACE_BAND_EXPANSION = torch.tensor([len(x) for x in SPACE_BAND_GROUPS_IDX.values()]).long()
TIME_BAND_EXPANSION = torch.tensor([len(x) for x in TIME_BAND_GROUPS_IDX.values()]).long()
STATIC_BAND_EXPANSION = torch.tensor([len(x) for x in STATIC_BAND_GROUPS_IDX.values()]).long()


STR2DICT = OrderedDict(
    {
        "space_time": SPACE_TIME_BANDS_GROUPS_IDX,
        "space": SPACE_BAND_GROUPS_IDX,
        "time": TIME_BAND_GROUPS_IDX,
        "static": STATIC_BAND_GROUPS_IDX,
    }
)
MASKING_MODES: List[Union[str, Tuple[str, str]]] = [
    ("space", "SRTM"),
    ("space", "DW"),
    ("space", "WC"),
    ("space_time", "NDVI"),
    ("space_time", "S1"),
    ("space_time", "S2_RGB"),
    ("space_time", "S2_SWIR"),
    ("space_time", "S2_Red_Edge"),
    ("space_time", "S2_NIR_10m"),
    ("space_time", "S2_NIR_20m"),
    ("time", "ERA5"),
    ("time", "TC"),
    ("time", "VIIRS"),
    ("static", "LS"),
    ("static", "location"),
    ("static", "DW_static"),
    ("static", "WC_static"),
]

MASKING_MODES_COARSE = ["space", "space_time", "time", "static"]

MAX_MASKING_STRATEGIES = 6
NUM_RECON_OBJS = 2


class MaskingFunctions(Enum):
    SPACE = 0
    TIME = 1
    RANDOM = 2


def return_masked_unmasked_bands(
    bands: List[str], band_groups: Dict[str, List]
) -> Tuple[List[int], List[int]]:
    def in_masked_bands(x):
        for b in bands:
            if b in x:
                return True
        return False

    return [idx for idx, val in enumerate(band_groups.keys()) if in_masked_bands(val)], [
        idx for idx, val in enumerate(band_groups.keys()) if not in_masked_bands(val)
    ]


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


def weighted_sample_without_replacement(population, weights, k, rng=random):
    if len(population) != len(weights):
        raise ValueError("Population and weights must have the same length")

    non_zero_indices = [i for i, w in enumerate(weights) if w > 0]
    if len(non_zero_indices) < k:
        raise ValueError("Not enough non-zero weights to sample k items")

    non_zero_population = [population[i] for i in non_zero_indices]
    non_zero_weights = [weights[i] for i in non_zero_indices]

    v = [rng.random() ** (1 / w) for w in non_zero_weights]
    order = sorted(range(len(non_zero_population)), key=lambda i: v[i])
    return [non_zero_population[i] for i in order[-k:]]


def check_modes_for_conflicts(
    modes: List[Tuple[str, str]], unmasking_modes: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    assert len(unmasking_modes) == 1
    for u_mode in unmasking_modes:
        assert u_mode in MASKING_MODES
        if u_mode in modes:
            modes.remove(u_mode)
        else:
            continue
    assert len(modes) >= 1
    assert len(unmasking_modes) >= 1
    return modes, unmasking_modes


def check_mode_and_return_channels(unmasking_modes: List[Tuple[str, str]]):
    outputs = []
    for data_type in STR2DICT.keys():
        relevant_bands = [x[1] for x in unmasking_modes if x[0] == data_type]
        if len(relevant_bands) > 0:
            outputs.append(return_masked_unmasked_bands(relevant_bands, STR2DICT[data_type]))
        else:
            outputs.append(([], []))
    return outputs


def round_school(x: float) -> float:
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def batch_subset_mask_presto(
    s_t_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    decoder_unmask_ratio: float,
    patch_size: int,
    image_size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
    masking_probabilities: List[float],
    unmasking_probabilities: List[float],
    masking_function: MaskingFunctions,
) -> Tuple[MaskedOutput, Optional[Dict]]:
    assert len(masking_probabilities) == len(unmasking_probabilities) == len(MASKING_MODES)

    conditioner_inputs: Optional[Dict] = {
        "output_channels": torch.zeros(len(MASKING_MODES_COARSE)).to(s_t_x.device),
    }

    if masking_function.value < 2:
        f: Callable = batch_mask_space if masking_function.value == 1 else batch_mask_time
        unmasking_mode = random.choice(MASKING_MODES_COARSE)
        possible_unmasking_modes, selected_unmasking_probs = [], []
        for idx, mode in enumerate(MASKING_MODES):
            if mode[0] == unmasking_mode:
                possible_unmasking_modes.append(mode)
                selected_unmasking_probs.append(unmasking_probabilities[idx])
        num_unmasking_modes = random.choice(list(range(1, len(possible_unmasking_modes) + 1)))
        num_masking_modes = num_unmasking_modes + 1
        masking_modes = weighted_sample_without_replacement(
            MASKING_MODES, weights=masking_probabilities, k=num_masking_modes
        )
        unmasking_modes = weighted_sample_without_replacement(
            possible_unmasking_modes, weights=selected_unmasking_probs, k=num_masking_modes
        )
        masking_modes, unmasking_modes = check_modes_for_conflicts(masking_modes, unmasking_modes)
        masked_output = f(
            *subset_and_augment_batch_of_images(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            mask_ratio=mask_ratio,
            decoder_unmask_ratio=decoder_unmask_ratio,
            patch_size=patch_size,
            mode=masking_modes,
            decoder_mode=unmasking_modes,
        )
        assert len(unmasking_modes) == 1
        assert conditioner_inputs is not None
        conditioner_inputs["output_channels"][
            MASKING_MODES_COARSE.index(unmasking_modes[0][0])
        ] = 1  # type: ignore

    elif masking_function.value == 2:
        # 2 is random
        masked_output = batch_mask_random(
            *subset_and_augment_batch_of_images(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                months,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            mask_ratio=mask_ratio,
            decoder_unmask_ratio=decoder_unmask_ratio,
            patch_size=patch_size,
        )
        conditioner_inputs = None

    else:
        raise AssertionError(f"Unexpected strategy {masking_function}")

    return masked_output, conditioner_inputs


def subset_and_augment_batch_of_images(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
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

    # do augmentations, if enabled
    space_time_x = space_time_x[
        :,
        start_h : start_h + size,
        start_w : start_w + size,
        start_t : start_t + num_timesteps,
    ]
    space_x = space_x[:, start_h : start_h + size, start_w : start_w + size]
    time_x = time_x[:, start_t : start_t + num_timesteps]
    months = months[:, start_t : start_t + num_timesteps]

    if augmentation_strategies is not None:
        return Augmentation(augmentation_strategies).apply(
            space_time_x, space_x, time_x, static_x, months
        )
    return space_time_x, space_x, time_x, static_x, months


def _random_mask_for_b(
    b: int, device: torch.device, mask_ratio: float, decoder_unmask_ratio: float
) -> torch.Tensor:
    mask = torch.rand(b, device=device)
    total_masked_tokens_ratio = mask_ratio + decoder_unmask_ratio
    mask[mask >= total_masked_tokens_ratio] = 0
    mask[mask <= decoder_unmask_ratio] = 2
    # all the rest is ignored by both the encoder and decoder
    mask[(mask != 0) | (mask != 2)] = 1
    return mask


def batch_mask_time(
    space_time_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    mask_ratio: float,
    decoder_unmask_ratio: float,
    patch_size: int,
    decoder_mode: List[Tuple[str, str]],
    mode: List[Tuple[str, str]],
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timeteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    b, h, w, t, _ = space_time_x.shape
    assert t >= 3

    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)
    # if there is only a single timestep, decode it
    num_timesteps_to_decode = max(int(t * decoder_unmask_ratio), 1)
    num_timesteps_to_encode = max(int(t * mask_ratio), 1)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_timesteps = np.concatenate(
        (
            np.ones(t - (num_timesteps_to_decode + num_timesteps_to_encode), dtype=np.int_),
            np.ones(num_timesteps_to_decode, dtype=np.int_) * 2,
            np.zeros(num_timesteps_to_encode, dtype=np.int_),
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
    # make the mask as if bands_to_mask and bands_to_decode both = None
    time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b t c_g",
        c_g=len(TIME_BAND_GROUPS_IDX),
    ).clone()
    space_mask = _random_mask_for_b(b, space_x.device, mask_ratio, decoder_unmask_ratio)
    space_mask = repeat(
        space_mask, "b -> b h w c_g", h=h, w=w, c_g=len(SPACE_BAND_GROUPS_IDX)
    ).clone()
    static_mask = _random_mask_for_b(b, static_x.device, mask_ratio, decoder_unmask_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()
    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in time data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        space_mask = torch.clamp(space_mask, min=1)

        s_t_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_e[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_e[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], min=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, min=1)

        if len(s_e[0]) > 0:
            s_bands_to_encode = s_e[0]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_encode] = 0

        if len(t_e[0]) > 0:
            t_bands_to_mask = t_e[1]
            time_mask[:, :, t_bands_to_mask] = torch.clamp(time_mask[:, :, t_bands_to_mask], min=1)
        else:
            time_mask = torch.clamp(time_mask, min=1)

        if len(st_e[0]) > 0:
            st_bands_to_encode = st_e[0]
            static_mask[:, st_bands_to_encode] = 0

    if max([len(x[0]) for x in bands_to_decode]) >= 1:  # decoder mode != random
        # for static in time data,
        # ignore all previous calculations about what should be decoded
        static_mask = torch.clamp(static_mask, max=1)
        space_mask = torch.clamp(space_mask, max=1)

        s_t_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_d[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_d[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], max=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, max=1)

        if len(s_d[0]) > 0:
            s_bands_to_decode = s_d[0]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_decode] = 2

        if len(t_d[0]) > 0:
            t_bands_to_mask = t_d[1]
            time_mask[:, :, t_bands_to_mask] = torch.clamp(time_mask[:, :, t_bands_to_mask], max=1)
        else:
            time_mask = torch.clamp(time_mask, max=1)

        if len(st_d[0]) > 0:
            st_bands_to_decode = st_d[0]
            static_mask[:, st_bands_to_decode] = 2

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
    patch_size: int,
    mask_ratio: float,
    decoder_unmask_ratio: float,
    mode: List[Tuple[str, str]],
    decoder_mode: List[Tuple[str, str]],
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
    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)
    b, h, w, t, _ = space_time_x.shape
    assert (h % patch_size == 0) and (w % patch_size == 0)
    h_p = int(h / patch_size)
    w_p = int(w / patch_size)
    total_patches = h_p * w_p
    num_patches_to_mask = int(total_patches * mask_ratio)
    num_patches_to_decode = int(total_patches * decoder_unmask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_patches = np.concatenate(
        (
            np.ones(num_patches_to_mask, dtype=np.int_),
            np.ones(num_patches_to_decode, dtype=np.int_) * 2,
            np.zeros(total_patches - (num_patches_to_mask + num_patches_to_decode), dtype=np.int_),
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
    space_time_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask,
                "b h w -> b h w t c_g",
                t=t,
                c_g=len(SPACE_TIME_BANDS_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_time_x.device)
    )

    space_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask,
                "b h w -> b h w c_g",
                c_g=len(SPACE_BAND_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_x.device)
    )
    time_mask = _random_mask_for_b(b, time_x.device, mask_ratio, decoder_unmask_ratio)
    time_mask = repeat(time_mask, "b -> b t c_g", t=t, c_g=len(TIME_BAND_GROUPS_IDX)).clone()
    static_mask = _random_mask_for_b(b, static_x.device, mask_ratio, decoder_unmask_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()

    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in space data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        time_mask = torch.clamp(time_mask, min=1)

        s_t_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_e[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_e[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], min=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, min=1)

        if len(s_e[0]) > 0:
            s_bands_to_mask = s_e[1]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_mask] = torch.clamp(
                space_mask[:, :, :, s_bands_to_mask], min=1
            )
        else:
            space_mask = torch.clamp(space_mask, min=1)

        if len(t_e[0]) > 0:
            t_bands_to_encode = t_e[0]
            time_mask[:, :, t_bands_to_encode] = 0

        if len(st_e[0]) > 0:
            st_bands_to_encode = st_e[0]
            static_mask[:, st_bands_to_encode] = 0

    if max([len(x[0]) for x in bands_to_decode]) >= 1:  # decoder mode != random
        # for static in space data,
        # ignore all previous calculations about what should be decoded
        static_mask = torch.clamp(static_mask, max=1)
        time_mask = torch.clamp(time_mask, max=1)

        s_t_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_d[0]) > 0:
            # there are space time bands to decode
            s_t_bands_to_mask = s_t_d[1]
            space_time_mask[:, :, :, :, s_t_bands_to_mask] = torch.clamp(
                space_time_mask[:, :, :, :, s_t_bands_to_mask], max=1
            )
        else:
            space_time_mask = torch.clamp(space_time_mask, max=1)

        if len(s_d[0]) > 0:
            s_bands_to_mask = s_d[1]
            # there are space bands to mask
            space_mask[:, :, :, s_bands_to_mask] = torch.clamp(
                space_mask[:, :, :, s_bands_to_mask], max=1
            )
        else:
            space_mask = torch.clamp(space_mask, max=1)

        if len(t_d[0]) > 0:
            t_bands_to_decode = t_d[0]
            time_mask[:, :, t_bands_to_decode] = 2

        if len(st_d[0]) > 0:
            st_bands_to_decode = st_d[0]
            static_mask[:, st_bands_to_decode] = 2

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
    unused_tokens = int(total_tokens * mask_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_tokens = np.concatenate(
        (
            np.ones(unused_tokens, dtype=np.int_),
            np.ones(tokens_the_decoder_will_unmask, dtype=np.int_) * 2,
            np.zeros(
                total_tokens - (unused_tokens + tokens_the_decoder_will_unmask),
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
