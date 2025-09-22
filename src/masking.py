import random
from collections import OrderedDict
from enum import Enum
from itertools import chain, combinations, product
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat

from src.data.earthengine.eo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
)
from src.data_augmentation import Augmentation

# This is to allow a quick expansion of the mask from
# group-channel space into real-channel space
SPACE_TIME_HIGH_RES_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.values()]
).long()
SPACE_TIME_MED_RES_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.values()]
).long()
SPACE_TIME_LOW_RES_BAND_EXPANSION = torch.tensor(
    [len(x) for x in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.values()]
).long()
SPACE_BAND_EXPANSION = torch.tensor([len(x) for x in SPACE_BAND_GROUPS_IDX.values()]).long()
TIME_BAND_EXPANSION = torch.tensor([len(x) for x in TIME_BANDS_GROUPS_IDX.values()]).long()
STATIC_BAND_EXPANSION = torch.tensor([len(x) for x in STATIC_BAND_GROUPS_IDX.values()]).long()

STR2DICT = OrderedDict(
    {
        "space_time_high_res": SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
        "space_time_med_res": SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
        "space_time_low_res": SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
        "space": SPACE_BAND_GROUPS_IDX,
        "time": TIME_BANDS_GROUPS_IDX,
        "static": STATIC_BAND_GROUPS_IDX,
    }
)

REVERSED_STR2DICT = {}
for key, values in STR2DICT.items():
    for v in values:
        REVERSED_STR2DICT[v] = key

SHAPES = list(STR2DICT.keys())
MASKING_MODES: List[Tuple[str, str]] = [
    ("space", "DEM"),
    ("space", "WC"),
    ("space_time_high_res", "S1"),
    ("space_time_high_res", "S2_RGB"),
    ("space_time_high_res", "S2_NIR"),
    ("space_time_high_res", "S2_SWIR"),
    ("space_time_high_res", "L_RGB"),
    ("space_time_high_res", "L_NIR"),
    ("space_time_high_res", "L_SWIR"),
    ("space_time_med_res", "S3_NIR"),
    ("space_time_low_res", "MODIS_RGB"),
    ("space_time_low_res", "MODIS_NIR"),
    ("space_time_low_res", "MODIS_SWIR"),
    ("space_time_low_res", "VIIRS_RGB_FINE"),
    ("space_time_low_res", "VIIRS_VNIR_FINE"),
    ("space_time_low_res", "NDSI"),
    ("space_time_low_res", "NDVI"),
    ("time", "VIIRS_RGB_COARSE"),
    ("time", "VIIRS_VNIR_COARSE"),
    ("time", "VIIRS_SWIR_COARSE"),
    ("time", "ERA5"),
    ("static", "location"),
]

UNMASKING_CHANNEL_GROUPS: List[Tuple[str, str]] = MASKING_MODES

MAX_MASKING_STRATEGIES = 6
NUM_RECON_OBJS = 2


def generate_combinations():
    all_combinations = []
    for r in range(1, len(SHAPES) + 1):
        shape_combos = combinations(SHAPES, r)

        for shape_combo in shape_combos:
            mode_lists = [STR2DICT[shape] for shape in shape_combo]
            mode_combos = product(*mode_lists)
            for mode_combo in mode_combos:
                all_combinations.append([(REVERSED_STR2DICT[x], x) for x in mode_combo])

    return all_combinations


def powerset(iterable):
    "powerset([1,2,3]) → (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return_list = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    return [item for item in return_list if len(item) > 0]


# Generate all 639 combinations
ALL_MASKING_COMBINATIONS_SHAPES = generate_combinations()
ALL_MASKING_COMBINATIONS = powerset(MASKING_MODES)


class MaskingFunctions(Enum):
    SPACE = 1
    TIME = 0
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

    space_time_high_x: torch.Tensor
    space_time_med_x: torch.Tensor
    space_time_low_x: torch.Tensor
    space_x: torch.Tensor
    time_x: torch.Tensor
    static_x: torch.Tensor
    space_time_high_mask: torch.Tensor
    space_time_med_mask: torch.Tensor
    space_time_low_mask: torch.Tensor
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
    output_modes: List[Tuple[str, str]] = []
    for mode in modes:
        assert mode in MASKING_MODES
        if mode in unmasking_modes:
            if len(unmasking_modes) == 1:
                # don't remove any more from the unmasking modes
                continue
            elif len(output_modes) == 0:
                output_modes.append(mode)
                unmasking_modes.remove(mode)
            else:
                # neither modes or unmasking_modes are bottlenecked;
                # randomly select which one to remove
                if random.random() <= 0.5:
                    output_modes.append(mode)
                    unmasking_modes.remove(mode)
        else:
            output_modes.append(mode)
    assert len(output_modes) >= 1
    assert len(unmasking_modes) >= 1
    return output_modes, unmasking_modes


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
    s_t_h_x: torch.Tensor,
    s_t_m_x: torch.Tensor,
    s_t_l_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_s_t_m: torch.Tensor,
    valid_data_mask_s_t_l: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size_high_res: int,
    patch_size_med_res: int,
    patch_size_low_res: int,
    image_size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
    masking_probabilities: List[float],
    masking_function: MaskingFunctions,
    max_unmasking_channels: int,
    unmasking_channels_combo: str = "shapes",
    ablate: str = "",
) -> MaskedOutput:
    assert len(masking_probabilities) == len(MASKING_MODES)

    # take care of ablations, TODO: maybe move this to the dataset class
    if ablate == "high_res":
        print("Ablating high res data")
        valid_data_mask_s_t_h = torch.zeros_like(valid_data_mask_s_t_h)
    elif ablate == "low_res":
        print("Ablating low res data including NDVI and NDSI")
        valid_data_mask_s_t_l = torch.zeros_like(valid_data_mask_s_t_l)
        valid_data_mask_s_t_m = torch.zeros_like(valid_data_mask_s_t_m)
        # the first 4 channels of time are VIIRS coarse resolution data
        valid_data_mask_t[..., :4] = torch.zeros_like(valid_data_mask_t[..., :4])
    elif ablate == "aux":
        print("Ablating auxiliary data")
        valid_data_mask_sp = torch.zeros_like(valid_data_mask_sp)
        # the last channels of time are ERA5 data
        valid_data_mask_t[..., 4:] = torch.zeros_like(valid_data_mask_t[..., 4:])
    elif ablate == "location":
        print("Ablating location data")
        valid_data_mask_st = torch.zeros_like(valid_data_mask_st)

    elif ablate == "time":
        print("Ablating all but one timestep")
        # default: keep the first timestep
        timestep_to_keep = 0
        # look for a timestep that contains landsat data, so where valid_data_mask_s_t_h[...,10] is not all zero
        for t in range(num_timesteps):
            if valid_data_mask_s_t_h[..., t].sum() > 0:
                timestep_to_keep = t
                break
        original_valid_data_mask_s_t_h = valid_data_mask_s_t_h.clone()
        original_valid_data_mask_s_t_m = valid_data_mask_s_t_m.clone()
        original_valid_data_mask_s_t_l = valid_data_mask_s_t_l.clone()
        original_valid_data_mask_t = valid_data_mask_t.clone()

        valid_data_mask_s_t_h = torch.zeros_like(valid_data_mask_s_t_h)
        valid_data_mask_s_t_h[:, :, :, timestep_to_keep, :] = original_valid_data_mask_s_t_h[..., timestep_to_keep, :]
        valid_data_mask_s_t_m = torch.zeros_like(valid_data_mask_s_t_m)
        valid_data_mask_s_t_m[:, :, :, timestep_to_keep, :] = original_valid_data_mask_s_t_m[..., timestep_to_keep, :]
        valid_data_mask_s_t_l = torch.zeros_like(valid_data_mask_s_t_l)
        valid_data_mask_s_t_l[:, :, :, timestep_to_keep, :] = original_valid_data_mask_s_t_l[..., timestep_to_keep, :]
        valid_data_mask_t = torch.zeros_like(valid_data_mask_t)
        valid_data_mask_t[:, timestep_to_keep, :] = original_valid_data_mask_t[..., timestep_to_keep, :]

    # not used by Snow Galileo so far (only random masking)
    if masking_function.value < 2:
        f: Callable = batch_mask_space if masking_function.value == 1 else batch_mask_time  # type: ignore
        num_masking_modes = random.choice(list(range(2, MAX_MASKING_STRATEGIES + 1)))
        masking_modes = weighted_sample_without_replacement(
            MASKING_MODES, weights=masking_probabilities, k=num_masking_modes
        )

        # isolate the unmasking candidates which (1) have the right number of channels and
        # (b) don't intersect with the masking_modes
        if unmasking_channels_combo == "shapes":
            unmasking_mode_candidates = [
                x
                for x in ALL_MASKING_COMBINATIONS_SHAPES
                if ((len(x) <= max_unmasking_channels) and (len(set(x) & set(masking_modes)) == 0))
            ]
        elif unmasking_channels_combo == "all":
            unmasking_mode_candidates = [
                x
                for x in ALL_MASKING_COMBINATIONS
                if ((len(x) <= max_unmasking_channels) and (len(set(x) & set(masking_modes)) == 0))
            ]
        else:
            raise ValueError(
                "Expected unmasking_channels_combo to be "
                f"'shapes' or 'all', got {unmasking_channels_combo}"
            )
        unmasking_modes = random.choice(unmasking_mode_candidates)

        masking_modes, unmasking_modes = check_modes_for_conflicts(masking_modes, unmasking_modes)
        masked_output = f(
            *subset_and_augment_batch_of_images(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            mode=masking_modes,
            decoder_mode=unmasking_modes,
            patch_size_high_res=patch_size_high_res,
            patch_size_med_res=patch_size_med_res,
            patch_size_low_res=patch_size_low_res,
        )

    # used by Snow Galileo
    elif masking_function.value == 2:
        # 2 is random
        masked_output = batch_mask_random(
            *subset_and_augment_batch_of_images(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
                size=image_size,
                num_timesteps=num_timesteps,
                augmentation_strategies=augmentation_strategies,
            ),
            encode_ratio=encode_ratio,
            decode_ratio=decode_ratio,
            patch_size_high_res=patch_size_high_res,
            patch_size_med_res=patch_size_med_res,
            patch_size_low_res=patch_size_low_res,
        )

    else:
        raise AssertionError(f"Unexpected strategy {masking_function}")

    return masked_output


def subset_and_augment_batch_of_images(
    space_time_high_x: torch.Tensor,
    space_time_med_x: torch.Tensor,
    space_time_low_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_s_t_m: torch.Tensor,
    valid_data_mask_s_t_l: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    size: int,
    num_timesteps: int,
    augmentation_strategies: Optional[Dict],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    assert (space_time_high_x.shape[1] == valid_data_mask_s_t_h.shape[1] == space_x.shape[1]) & (
        space_time_high_x.shape[2]
        == valid_data_mask_s_t_h.shape[2]
        == valid_data_mask_sp.shape[2]
        == space_x.shape[2]
    )
    assert (
        time_x.shape[1]
        == space_time_high_x.shape[3]
        == months.shape[1]
        == space_time_med_x.shape[3]
        == space_time_low_x.shape[3]
        == valid_data_mask_t.shape[1]
        == valid_data_mask_s_t_h.shape[3]
        == valid_data_mask_s_t_m.shape[3]
        == valid_data_mask_s_t_l.shape[3]
    )
    # assert space_time_med_x.shape[1] == space_time_med_x.shape[2] == 3
    assert space_time_low_x.shape[1] == space_time_low_x.shape[2] == 2
    if augmentation_strategies is not None:
        return Augmentation(augmentation_strategies).apply(
            space_time_high_x,
            space_time_med_x,
            space_time_low_x,
            space_x,
            time_x,
            static_x,
            months,
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        )
    return (
        space_time_high_x,
        space_time_med_x,
        space_time_low_x,
        space_x,
        time_x,
        static_x,
        months,
        valid_data_mask_s_t_h,
        valid_data_mask_s_t_m,
        valid_data_mask_s_t_l,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
    )


def _random_mask_for_b(
    b: int, device: torch.device, encode_ratio: float, decode_ratio: float
) -> torch.Tensor:
    mask = torch.rand(b, device=device)
    mask[mask >= (1 - encode_ratio)] = 0
    mask[mask <= decode_ratio] = 2
    # all the rest is ignored by both the encoder and decoder
    mask[(mask != 0) | (mask != 2)] = 1
    return mask


def _aggregate_mask_per_channel_group(
    invalid_data_mask_s_t_h,
    invalid_data_mask_s_t_m,
    invalid_data_mask_s_t_l,
    invalid_data_mask_sp,
    invalid_data_mask_t,
    invalid_data_mask_st,
):
    SPACE_TIME_HIGH_RES_BAND_EXPANSION = [
        len(i) for i in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.values()
    ]
    SPACE_TIME_MED_RES_BAND_EXPANSION = [
        len(i) for i in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.values()
    ]
    SPACE_TIME_LOW_RES_BAND_EXPANSION = [
        len(i) for i in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.values()
    ]
    SPACE_BAND_EXPANSION = [len(i) for i in SPACE_BAND_GROUPS_IDX.values()]
    TIME_BAND_EXPANSION = [len(i) for i in TIME_BANDS_GROUPS_IDX.values()]
    STATIC_BAND_EXPANSION = [len(i) for i in STATIC_BAND_GROUPS_IDX.values()]

    # Split tensor into groups and perform logical AND across each group to make sure all invalid data is masked out
    aggregated_invalid_data_mask_s_t_h = torch.stack(
        [
            invalid_data_mask_s_t_h[
                ...,
                sum(SPACE_TIME_HIGH_RES_BAND_EXPANSION[:i]) : sum(
                    SPACE_TIME_HIGH_RES_BAND_EXPANSION[:i]
                )
                + size,
            ].any(dim=-1)
            for i, size in enumerate(SPACE_TIME_HIGH_RES_BAND_EXPANSION)
        ],
        dim=-1,
    )
    aggregated_invalid_data_mask_s_t_m = torch.stack(
        [
            invalid_data_mask_s_t_m[
                ...,
                sum(SPACE_TIME_MED_RES_BAND_EXPANSION[:i]) : sum(
                    SPACE_TIME_MED_RES_BAND_EXPANSION[:i]
                )
                + size,
            ].any(dim=-1)
            for i, size in enumerate(SPACE_TIME_MED_RES_BAND_EXPANSION)
        ],
        dim=-1,
    )
    aggregated_invalid_data_mask_s_t_l = torch.stack(
        [
            invalid_data_mask_s_t_l[
                ...,
                sum(SPACE_TIME_LOW_RES_BAND_EXPANSION[:i]) : sum(
                    SPACE_TIME_LOW_RES_BAND_EXPANSION[:i]
                )
                + size,
            ].any(dim=-1)
            for i, size in enumerate(SPACE_TIME_LOW_RES_BAND_EXPANSION)
        ],
        dim=-1,
    )
    aggregated_invalid_data_mask_sp = torch.stack(
        [
            invalid_data_mask_sp[
                ..., sum(SPACE_BAND_EXPANSION[:i]) : sum(SPACE_BAND_EXPANSION[:i]) + size
            ].any(dim=-1)
            for i, size in enumerate(SPACE_BAND_EXPANSION)
        ],
        dim=-1,
    )
    aggregated_invalid_data_mask_t = torch.stack(
        [
            invalid_data_mask_t[
                ..., sum(TIME_BAND_EXPANSION[:i]) : sum(TIME_BAND_EXPANSION[:i]) + size
            ].any(dim=-1)
            for i, size in enumerate(TIME_BAND_EXPANSION)
        ],
        dim=-1,
    )
    aggregated_invalid_data_mask_st = torch.stack(
        [
            invalid_data_mask_st[
                ..., sum(STATIC_BAND_EXPANSION[:i]) : sum(STATIC_BAND_EXPANSION[:i]) + size
            ].any(dim=-1)
            for i, size in enumerate(STATIC_BAND_EXPANSION)
        ],
        dim=-1,
    )

    return (
        aggregated_invalid_data_mask_s_t_h,
        aggregated_invalid_data_mask_s_t_m,
        aggregated_invalid_data_mask_s_t_l,
        aggregated_invalid_data_mask_sp,
        aggregated_invalid_data_mask_t,
        aggregated_invalid_data_mask_st,
    )


def batch_mask_time(
    space_time_high_x: torch.Tensor,
    space_time_med_x: torch.Tensor,
    space_time_low_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_s_t_m: torch.Tensor,
    valid_data_mask_s_t_l: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size_high_res: int,
    patch_size_med_res: int,
    patch_size_low_res: int,
    decoder_mode: List[Tuple[str, str]],
    mode: List[Tuple[str, str]],
):
    """
    Masks out blocks of hxwx1xBAND_GROUPs.
    e.g. if mask_ratio=0.25, then 1/4 of the timesteps
    (and the static channel groups, with 1/4 probability) will be masked out

    Operates over batches where each item in the batch has independently masked timesteps
    """
    # TODO: This function is not tested yet
    b, h_h, w_h, t, _ = space_time_high_x.shape
    b, h_m, w_m, _, _ = space_time_med_x.shape
    b, h_l, w_l, _, _ = space_time_low_x.shape
    assert t >= 3

    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)
    # if there is only a single timestep, decode it
    num_timesteps_to_decode = max(int(t * decode_ratio), 1)
    num_timesteps_to_encode = max(int(t * encode_ratio), 1)
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
        space_time_high_x.device
    )
    space_time_high_res_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h_h,
        w=w_h,
        c_g=len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX),
    ).clone()
    space_time_med_res_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h_m,
        w=w_m,
        c_g=len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX),
    ).clone()
    space_time_low_res_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b h w t c_g",
        h=h_l,
        w=w_l,
        c_g=len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX),
    ).clone()
    # make the mask as if bands_to_mask and bands_to_decode both = None
    time_mask = repeat(
        b_flat_timesteps_t,
        "b t-> b t c_g",
        c_g=len(TIME_BANDS_GROUPS_IDX),
    ).clone()
    space_mask = _random_mask_for_b(b, space_x.device, encode_ratio, decode_ratio)
    space_mask = repeat(
        space_mask, "b -> b h w c_g", h=h_h, w=w_h, c_g=len(SPACE_BAND_GROUPS_IDX)
    ).clone()
    static_mask = _random_mask_for_b(b, static_x.device, encode_ratio, decode_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()
    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in time data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        space_mask = torch.clamp(space_mask, min=1)

        s_t_h_e, s_t_m_e, s_t_l_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_h_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_high_bands_to_mask = s_t_h_e[1]
            space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask] = torch.clamp(
                space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask], min=1
            )
        else:
            space_time_high_res_mask = torch.clamp(space_time_high_res_mask, min=1)

        if len(s_t_m_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_med_bands_to_mask = s_t_m_e[1]
            space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask] = torch.clamp(
                space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask], min=1
            )
        else:
            space_time_med_res_mask = torch.clamp(space_time_med_res_mask, min=1)

        if len(s_t_l_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_low_bands_to_mask = s_t_l_e[1]
            space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask] = torch.clamp(
                space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask], min=1
            )
        else:
            space_time_low_res_mask = torch.clamp(space_time_low_res_mask, min=1)

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

        s_t_h_d, s_t_m_d, s_t_l_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_h_d[0]) > 0:
            # there are space time bands to decode
            s_t_high_bands_to_mask = s_t_h_d[1]
            space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask] = torch.clamp(
                space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask], max=1
            )
        else:
            space_time_high_res_mask = torch.clamp(space_time_high_res_mask, max=1)

        if len(s_t_m_d[0]) > 0:
            # there are space time bands to decode
            s_t_med_bands_to_mask = s_t_m_d[1]
            space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask] = torch.clamp(
                space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask], max=1
            )
        else:
            space_time_med_res_mask = torch.clamp(space_time_med_res_mask, max=1)

        if len(s_t_l_d[0]) > 0:
            # there are space time bands to decode
            s_t_low_bands_to_mask = s_t_l_d[1]
            space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask] = torch.clamp(
                space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask], max=1
            )
        else:
            space_time_low_res_mask = torch.clamp(space_time_low_res_mask, max=1)

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

    invalid_data_mask_s_t_h = np.logical_not(valid_data_mask_s_t_h)
    invalid_data_mask_s_t_m = np.logical_not(valid_data_mask_s_t_m)
    invalid_data_mask_s_t_l = np.logical_not(valid_data_mask_s_t_l)
    invalid_data_mask_sp = np.logical_not(valid_data_mask_sp)
    invalid_data_mask_t = np.logical_not(valid_data_mask_t)
    invalid_data_mask_st = np.logical_not(valid_data_mask_st)

    cg_mask_s_t_h, cg_mask_s_t_m, cg_mask_s_t_l, cg_mask_sp, cg_mask_t, cg_mask_st = (
        _aggregate_mask_per_channel_group(
            invalid_data_mask_s_t_h,
            invalid_data_mask_s_t_m,
            invalid_data_mask_s_t_l,
            invalid_data_mask_sp,
            invalid_data_mask_t,
            invalid_data_mask_st,
        )
    )

    # since we mask out the same values within each channel we can assume that the mask is the same for each channel group
    space_time_high_res_mask[cg_mask_s_t_h.bool()] = 1
    space_time_med_res_mask[cg_mask_s_t_m.bool()] = 1
    space_time_low_res_mask[cg_mask_s_t_l.bool()] = 1
    space_mask[cg_mask_sp.bool()] = 1
    time_mask[cg_mask_t.bool()] = 1
    static_mask[cg_mask_st.bool()] = 1

    return MaskedOutput(
        space_time_high_x.clone(),
        space_time_med_x.clone(),
        space_time_low_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_high_res_mask,
        space_time_med_res_mask,
        space_time_low_res_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


def create_two_d_space_mask(
    batch_size: int, height: int, width: int, patch_size: int, encode_ratio, decode_ratio
):
    # TODO: this function is not adjusted yet
    assert (height % patch_size == 0) and (width % patch_size == 0)
    h_p = int(height / patch_size)
    w_p = int(width / patch_size)
    total_patches = h_p * w_p
    num_patches_to_encode = int(total_patches * encode_ratio)
    num_patches_to_decode = int(total_patches * decode_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_patches = np.concatenate(
        (
            np.ones(
                total_patches - (num_patches_to_encode + num_patches_to_decode), dtype=np.int_
            ),
            np.ones(num_patches_to_decode, dtype=np.int_) * 2,
            np.zeros(num_patches_to_encode, dtype=np.int_),
        )
    )
    b_flat_patches = repeat(flat_patches, "p -> b p", b=batch_size)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_patches = rng.permuted(b_flat_patches, axis=1)
    two_d_patch_mask = rearrange(b_flat_patches, "b (h w) -> b h w", h=h_p, w=w_p)
    two_d_mask = np.repeat(
        np.repeat(two_d_patch_mask, repeats=patch_size, axis=1), repeats=patch_size, axis=2
    )
    return two_d_mask


# batch mask space will (at this point) only work for high res data
def batch_mask_space(
    space_time_high_x: torch.Tensor,
    space_time_med_x: torch.Tensor,
    space_time_low_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_s_t_m: torch.Tensor,
    valid_data_mask_s_t_l: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    mode: List[Tuple[str, str]],
    decoder_mode: List[Tuple[str, str]],
    patch_size_high_res: int,
    patch_size_med_res: int = 1,
    patch_size_low_res: int = 1,
):
    """
    Masks out patches (blocks of of pxpxtxBAND_GROUPs).
    e.g. if mask_ratio=0.25, h = w = 8 and p=2, then a high res mask might be:
    [0 0 1 1]
    [0 0 1 1]
    [0 0 0 0]
    [0 0 0 0]
    a med res mask (h = w = 3 and p = 1) might be:
    [0 0 1]
    [0 0 1]
    [0 0 0]
    and a low res mask (h = w = 2 and p = 1):
    [0 1]
    [0 0]
    repeated over all dynamic timesteps + channel groups and static channel groups
    Operates over batches where each item in the batch is independently masked
    """
    # TODO: this function is not adjusted yet
    bands_to_encode = check_mode_and_return_channels(mode)
    bands_to_decode = check_mode_and_return_channels(decoder_mode)

    b, h_h, w_h, t, _ = space_time_high_x.shape
    b, h_m, w_m, t, _ = space_time_med_x.shape
    b, h_l, w_l, t, _ = space_time_low_x.shape

    two_d_mask_high_res = create_two_d_space_mask(
        batch_size=b,
        height=h_h,
        width=w_h,
        patch_size=patch_size_high_res,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )
    two_d_mask_med_res = create_two_d_space_mask(
        batch_size=b,
        height=h_m,
        width=w_m,
        patch_size=patch_size_med_res,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )
    two_d_mask_low_res = create_two_d_space_mask(
        batch_size=b,
        height=h_l,
        width=w_l,
        patch_size=patch_size_low_res,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )

    space_time_high_res_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask_high_res,
                "b h w -> b h w t c_g",
                t=t,
                c_g=len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_time_high_x.device)
    )
    space_time_med_res_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask_med_res,
                "b h w -> b h w t c_g",
                t=t,
                c_g=len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_time_med_x.device)
    )
    space_time_low_res_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask_low_res,
                "b h w -> b h w t c_g",
                t=t,
                c_g=len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_time_low_x.device)
    )
    space_mask = (
        torch.from_numpy(
            repeat(
                two_d_mask_high_res,
                "b h w -> b h w c_g",
                c_g=len(SPACE_BAND_GROUPS_IDX),
            )
        )
        .clone()
        .to(space_x.device)
    )
    time_mask = _random_mask_for_b(b, time_x.device, encode_ratio, decode_ratio)
    time_mask = repeat(time_mask, "b -> b t c_g", t=t, c_g=len(TIME_BANDS_GROUPS_IDX)).clone()
    static_mask = _random_mask_for_b(b, static_x.device, encode_ratio, decode_ratio)
    static_mask = repeat(static_mask, "b -> b c_g", c_g=len(STATIC_BAND_GROUPS_IDX)).clone()

    if max([len(x[0]) for x in bands_to_encode]) >= 1:  # encoder mode != random
        # for static in space data,
        # ignore all previous calculations about what should be encoded
        static_mask = torch.clamp(static_mask, min=1)
        time_mask = torch.clamp(time_mask, min=1)

        s_t_h_e, s_t_m_e, s_t_l_e, s_e, t_e, st_e = bands_to_encode

        if len(s_t_h_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_high_bands_to_mask = s_t_h_e[1]
            space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask] = torch.clamp(
                space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask], min=1
            )
        else:
            space_time_high_res_mask = torch.clamp(space_time_high_res_mask, min=1)
        if len(s_t_m_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_med_bands_to_mask = s_t_m_e[1]
            space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask] = torch.clamp(
                space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask], min=1
            )
        else:
            space_time_med_res_mask = torch.clamp(space_time_med_res_mask, min=1)
        if len(s_t_l_e[0]) > 0:
            # there are high res space time bands to encode
            s_t_low_bands_to_mask = s_t_l_e[1]
            space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask] = torch.clamp(
                space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask], min=1
            )
        else:
            space_time_low_res_mask = torch.clamp(space_time_low_res_mask, min=1)
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

        s_t_h_d, s_t_m_d, s_t_l_d, s_d, t_d, st_d = bands_to_decode

        if len(s_t_h_d[0]) > 0:
            # there are space time bands to decode
            s_t_high_bands_to_mask = s_t_h_d[1]
            space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask] = torch.clamp(
                space_time_high_res_mask[:, :, :, :, s_t_high_bands_to_mask], max=1
            )
        else:
            space_time_med_res_mask = torch.clamp(space_time_med_res_mask, max=1)
        if len(s_t_m_d[0]) > 0:
            # there are space time bands to decode
            s_t_med_bands_to_mask = s_t_m_d[1]
            space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask] = torch.clamp(
                space_time_med_res_mask[:, :, :, :, s_t_med_bands_to_mask], max=1
            )
        else:
            space_time_med_res_mask = torch.clamp(space_time_med_res_mask, max=1)
        if len(s_t_l_d[0]) > 0:
            # there are space time bands to decode
            s_t_low_bands_to_mask = s_t_l_d[1]
            space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask] = torch.clamp(
                space_time_low_res_mask[:, :, :, :, s_t_low_bands_to_mask], max=1
            )
        else:
            space_time_low_res_mask = torch.clamp(space_time_low_res_mask, max=1)

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

    invalid_data_mask_s_t_h = np.logical_not(valid_data_mask_s_t_h)
    invalid_data_mask_s_t_m = np.logical_not(valid_data_mask_s_t_m)
    invalid_data_mask_s_t_l = np.logical_not(valid_data_mask_s_t_l)
    invalid_data_mask_sp = np.logical_not(valid_data_mask_sp)
    invalid_data_mask_t = np.logical_not(valid_data_mask_t)
    invalid_data_mask_st = np.logical_not(valid_data_mask_st)

    cg_mask_s_t_h, cg_mask_s_t_m, cg_mask_s_t_l, cg_mask_sp, cg_mask_t, cg_mask_st = (
        _aggregate_mask_per_channel_group(
            invalid_data_mask_s_t_h,
            invalid_data_mask_s_t_m,
            invalid_data_mask_s_t_l,
            invalid_data_mask_sp,
            invalid_data_mask_t,
            invalid_data_mask_st,
        )
    )

    # since we mask out the same values within each channel we can assume that the mask is the same for each channel group
    space_time_high_res_mask[cg_mask_s_t_h.bool()] = 1
    space_time_med_res_mask[cg_mask_s_t_m.bool()] = 1
    space_time_low_res_mask[cg_mask_s_t_l.bool()] = 1
    space_mask[cg_mask_sp.bool()] = 1
    time_mask[cg_mask_t.bool()] = 1
    static_mask[cg_mask_st.bool()] = 1

    return MaskedOutput(
        space_time_high_x.clone(),
        space_time_med_x.clone(),
        space_time_low_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_high_res_mask,
        space_time_med_res_mask,
        space_time_low_res_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )


# not functional at this point (because med and low are treated the same as high)
def batch_mask_random(
    space_time_high_x: torch.Tensor,
    space_time_med_x: torch.Tensor,
    space_time_low_x: torch.Tensor,
    space_x: torch.Tensor,
    time_x: torch.Tensor,
    static_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_s_t_m: torch.Tensor,
    valid_data_mask_s_t_l: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    encode_ratio: float,
    decode_ratio: float,
    patch_size_high_res: int,
    patch_size_med_res: int = 1,
    patch_size_low_res: int = 1,
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
    b, h_h, w_h, t, _ = space_time_high_x.shape
    b, h_m, w_m, t, _ = space_time_med_x.shape
    b, h_l, w_l, t, _ = space_time_low_x.shape

    # extract the number of tokens for each type of data
    # we assume that the patch sizes divide height and width exactly
    assert (h_h % patch_size_high_res == 0) and (w_h % patch_size_high_res == 0)
    h_p_h = int(h_h / patch_size_high_res)
    w_p_h = int(w_h / patch_size_high_res)

    assert (h_m % patch_size_med_res == 0) and (w_m % patch_size_med_res == 0)
    h_p_m = int(h_m / patch_size_med_res)
    w_p_m = int(w_m / patch_size_med_res)

    assert (h_l % patch_size_low_res == 0) and (w_l % patch_size_low_res == 0)
    h_p_l = int(h_l / patch_size_low_res)
    w_p_l = int(w_l / patch_size_low_res)

    c_s_t_h = len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)
    c_s_t_m = len(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX)
    c_s_t_l = len(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX)
    c_sp = len(SPACE_BAND_GROUPS_IDX)
    c_t = len(TIME_BANDS_GROUPS_IDX)
    c_st = len(STATIC_BAND_GROUPS_IDX)

    num_space_time_high_res_tokens = h_p_h * w_p_h * t * c_s_t_h
    num_space_time_med_res_tokens = h_p_m * w_p_m * t * c_s_t_m
    num_space_time_low_res_tokens = h_p_l * w_p_l * t * c_s_t_l
    # space tokens are encoded as high resolution
    num_space_tokens = h_p_h * w_p_h * c_sp
    num_time_tokens = t * c_t
    num_static_tokens = c_st

    total_tokens = (
        num_space_time_high_res_tokens
        + num_space_time_med_res_tokens
        + num_space_time_low_res_tokens
        + num_space_tokens
        + num_time_tokens
        + num_static_tokens
    )
    tokens_the_decoder_will_unmask = int(total_tokens * decode_ratio)
    tokens_the_encoder_will_encode = int(total_tokens * encode_ratio)
    # we do this as a numpy array to take advantage of
    # numpy's permuted function
    flat_tokens = np.concatenate(
        (
            np.ones(
                total_tokens - (tokens_the_encoder_will_encode + tokens_the_decoder_will_unmask),
                dtype=np.int_,
            ),
            np.ones(tokens_the_decoder_will_unmask, dtype=np.int_) * 2,
            np.zeros(
                tokens_the_encoder_will_encode,
                dtype=np.int_,
            ),
        )
    )
    b_flat_tokens = repeat(flat_tokens, "t -> b t", b=b)
    # hopefully this will allow for reproducibility, since random is seeded
    rng = np.random.default_rng(random.randint(0, 100))
    b_flat_tokens = rng.permuted(b_flat_tokens, axis=1)

    s_t_h_tokens = b_flat_tokens[:, :num_space_time_high_res_tokens]
    s_t_h_tokens = rearrange(
        s_t_h_tokens, "b (h w t c) -> b h w t c", h=h_p_h, w=w_p_h, t=t, c=c_s_t_h
    )
    space_time_high_res_mask = torch.from_numpy(
        np.repeat(
            np.repeat(s_t_h_tokens, repeats=patch_size_high_res, axis=1),
            repeats=patch_size_high_res,
            axis=2,
        )
    ).to(space_time_high_x.device)

    s_t_m_tokens = b_flat_tokens[
        :,
        num_space_time_high_res_tokens : (
            num_space_time_high_res_tokens + num_space_time_med_res_tokens
        ),
    ]
    s_t_m_tokens = rearrange(
        s_t_m_tokens, "b (h w t c) -> b h w t c", h=h_p_m, w=w_p_m, t=t, c=c_s_t_m
    )
    space_time_med_res_mask = torch.from_numpy(
        np.repeat(
            np.repeat(s_t_m_tokens, repeats=patch_size_med_res, axis=1),
            repeats=patch_size_med_res,
            axis=2,
        )
    ).to(space_time_med_x.device)

    s_t_l_tokens = b_flat_tokens[
        :,
        (num_space_time_high_res_tokens + num_space_time_med_res_tokens) : (
            num_space_time_high_res_tokens
            + num_space_time_med_res_tokens
            + num_space_time_low_res_tokens
        ),
    ]
    s_t_l_tokens = rearrange(
        s_t_l_tokens, "b (h w t c) -> b h w t c", h=h_p_l, w=w_p_l, t=t, c=c_s_t_l
    )
    space_time_low_res_mask = torch.from_numpy(
        np.repeat(
            np.repeat(s_t_l_tokens, repeats=patch_size_low_res, axis=1),
            repeats=patch_size_low_res,
            axis=2,
        )
    ).to(space_time_low_x.device)

    space_tokens = b_flat_tokens[
        :,
        -(num_space_tokens + num_time_tokens + num_static_tokens) : -(
            num_time_tokens + num_static_tokens
        ),
    ]
    # space only tokens are in high resolution
    space_tokens = rearrange(space_tokens, "b (h w c) -> b h w c", h=h_p_h, w=w_p_h, c=c_sp)
    space_mask = torch.from_numpy(
        np.repeat(
            np.repeat(space_tokens, repeats=patch_size_high_res, axis=1),
            repeats=patch_size_high_res,
            axis=2,
        )
    ).to(space_x.device)

    time_tokens = b_flat_tokens[:, -(num_time_tokens + num_static_tokens) : -num_static_tokens]
    time_mask = torch.from_numpy(rearrange(time_tokens, "b (t c) -> b t c", t=t, c=c_t)).to(
        time_x.device
    )

    static_tokens = b_flat_tokens[:, -num_static_tokens:]
    static_mask = torch.from_numpy(static_tokens).to(static_x.device)

    invalid_data_mask_s_t_h = np.logical_not(valid_data_mask_s_t_h)
    invalid_data_mask_s_t_m = np.logical_not(valid_data_mask_s_t_m)
    invalid_data_mask_s_t_l = np.logical_not(valid_data_mask_s_t_l)
    invalid_data_mask_sp = np.logical_not(valid_data_mask_sp)
    invalid_data_mask_t = np.logical_not(valid_data_mask_t)
    invalid_data_mask_st = np.logical_not(valid_data_mask_st)

    cg_mask_s_t_h, cg_mask_s_t_m, cg_mask_s_t_l, cg_mask_sp, cg_mask_t, cg_mask_st = (
        _aggregate_mask_per_channel_group(
            invalid_data_mask_s_t_h,
            invalid_data_mask_s_t_m,
            invalid_data_mask_s_t_l,
            invalid_data_mask_sp,
            invalid_data_mask_t,
            invalid_data_mask_st,
        )
    )

    # since we mask out the same values within each channel we can assume that the mask is the same for each channel group
    space_time_high_res_mask[cg_mask_s_t_h.bool()] = 1
    space_time_med_res_mask[cg_mask_s_t_m.bool()] = 1
    space_time_low_res_mask[cg_mask_s_t_l.bool()] = 1
    space_mask[cg_mask_sp.bool()] = 1
    time_mask[cg_mask_t.bool()] = 1
    static_mask[cg_mask_st.bool()] = 1

    return MaskedOutput(
        space_time_high_x.clone(),
        space_time_med_x.clone(),
        space_time_low_x.clone(),
        space_x.clone(),
        time_x.clone(),
        static_x.clone(),
        space_time_high_res_mask,
        space_time_med_res_mask,
        space_time_low_res_mask,
        space_mask,
        time_mask,
        static_mask,
        months,
    )
