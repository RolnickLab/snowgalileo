import random
from typing import Dict, NamedTuple, Optional, Tuple

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


def batch_subset_mask_galileo(
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
    augmentation_strategies: Optional[Dict],
) -> MaskedOutput:
    masked_output = batch_mask_random(
        *check_and_augment_batch_of_images(
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
            augmentation_strategies=augmentation_strategies,
        ),
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
        patch_size_high_res=patch_size_high_res,
        patch_size_med_res=patch_size_med_res,
        patch_size_low_res=patch_size_low_res,
    )
    return masked_output


def check_and_augment_batch_of_images(
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
    # better check too often than no
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
    assert space_time_med_x.shape[1] == space_time_med_x.shape[2]
    assert space_time_low_x.shape[1] == space_time_low_x.shape[2]
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
