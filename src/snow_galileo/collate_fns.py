### Original Code:
### Copyright (c) 2024 Presto Authors
### Licensed under the MIT License.
### A copy of the MIT License is available in the LICENSE file in the root directory of this project.

### Modifications by marlens123:
### - Included medium and low resolution data

from typing import NamedTuple, Tuple

import torch
from torch.utils.data import default_collate

from snow_galileo.masking import (
    batch_subset_mask_galileo,
)


class CollateFnOutput(NamedTuple):
    s_t_h_x: torch.Tensor
    s_t_m_x: torch.Tensor
    s_t_l_x: torch.Tensor
    sp_x: torch.Tensor
    t_x: torch.Tensor
    st_x: torch.Tensor
    s_t_h_m: torch.Tensor
    s_t_m_m: torch.Tensor
    s_t_l_m: torch.Tensor
    sp_m: torch.Tensor
    t_m: torch.Tensor
    st_m: torch.Tensor
    months: torch.Tensor
    patch_size_high_res: float
    patch_size_med_res: float
    patch_size_low_res: float


def collated_batch_to_output(
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
    patch_size_high_res,
    patch_size_med_res,
    patch_size_low_res,
    encode_ratio,
    decode_ratio,
    augmentation_strategies=None,
) -> CollateFnOutput:
    masked_output = batch_subset_mask_galileo(
        s_t_h_x=s_t_h_x,
        s_t_m_x=s_t_m_x,
        s_t_l_x=s_t_l_x,
        sp_x=sp_x,
        t_x=t_x,
        st_x=st_x,
        months=months,
        valid_data_mask_s_t_h=valid_data_mask_s_t_h,
        valid_data_mask_s_t_m=valid_data_mask_s_t_m,
        valid_data_mask_s_t_l=valid_data_mask_s_t_l,
        valid_data_mask_sp=valid_data_mask_sp,
        valid_data_mask_t=valid_data_mask_t,
        valid_data_mask_st=valid_data_mask_st,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
        patch_size_high_res=patch_size_high_res,
        patch_size_med_res=patch_size_med_res,
        patch_size_low_res=patch_size_low_res,
        augmentation_strategies=augmentation_strategies,
    )

    s_t_h_x = masked_output.space_time_high_x
    s_t_m_x = masked_output.space_time_med_x
    s_t_l_x = masked_output.space_time_low_x
    sp_x = masked_output.space_x
    t_x = masked_output.time_x
    st_x = masked_output.static_x
    s_t_h_m = masked_output.space_time_high_mask
    s_t_m_m = masked_output.space_time_med_mask
    s_t_l_m = masked_output.space_time_low_mask
    sp_m = masked_output.space_mask
    t_m = masked_output.time_mask
    st_m = masked_output.static_mask
    months = masked_output.months

    return CollateFnOutput(
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        s_t_m_m,
        s_t_l_m,
        sp_m,
        t_m,
        st_m,
        months,
        patch_size_high_res,
        patch_size_med_res,
        patch_size_low_res,
    )


@torch.no_grad()
def mae_collate_fn(
    batch,
    patch_size_high_res,
    patch_size_med_res,
    patch_size_low_res,
    encode_ratio,
    decode_ratio,
    augmentation_strategies=None,
) -> Tuple[CollateFnOutput, CollateFnOutput, CollateFnOutput, CollateFnOutput]:
    (
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
    ) = default_collate(batch)

    input_args = {
        "s_t_h_x": s_t_h_x,
        "s_t_m_x": s_t_m_x,
        "s_t_l_x": s_t_l_x,
        "sp_x": sp_x,
        "t_x": t_x,
        "st_x": st_x,
        "months": months,
        "valid_data_mask_s_t_h": valid_data_mask_s_t_h,
        "valid_data_mask_s_t_m": valid_data_mask_s_t_m,
        "valid_data_mask_s_t_l": valid_data_mask_s_t_l,
        "valid_data_mask_sp": valid_data_mask_sp,
        "valid_data_mask_t": valid_data_mask_t,
        "valid_data_mask_st": valid_data_mask_st,
        "patch_size_high_res": patch_size_high_res,
        "patch_size_med_res": patch_size_med_res,
        "patch_size_low_res": patch_size_low_res,
        "encode_ratio": encode_ratio,
        "decode_ratio": decode_ratio,
        "augmentation_strategies": augmentation_strategies,
    }
    return (
        collated_batch_to_output(
            **input_args,
        ),
        collated_batch_to_output(
            **input_args,
        ),
        collated_batch_to_output(
            **input_args,
        ),
        collated_batch_to_output(
            **input_args,
        ),
    )
