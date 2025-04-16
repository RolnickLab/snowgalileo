from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import default_collate

from src.masking import (
    MASKING_MODES,
    MaskingFunctions,
    batch_subset_mask_presto,
)


class CollateFnOutput(NamedTuple):
    s_t_h_x: torch.Tensor
    sp_x: torch.Tensor
    t_x: torch.Tensor
    st_x: torch.Tensor
    s_t_h_m: torch.Tensor
    sp_m: torch.Tensor
    t_m: torch.Tensor
    st_m: torch.Tensor
    months: torch.Tensor
    patch_size: float
    c_i: Optional[Dict]


def collated_batch_to_output(
    s_t_h_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    valid_data_mask_s_t_h: torch.Tensor,
    valid_data_mask_sp: torch.Tensor,
    valid_data_mask_t: torch.Tensor,
    valid_data_mask_st: torch.Tensor,
    patch_sizes,
    shape_time_combinations,
    encode_ratio,
    decode_ratio,
    masking_function: MaskingFunctions,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
    max_unmasking_channels=4,
    unmasking_channels_combo: str = "shapes",
) -> CollateFnOutput:
    if fixed_patch_size is not None:
        patch_size = fixed_patch_size
    else:
        # randomly sample a patch size, and a corresponding image size
        patch_size = np.random.choice(patch_sizes)

    if fixed_space_time_combination is not None:
        space_time_combination = fixed_space_time_combination
    else:
        space_time_combination = np.random.choice(shape_time_combinations)
        spatial_patches_per_dim = space_time_combination["size"]
        if int(spatial_patches_per_dim * patch_size) > s_t_h_x.shape[1]:
            spatial_patches_per_dim = int(s_t_h_x.shape[1] / patch_size)

    timesteps = space_time_combination["timesteps"]

    image_size = patch_size * spatial_patches_per_dim
    if masking_probabilities is None:
        masking_probabilities = [1] * len(MASKING_MODES)

    # randomly select a masking strategy
    (
        (
            s_t_h_x,
            sp_x,
            t_x,
            st_x,
            s_t_h_m,
            sp_m,
            t_m,
            st_m,
            months,
        ),
        c_i,
    ) = batch_subset_mask_presto(
        s_t_h_x,
        sp_x,
        t_x,
        st_x,
        months,
        valid_data_mask_s_t_h,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
        encode_ratio=encode_ratio,
        patch_size=patch_size,
        image_size=image_size,
        num_timesteps=timesteps,
        decode_ratio=decode_ratio,
        augmentation_strategies=augmentation_strategies,
        masking_probabilities=masking_probabilities,
        masking_function=masking_function,
        max_unmasking_channels=max_unmasking_channels,
        unmasking_channels_combo=unmasking_channels_combo,
    )

    return CollateFnOutput(
        s_t_h_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        sp_m,
        t_m,
        st_m,
        months,
        patch_size,
        c_i,
    )


@torch.no_grad()
def mae_collate_fn(
    batch,
    patch_sizes,
    shape_time_combinations,
    encode_ratio,
    decode_ratio,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
    max_unmasking_channels=4,
    random_masking: str = "None",
    unmasking_channels_combo: str = "shapes",
) -> Tuple[CollateFnOutput, CollateFnOutput, CollateFnOutput, CollateFnOutput]:
    (
        s_t_h_x,
        sp_x,
        t_x,
        st_x,
        months,
        valid_data_mask_s_t_h,
        valid_data_mask_sp,
        valid_data_mask_t,
        valid_data_mask_st,
    ) = default_collate(batch)

    input_args = {
        "s_t_h_x": s_t_h_x,
        "sp_x": sp_x,
        "t_x": t_x,
        "st_x": st_x,
        "months": months,
        "valid_data_mask_s_t_h": valid_data_mask_s_t_h,
        "valid_data_mask_sp": valid_data_mask_sp,
        "valid_data_mask_t": valid_data_mask_t,
        "valid_data_mask_st": valid_data_mask_st,
        "patch_sizes": patch_sizes,
        "encode_ratio": encode_ratio,
        "decode_ratio": decode_ratio,
        "augmentation_strategies": augmentation_strategies,
        "fixed_patch_size": fixed_patch_size,
        "fixed_space_time_combination": fixed_space_time_combination,
        "masking_probabilities": masking_probabilities,
        "shape_time_combinations": shape_time_combinations,
        "max_unmasking_channels": max_unmasking_channels,
        "unmasking_channels_combo": unmasking_channels_combo,
    }
    if random_masking == "none":
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.SPACE,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.SPACE,
            ),
        )
    elif random_masking == "half":
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.SPACE,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
        )
    elif random_masking == "time_only":
        print("only masks over time")
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.TIME,
            ),
        )
    elif random_masking == "full":
        return (
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
            collated_batch_to_output(
                **input_args,
                masking_function=MaskingFunctions.RANDOM,
            ),
        )
    else:
        raise ValueError(
            f"Expected random_masking to be one of none, half full, got {random_masking}"
        )
