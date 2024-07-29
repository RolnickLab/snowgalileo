from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import default_collate
from torchvision.transforms.functional import resize

from src.masking import (
    MASKING_MODES,
    SPACE_BAND_EXPANSION,
    SPACE_TIME_BAND_EXPANSION,
    STATIC_BAND_EXPANSION,
    TIME_BAND_EXPANSION,
    MaskingFunctions,
    batch_subset_mask_presto,
)


class CollateFnOutput(NamedTuple):
    s_t_x: torch.Tensor
    sp_x: torch.Tensor
    t_x: torch.Tensor
    st_x: torch.Tensor
    s_t_m: torch.Tensor
    sp_m: torch.Tensor
    t_m: torch.Tensor
    st_m: torch.Tensor
    months: torch.Tensor
    expanded_s_t_x: torch.Tensor
    expanded_sp_x: torch.Tensor
    expanded_s_t: torch.Tensor
    expanded_sp: torch.Tensor
    expanded_t: torch.Tensor
    expanded_st: torch.Tensor
    patch_size: float
    c_i: Optional[Dict]


def collated_batch_to_output(
    s_t_x: torch.Tensor,
    sp_x: torch.Tensor,
    t_x: torch.Tensor,
    st_x: torch.Tensor,
    months: torch.Tensor,
    patch_sizes,
    shape_time_combinations,
    mask_ratio,
    decoder_unmask_ratio,
    masking_function: MaskingFunctions,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
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
    timesteps = space_time_combination["timesteps"]

    image_size = patch_size * spatial_patches_per_dim
    if masking_probabilities is None:
        masking_probabilities = [1] * len(MASKING_MODES)

    # randomly select a masking strategy
    (s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months), c_i = batch_subset_mask_presto(
        s_t_x,
        sp_x,
        t_x,
        st_x,
        months,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        image_size=image_size,
        num_timesteps=timesteps,
        decoder_unmask_ratio=decoder_unmask_ratio,
        augmentation_strategies=augmentation_strategies,
        masking_probabilities=masking_probabilities,
        masking_function=masking_function,
    )

    # transform the masks from channel-groups to individual channels
    expanded_s_t = torch.repeat_interleave(
        s_t_m, repeats=SPACE_TIME_BAND_EXPANSION.long(), dim=-1
    ).int()
    expanded_sp = torch.repeat_interleave(sp_m, repeats=SPACE_BAND_EXPANSION.long(), dim=-1).int()
    expanded_t = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION.long(), dim=-1).int()
    expanded_st = torch.repeat_interleave(st_m, repeats=STATIC_BAND_EXPANSION.long(), dim=-1).int()

    # p_s_t and p_sp always assume the maximum patch size, so we need to
    # resample if its smaller
    if patch_size < patch_sizes[-1]:
        output_hw = spatial_patches_per_dim * patch_sizes[-1]
        t, d = s_t_x.shape[3], s_t_x.shape[4]
        expanded_s_t_x = rearrange(
            resize(
                rearrange(s_t_x, "b h w t d -> b (t d) h w"),
                size=(output_hw, output_hw),
            ),
            "b (t d) h w -> b h w t d",
            t=t,
            d=d,
        )
        expanded_sp_x = rearrange(
            resize(rearrange(sp_x, "b h w d -> b d h w"), size=(output_hw, output_hw)),
            "b d h w -> b h w d",
        )

        # fix the mask too
        expanded_s_t = repeat(
            expanded_s_t[:, 0::patch_size, 0::patch_size],
            "b h w t c -> b (h h2) (w w2) t c",
            h2=patch_sizes[-1],
            w2=patch_sizes[-1],
        )

        expanded_sp = repeat(
            expanded_sp[:, 0::patch_size, 0::patch_size],
            "b h w c -> b (h h2) (w w2) c",
            h2=patch_sizes[-1],
            w2=patch_sizes[-1],
        )
    else:
        expanded_s_t_x = s_t_x
        expanded_sp_x = sp_x

    return CollateFnOutput(
        s_t_x,
        sp_x,
        t_x,
        st_x,
        s_t_m,
        sp_m,
        t_m,
        st_m,
        months,
        expanded_s_t_x,
        expanded_sp_x,
        expanded_s_t,
        expanded_sp,
        expanded_t,
        expanded_st,
        patch_size,
        c_i,
    )


@torch.no_grad()
def mae_collate_fn(
    batch,
    patch_sizes,
    shape_time_combinations,
    mask_ratio,
    decoder_unmask_ratio,
    augmentation_strategies=None,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
    masking_probabilities=None,
) -> Tuple[CollateFnOutput, CollateFnOutput, CollateFnOutput]:
    s_t_x, sp_x, t_x, st_x, months = default_collate(batch)

    return (
        collated_batch_to_output(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            months,
            patch_sizes,
            shape_time_combinations,
            mask_ratio,
            decoder_unmask_ratio,
            MaskingFunctions.RANDOM,
            augmentation_strategies,
            fixed_patch_size,
            fixed_space_time_combination,
            masking_probabilities,
        ),
        collated_batch_to_output(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            months,
            patch_sizes,
            shape_time_combinations,
            mask_ratio,
            decoder_unmask_ratio,
            MaskingFunctions.SPACE,
            augmentation_strategies,
            fixed_patch_size,
            fixed_space_time_combination,
            masking_probabilities,
        ),
        collated_batch_to_output(
            s_t_x,
            sp_x,
            t_x,
            st_x,
            months,
            patch_sizes,
            shape_time_combinations,
            mask_ratio,
            decoder_unmask_ratio,
            MaskingFunctions.TIME,
            augmentation_strategies,
            fixed_patch_size,
            fixed_space_time_combination,
            masking_probabilities,
        ),
    )
