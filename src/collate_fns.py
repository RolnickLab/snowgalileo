import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import default_collate
from torchvision.transforms.functional import resize

from src.masking import (
    SPACE_BAND_EXPANSION,
    SPACE_TIME_BAND_EXPANSION,
    TIME_BAND_EXPANSION,
    batch_mask_presto,
    subset_batch_of_images,
)


@torch.no_grad()
def mae_collate_fn(
    batch,
    patch_sizes,
    shape_time_combinations,
    mask_ratio,
    time_ratio,
    space_ratio,
    channel_ratio,
    fixed_patch_size=None,
    fixed_space_time_combination=None,
):
    s_t_x, s_x, t_x, months = default_collate(batch)

    if fixed_patch_size is not None:
        patch_size = fixed_patch_size
    else:
        # randomly sample a patch size, and a corresponding image size
        patch_size = np.random.choice(patch_sizes)

    if fixed_space_time_combination is not None:
        space_time_combination = shape_time_combinations
    else:
        space_time_combination = np.random.choice(patch_sizes)

    spatial_patches_per_dim = space_time_combination["size"]
    timesteps = space_time_combination["timesteps"]

    image_size = patch_size * spatial_patches_per_dim
    s_t_x, s_x, t_x, months = subset_batch_of_images(
        s_t_x, s_x, t_x, months, size=image_size, num_timesteps=timesteps
    )
    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = batch_mask_presto(
        s_t_x,
        s_x,
        t_x,
        months,
        mask_ratio,
        patch_size,
        time_ratio,
        space_ratio,
        channel_ratio,
    )

    # transform the masks from channel-groups to individual channels
    expanded_s_t = torch.repeat_interleave(
        s_t_m, repeats=SPACE_TIME_BAND_EXPANSION.long(), dim=-1
    ).bool()
    expanded_s = torch.repeat_interleave(s_m, repeats=SPACE_BAND_EXPANSION.long(), dim=-1).bool()
    expanded_t = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION.long(), dim=-1).bool()

    # p_s_t and p_s always assume the maximum patch size, so we need to
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
        expanded_s_x = rearrange(
            resize(rearrange(s_x, "b h w d -> b d h w"), size=(output_hw, output_hw)),
            "b d h w -> b h w d",
        )

        # fix the mask too
        expanded_s_t = repeat(
            expanded_s_t[:, 0::patch_size, 0::patch_size],
            "b h w t c -> b (h h2) (w w2) t c",
            h2=patch_sizes[-1],
            w2=patch_sizes[-1],
        )

        expanded_s = repeat(
            expanded_s[:, 0::patch_size, 0::patch_size],
            "b h w c -> b (h h2) (w w2) c",
            h2=patch_sizes[-1],
            w2=patch_sizes[-1],
        )
    else:
        expanded_s_t_x = s_t_x
        expanded_s_x = s_x

    return (
        s_t_x,
        s_x,
        t_x,
        s_t_m,
        s_m,
        t_m,
        months,
        expanded_s_t_x,
        expanded_s_x,
        expanded_s_t,
        expanded_s,
        expanded_t,
        patch_size,
    )
