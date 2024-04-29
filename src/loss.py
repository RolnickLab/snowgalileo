import torch
import torch.nn.functional as F
from einops import rearrange

from .data import SPACE_BAND_GROUPS_IDX, SPACE_TIME_BANDS_GROUPS_IDX, TIME_BAND_GROUPS_IDX


def patchify_and_concat_space(space_time_array, space_only_array, patch_size):
    space_time_array = rearrange(
        space_time_array,
        "b (t_h p_h) (t_w p_w) t c -> b t_h t_w (p_h p_w t c)",
        p_h=patch_size,
        p_w=patch_size,
    )
    space_only_array = rearrange(
        space_only_array,
        "b (t_h p_h) (t_w p_w) c -> b t_h t_w (p_h p_w c)",
        p_h=patch_size,
        p_w=patch_size,
    )
    return torch.concat([space_time_array, space_only_array], dim=-1)


def group_channels(space_time_array, space_only_array, time_only_array):
    s_t_c_g, s_c_g, t_c_g = [], [], []
    for _, channel_idxs in SPACE_TIME_BANDS_GROUPS_IDX.items():
        s_t_c_g.append(space_time_array[:, :, :, :, channel_idxs])
    for _, channel_idxs in SPACE_BAND_GROUPS_IDX.items():
        s_c_g.append(space_only_array[:, :, :, channel_idxs])
    for _, channel_idxs in TIME_BAND_GROUPS_IDX.items():
        t_c_g.append(time_only_array[:, :, channel_idxs])

    s_t_c_g = torch.concat([rearrange(x, "b h w t c_g -> b c_g (h w t)") for x in s_t_c_g], dim=-2)
    s_c_g = torch.concat([rearrange(x, "b h w c_g -> b c_g (h w)") for x in s_c_g], dim=-2)
    t_c_g = torch.concat([rearrange(x, "b t c_g -> b c_g t") for x in t_c_g], dim=-2)

    return (s_t_c_g, s_c_g, t_c_g)


def norm_per_c_g_loss(
    expanded_s_t_x,
    expanded_s_x,
    t_x,
    p_s_t,
    p_s,
    p_t,
    expanded_s_t,
    expanded_s,
    expanded_t,
):
    x_s_t_c_g, x_s_c_g, x_t_c_g = group_channels(expanded_s_t_x, expanded_s_x, t_x)
    pred_s_t_c_g, pred_s_c_g, pred_t_c_g = group_channels(p_s_t, p_s, p_t)
    mask_s_t_c_g, mask_s_c_g, mask_t_c_g = group_channels(expanded_s_t, expanded_s, expanded_t)

    # normalize the targets per channel group
    norm_s_t_x_c_g = (x_s_t_c_g - x_s_t_c_g.mean(dim=-1, keepdim=True)) / (
        x_s_t_c_g.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5
    norm_x_s_c_g = (x_s_c_g - x_s_c_g.mean(dim=-1, keepdim=True)) / (
        x_s_c_g.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5
    norm_x_t_c_g = (x_t_c_g - x_t_c_g.mean(dim=-1, keepdim=True)) / (
        x_t_c_g.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5

    return F.mse_loss(
        torch.concat(
            [norm_s_t_x_c_g[mask_s_t_c_g], norm_x_s_c_g[mask_s_c_g], norm_x_t_c_g[mask_t_c_g]]
        ),
        torch.concat(
            [pred_s_t_c_g[mask_s_t_c_g], pred_s_c_g[mask_s_c_g], pred_t_c_g[mask_t_c_g]]
        ).float(),
    )


def norm_per_patch_loss(
    expanded_s_t_x,
    expanded_s_x,
    t_x,
    p_s_t,
    p_s,
    p_t,
    expanded_s_t,
    expanded_s,
    expanded_t,
    patch_size,
):
    """
    If true, returns norm pix loss
    If false, returns MSE loss
    Inspired by: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L198
    """
    x_per_patch = patchify_and_concat_space(expanded_s_t_x, expanded_s_x, patch_size)
    pred_per_patch = patchify_and_concat_space(p_s_t, p_s, patch_size)
    mask_per_patch = patchify_and_concat_space(expanded_s_t, expanded_s, patch_size)

    # normalize the target per patch for dynamic in space variables
    norm_x_per_patch = (x_per_patch - x_per_patch.mean(dim=-1, keepdim=True)) / (
        x_per_patch.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5

    # concatenate with time only variables
    return F.mse_loss(
        torch.concat([pred_per_patch[mask_per_patch], p_t[expanded_t]]),
        torch.concat([norm_x_per_patch[mask_per_patch], t_x[expanded_t]]).float(),
    )


def mse_loss(
    expanded_s_t_x,
    expanded_s_x,
    t_x,
    p_s_t,
    p_s,
    p_t,
    expanded_s_t,
    expanded_s,
    expanded_t,
):
    return F.mse_loss(
        torch.concat([p_s_t[expanded_s_t], p_s[expanded_s], p_t[expanded_t]]),
        torch.concat(
            [expanded_s_t_x[expanded_s_t], expanded_s_x[expanded_s], t_x[expanded_t]]
        ).float(),
    )
