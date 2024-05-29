import torch
import torch.nn.functional as F
from einops import rearrange

from .data import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)


def group_per_patch(space_time_array, space_only_array, patch_size):
    return (
        rearrange(
            space_time_array,
            "b (t_h p_h) (t_w p_w) t c -> b t_h t_w (p_h p_w t c)",
            p_h=patch_size,
            p_w=patch_size,
        ),
        rearrange(
            space_only_array,
            "b (t_h p_h) (t_w p_w) c -> b t_h t_w (p_h p_w c)",
            p_h=patch_size,
            p_w=patch_size,
        ),
    )


def group_per_channel(space_time_array, space_only_array, time_only_array, static_array):
    return (
        rearrange(
            space_time_array,
            "b h w t c -> b c (h w t)",
        ),
        rearrange(
            space_only_array,
            "b h w c -> b c (h w)",
        ),
        rearrange(
            time_only_array,
            "b t c -> b c t",
        ),
        static_array,
    )


def normalize(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True) + 1.0e-6) ** 0.5


def mse_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
):
    return F.mse_loss(
        torch.concat(
            [p_s_t[expanded_s_t_m], p_sp[expanded_sp_m], p_t[expanded_t_m], p_st[expanded_st_m]]
        ),
        torch.concat(
            [
                expanded_s_t_x[expanded_s_t_m],
                expanded_sp_x[expanded_sp_m],
                t_x[expanded_t_m],
                st_x[expanded_st_m],
            ]
        ).float(),
    )


def norm_per_channel_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
):
    """
    MSE loss with target normalization per channel.
    """
    expanded_s_t_x, expanded_sp_x, t_x, st_x = group_per_channel(
        expanded_s_t_x, expanded_sp_x, t_x, st_x
    )
    p_s_t, p_sp, p_t, p_st = group_per_channel(p_s_t, p_sp, p_t, p_st)
    expanded_s_t_m, expanded_sp_m, expanded_t_m = group_per_channel(
        expanded_s_t_m,
        expanded_sp_m,
        expanded_t_m,
        expanded_st_m,
    )

    # normalize the targets per channel
    norm_expanded_s_t_x = normalize(expanded_s_t_x)
    norm_expanded_sp_x = normalize(expanded_sp_x)
    norm_t_x = normalize(t_x)
    norm_st_x = normalize(st_x)

    return mse_loss(
        norm_expanded_s_t_x,
        norm_expanded_sp_x,
        norm_t_x,
        norm_st_x,
        p_s_t,
        p_sp,
        p_t,
        p_st,
        expanded_s_t_m,
        expanded_sp_m,
        expanded_t_m,
        expanded_st_m,
    )


def norm_per_c_g_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
):
    """
    MSE loss with target normalization per channel group.
    """
    target_s_t_l, target_sp_l, target_t_l, target_st_l, p_s_t_l, p_sp_l, p_t_l, p_st_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # group, normalize, and mask arrays per channel group
    for _, channel_idxs in SPACE_TIME_BANDS_GROUPS_IDX.items():
        norm_s_t_x_c_g = normalize(
            rearrange((expanded_s_t_x[:, :, :, :, channel_idxs]), "b h w t c -> b (c h w t)")
        )
        s_t_m_c_g = rearrange(
            (expanded_s_t_m[:, :, :, :, channel_idxs]), "b h w t c -> b (c h w t)"
        )

        p_s_t_l.append(
            rearrange((p_s_t[:, :, :, :, channel_idxs]), "b h w t c -> b (c h w t)")[s_t_m_c_g]
        )
        target_s_t_l.append(norm_s_t_x_c_g[s_t_m_c_g])

    for _, channel_idxs in SPACE_BAND_GROUPS_IDX.items():
        norm_sp_x_c_g = normalize(
            rearrange((expanded_sp_x[:, :, :, channel_idxs]), "b h w c -> b (c h w)")
        )
        sp_m_c_g = rearrange((expanded_sp_m[:, :, :, channel_idxs]), "b h w c -> b (c h w)")

        p_sp_l.append(rearrange((p_sp[:, :, :, channel_idxs]), "b h w c -> b (c h w)")[sp_m_c_g])
        target_sp_l.append(norm_sp_x_c_g[sp_m_c_g])

    for _, channel_idxs in TIME_BAND_GROUPS_IDX.items():
        norm_t_x_c_g = normalize(rearrange((t_x[:, :, channel_idxs]), "b t c -> b (c t)"))
        t_m_c_g = rearrange((expanded_t_m[:, :, channel_idxs]), "b t c -> b (c t)")

        p_t_l.append(rearrange((p_t[:, :, channel_idxs]), "b t c -> b (c t)")[t_m_c_g])
        target_t_l.append(norm_t_x_c_g[t_m_c_g])

    for _, channel_idxs in STATIC_BAND_GROUPS_IDX.items():
        norm_st_x_c_g = normalize(st_x[:, :, channel_idxs])
        st_m_c_g = expanded_st_m[:, :, channel_idxs]

        p_st_l.append(p_st[:, :, channel_idxs][st_m_c_g])
        target_st_l.append(norm_st_x_c_g[st_m_c_g])

    return mse_loss(
        torch.concat(target_s_t_l),
        torch.concat(target_sp_l),
        torch.concat(target_t_l),
        torch.concat(target_st_l),
        torch.concat(p_s_t_l),
        torch.concat(p_sp_l),
        torch.concat(p_t_l),
        torch.concat(p_st_l),
        # mask has already been applied, so unmask everything for the mse loss
        torch.ones_like(torch.concat(target_s_t_l)).bool(),
        torch.ones_like(torch.concat(target_sp_l)).bool(),
        torch.ones_like(torch.concat(target_t_l)).bool(),
        torch.ones_like(torch.concat(target_st_l)).bool(),
    )


def norm_per_timestep_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
):
    """
    MSE loss with target normalization per timestep.
    """
    # group s_t arrays per timestep
    # time only arrays don't need to be rearranged because already in shape (b, t, c)
    expanded_s_t_x = rearrange(expanded_s_t_x, "b h w t c -> b t (h w c)")
    p_s_t = rearrange(p_s_t, "b h w t c -> b t (h w c)")
    expanded_s_t_m = rearrange(expanded_s_t_m, "b h w t c -> b t (h w c)")

    # normalize the targets per timestep
    norm_expanded_s_t_x = normalize(expanded_s_t_x)
    norm_t_x = normalize(t_x)

    return mse_loss(
        norm_expanded_s_t_x,
        expanded_sp_x,
        norm_t_x,
        st_x,
        p_s_t,
        p_sp,
        p_t,
        p_st,
        expanded_s_t_m,
        expanded_sp_m,
        expanded_t_m,
        expanded_st_m,
    )


def norm_per_patch_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
    patch_size,
):
    """
    MSE loss with target normalization per patch.
    Inspired by: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L198
    """
    # group variable in space arrays per patch
    expanded_s_t_x, expanded_sp_x = group_per_patch(expanded_s_t_x, expanded_sp_x, patch_size)
    p_s_t, p_sp = group_per_patch(p_s_t, p_sp, patch_size)
    expanded_s_t_m, expanded_sp_m = group_per_patch(expanded_s_t_m, expanded_sp_m, patch_size)

    # normalize the targets per patch
    norm_expanded_s_t_x = (expanded_s_t_x - expanded_s_t_x.mean(dim=-1, keepdim=True)) / (
        expanded_s_t_x.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5
    norm_expanded_sp_x = (expanded_sp_x - expanded_sp_x.mean(dim=-1, keepdim=True)) / (
        expanded_sp_x.var(dim=-1, keepdim=True) + 1.0e-6
    ) ** 0.5

    return mse_loss(
        norm_expanded_s_t_x,
        norm_expanded_sp_x,
        t_x,
        st_x,
        p_s_t,
        p_sp,
        p_t,
        p_st,
        expanded_s_t_m,
        expanded_sp_m,
        expanded_t_m,
        expanded_st_m,
    )


def mae_loss(
    expanded_s_t_x,
    expanded_sp_x,
    t_x,
    st_x,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    expanded_s_t_m,
    expanded_sp_m,
    expanded_t_m,
    expanded_st_m,
    patch_size,
    loss_type,
):
    assert loss_type in [
        "mse",
        "norm_per_patch",
        "norm_per_c_g",
        "norm_per_channel",
        "norm_per_timestep",
    ]

    if loss_type == "norm_per_patch":
        return norm_per_patch_loss(
            expanded_s_t_x,
            expanded_sp_x,
            t_x,
            st_x,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            expanded_s_t_m,
            expanded_sp_m,
            expanded_t_m,
            expanded_st_m,
            patch_size=patch_size,
        )
    elif loss_type == "norm_per_c_g":
        return norm_per_c_g_loss(
            expanded_s_t_x,
            expanded_sp_x,
            t_x,
            st_x,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            expanded_s_t_m,
            expanded_sp_m,
            expanded_t_m,
            expanded_st_m,
        )
    elif loss_type == "norm_per_channel":
        return norm_per_channel_loss(
            expanded_s_t_x,
            expanded_sp_x,
            t_x,
            st_x,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            expanded_s_t_m,
            expanded_sp_m,
            expanded_t_m,
            expanded_st_m,
        )
    elif loss_type == "norm_per_timestep":
        return norm_per_timestep_loss(
            expanded_s_t_x,
            expanded_sp_x,
            t_x,
            st_x,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            expanded_s_t_m,
            expanded_sp_m,
            expanded_t_m,
            expanded_st_m,
        )
    else:
        return mse_loss(
            expanded_s_t_x,
            expanded_sp_x,
            t_x,
            st_x,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            expanded_s_t_m,
            expanded_sp_m,
            expanded_t_m,
            expanded_st_m,
        )
