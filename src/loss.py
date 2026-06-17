### Original Code:
### Copyright (c) 2024 Presto Authors
### Licensed under the MIT License.
### A copy of the MIT License is available in the LICENSE file in the root directory of this project.

### Modifications by marlens123:
### - Included medium and low resolution data

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from src.data.config import NO_DATA_VALUE
from src.data.earthengine.eo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
)


def mse_loss(
    t_s_t_h,
    t_s_t_m,
    t_s_t_l,
    t_sp,
    t_t,
    t_st,
    p_s_t_h,
    p_s_t_m,
    p_s_t_l,
    p_sp,
    p_t,
    p_st,
    s_t_h_m,
    s_t_m_m,
    s_t_l_m,
    sp_m,
    t_m,
    st_m,
):
    """
    Computes MSE loss between predicted and target values, only for pixels to be decoded (where mask value is 2).
    """
    encoder_size = t_s_t_h.shape[-1]
    expanded_s_t_h_m = repeat(s_t_h_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_s_t_m_m = repeat(s_t_m_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_s_t_l_m = repeat(s_t_l_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_sp_m = repeat(sp_m, "b h w c_g -> b h w c_g d", d=encoder_size)
    expanded_t_m = repeat(t_m, "b t c_g -> b t c_g d", d=encoder_size)
    expanded_st_m = repeat(st_m, "b c_g -> b c_g d", d=encoder_size)
    return F.mse_loss(
        torch.concat(
            [
                p_s_t_h[expanded_s_t_h_m == 2],
                p_s_t_m[expanded_s_t_m_m == 2],
                p_s_t_l[expanded_s_t_l_m == 2],
                p_sp[expanded_sp_m == 2],
                p_t[expanded_t_m == 2],
                p_st[expanded_st_m == 2],
            ]
        ),
        torch.concat(
            [
                t_s_t_h[expanded_s_t_h_m == 2],
                t_s_t_m[expanded_s_t_m_m == 2],
                t_s_t_l[expanded_s_t_l_m == 2],
                t_sp[expanded_sp_m == 2],
                t_t[expanded_t_m == 2],
                t_st[expanded_st_m == 2],
            ]
        ).float(),
    )


def mae_loss(
    p_s_t_h,
    p_s_t_m,
    p_s_t_l,
    p_sp,
    p_t,
    p_st,
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
    patch_size_high_res,
    patch_size_med_res,
    patch_size_low_res,
):
    """
    Computes MAE loss between predicted and target values.
    To this end, convert the predicted values (which are in token format) into the same pixel-wise shape as the target values,
    and then compute MAE only for pixels to be decoded (where mask value is 2).
    """

    assert not torch.isnan(p_s_t_h).any(), "p_s_t_h contains NaN!"
    assert not torch.isnan(s_t_h_x).any(), "s_t_h_x contains NaN!"
    assert not torch.isnan(p_s_t_m).any(), "p_s_t_m contains NaN!"
    assert not torch.isnan(s_t_m_x).any(), "s_t_m_x contains NaN!"
    assert not torch.isnan(p_s_t_l).any(), "p_s_t_l contains NaN!"
    assert not torch.isnan(s_t_l_x).any(), "s_t_l_x contains NaN!"
    assert not torch.isnan(p_sp).any(), "p_sp contains NaN!"
    assert not torch.isnan(sp_x).any(), "sp_x contains NaN!"
    assert not torch.isnan(p_t).any(), "p_t contains NaN!"
    assert not torch.isnan(t_x).any(), "t_x contains NaN!"
    assert not torch.isnan(p_st).any(), "p_st contains NaN!"
    assert not torch.isnan(st_x).any(), "st_x contains NaN!"
    assert not torch.isnan(sp_m).any(), "sp_m contains NaN!"
    assert not torch.isnan(s_t_h_m).any(), "s_t_h_m contains NaN!"
    assert not torch.isnan(s_t_m_m).any(), "s_t_m_m contains NaN!"
    assert not torch.isnan(s_t_l_m).any(), "s_t_l_m contains NaN!"
    assert not torch.isnan(t_m).any(), "t_m contains NaN!"
    assert not torch.isnan(st_m).any(), "st_m contains NaN!"

    # Bring tokens from channel group format into channel-wise format
    SPACE_TIME_HIGH_RES_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    SPACE_TIME_MED_RES_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    SPACE_TIME_LOW_RES_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    SPACE_BAND_EXPANSION = torch.tensor(
        [len(x) for x in SPACE_BAND_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    TIME_BAND_EXPANSION = torch.tensor(
        [len(x) for x in TIME_BANDS_GROUPS_IDX.values()], device=sp_m.device
    ).long()
    STATIC_BAND_EXPANSION = torch.tensor(
        [len(x) for x in STATIC_BAND_GROUPS_IDX.values()], device=sp_m.device
    ).long()

    pixel_s_t_h_m = torch.repeat_interleave(
        s_t_h_m, repeats=SPACE_TIME_HIGH_RES_BAND_EXPANSION, dim=-1
    )
    pixel_s_t_m_m = torch.repeat_interleave(
        s_t_m_m, repeats=SPACE_TIME_MED_RES_BAND_EXPANSION, dim=-1
    )
    pixel_s_t_l_m = torch.repeat_interleave(
        s_t_l_m, repeats=SPACE_TIME_LOW_RES_BAND_EXPANSION, dim=-1
    )
    pixel_sp_m = torch.repeat_interleave(sp_m, repeats=SPACE_BAND_EXPANSION, dim=-1)
    pixel_st_m = torch.repeat_interleave(st_m, repeats=STATIC_BAND_EXPANSION, dim=-1)
    pixel_t_m = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION, dim=-1)

    # bring tokens from patch-wise into pixel-wise and channel-wise format, channel-group-wise
    output_p_s_t_h = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.items()):
        channel_group_p_s_t_h = p_s_t_h[:, :, :, :, idx, : ((patch_size_high_res**2) * len(c_g))]
        channel_group_p_s_t_h = rearrange(
            channel_group_p_s_t_h,
            "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
            c_g=len(c_g),
            p_w=patch_size_high_res,
            p_h=patch_size_high_res,
        )
        output_p_s_t_h.append(channel_group_p_s_t_h)

    output_p_s_t_m = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.items()):
        assert patch_size_med_res == 1, "patch_size_med_res != 1 not implemented yet"
        channel_group_p_s_t_m = p_s_t_m[:, :, :, :, idx, : len(c_g)]
        output_p_s_t_m.append(channel_group_p_s_t_m)

    output_p_s_t_l = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.items()):
        assert patch_size_low_res == 1, "patch_size_low_res != 1 not implemented yet"
        channel_group_p_s_t_l = p_s_t_l[:, :, :, :, idx, : len(c_g)]
        output_p_s_t_l.append(channel_group_p_s_t_l)

    output_p_sp = []
    for idx, (_, c_g) in enumerate(SPACE_BAND_GROUPS_IDX.items()):
        channel_group_p_sp = p_sp[:, :, :, idx, : ((patch_size_high_res**2) * len(c_g))]
        channel_group_p_sp = rearrange(
            channel_group_p_sp,
            "b t_h t_w (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) c_g",
            c_g=len(c_g),
            p_w=patch_size_high_res,
            p_h=patch_size_high_res,
        )
        output_p_sp.append(channel_group_p_sp)

    output_p_t = []
    for idx, (_, c_g) in enumerate(TIME_BANDS_GROUPS_IDX.items()):
        channel_group_p_t = p_t[:, :, idx, : len(c_g)]
        output_p_t.append(channel_group_p_t)

    output_p_st = []
    for idx, (_, c_g) in enumerate(STATIC_BAND_GROUPS_IDX.items()):
        channel_group_st_t = p_st[:, idx, : len(c_g)]
        output_p_st.append(channel_group_st_t)

    assert output_p_s_t_h, "output_p_s_t_h is empty"
    assert output_p_s_t_m, "output_p_s_t_m is empty"
    assert output_p_s_t_l, "output_p_s_t_l is empty"
    assert output_p_t, "output_p_t is empty"
    assert output_p_sp, "output_p_sp is empty"
    assert output_p_st, "output_p_st is empty"

    # concatenate the channels of each data group back together
    # these now have the same shape as s_t_x, etc.
    p_s_t_h = torch.cat(output_p_s_t_h, dim=-1)
    p_s_t_m = torch.cat(output_p_s_t_m, dim=-1)
    p_s_t_l = torch.cat(output_p_s_t_l, dim=-1)
    p_sp = torch.cat(output_p_sp, dim=-1)
    p_st = torch.cat(output_p_st, dim=-1)
    p_t = torch.cat(output_p_t, dim=-1)

    assert not (p_s_t_h[pixel_s_t_h_m == 2] == NO_DATA_VALUE).any()
    assert not (p_s_t_m[pixel_s_t_m_m == 2] == NO_DATA_VALUE).any()
    assert not (p_s_t_l[pixel_s_t_l_m == 2] == NO_DATA_VALUE).any()
    assert not (p_sp[pixel_sp_m == 2] == NO_DATA_VALUE).any()
    assert not (p_t[pixel_t_m == 2] == NO_DATA_VALUE).any()
    assert not (p_st[pixel_st_m == 2] == NO_DATA_VALUE).any()

    assert not (s_t_h_x[pixel_s_t_h_m == 2] == NO_DATA_VALUE).any()
    assert not (s_t_m_x[pixel_s_t_m_m == 2] == NO_DATA_VALUE).any()
    assert not (s_t_l_x[pixel_s_t_l_m == 2] == NO_DATA_VALUE).any()
    assert not (sp_x[pixel_sp_m == 2] == NO_DATA_VALUE).any()
    assert not (t_x[pixel_t_m == 2] == NO_DATA_VALUE).any()
    assert not (st_x[pixel_st_m == 2] == NO_DATA_VALUE).any()

    return F.smooth_l1_loss(
        torch.concat(
            [
                p_s_t_h[pixel_s_t_h_m == 2],
                p_s_t_m[pixel_s_t_m_m == 2],
                p_s_t_l[pixel_s_t_l_m == 2],
                p_sp[pixel_sp_m == 2],
                p_t[pixel_t_m == 2],
                p_st[pixel_st_m == 2],
            ]
        ),
        torch.concat(
            [
                s_t_h_x[pixel_s_t_h_m == 2],
                s_t_m_x[pixel_s_t_m_m == 2],
                s_t_l_x[pixel_s_t_l_m == 2],
                sp_x[pixel_sp_m == 2],
                t_x[pixel_t_m == 2],
                st_x[pixel_st_m == 2],
            ]
        ),
    )


def do_loss(config, loss_inputs):
    if config["loss_type"] == "mse":
        loss = mse_loss(*loss_inputs)
    elif config["loss_type"] == "MAE":
        loss = mae_loss(*loss_inputs)
    else:
        raise f"loss_type must be MAE or mse, not {config['loss_type']}"

    return loss
