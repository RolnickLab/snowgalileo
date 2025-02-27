from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.functional import resize

from src.data.earthengine.eo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
)

from src.data.config import (
    NO_DATA_VALUE
)


def construct_target_encoder_masks(
    s_t_h_m: torch.Tensor, s_t_m_m: torch.Tensor, s_t_l_m: torch.Tensor, sp_m: torch.Tensor, t_m: torch.Tensor, st_m: torch.Tensor, method: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "decoder_only":
        # we want 0s where the mask == 2
        return ~(s_t_h_m == 2), ~(s_t_m_m == 2), ~(s_t_l_m == 2), ~(sp_m == 2), ~(t_m == 2), ~(st_m == 2)
    elif method == "all":
        # we want all zeros
        return (
            torch.zeros_like(s_t_h_m),
            torch.zeros_like(s_t_m_m),
            torch.zeros_like(s_t_l_m),
            torch.zeros_like(sp_m),
            torch.zeros_like(t_m),
            torch.zeros_like(st_m),
        )
    elif method == "decoder_and_encoder":
        # we want 0s where the mask is not equal to 1
        return s_t_h_m == 1, s_t_m_m == 1, s_t_l_m == 1, sp_m == 1, t_m == 1, st_m == 1
    else:
        raise ValueError(f"Unexpected method {method}")


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
    # TODO: See if this is possible like this with the d
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


def seq_and_cat(s_t_h, s_t_m, s_t_l, sp, t, st):
    s_t_h = rearrange(s_t_h, "b h w t c_g d -> b (h w t c_g) d")
    s_t_m = rearrange(s_t_m, "b h w t c_g d -> b (h w t c_g) d")
    s_t_l = rearrange(s_t_l, "b h w t c_g d -> b (h w t c_g) d")
    sp = rearrange(sp, "b h w c_g d -> b (h w c_g) d")
    t = rearrange(t, "b t c_g d -> b (t c_g) d")
    # st is already a sequence
    return torch.cat([s_t_h, s_t_m, s_t_l, sp, t, st], dim=1)


def expand_and_reciprocate(t):
    reciprocals = torch.reciprocal(t.float())
    return torch.repeat_interleave(reciprocals, t)


def patch_disc_loss(
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
    mask_other_samples: bool,
    pred2unit: bool = True,
    tau: float = 0.2,
):
    # create tensors of shape (bsz, seq_len, dim)
    all_masks = seq_and_cat(
        s_t_h_m.unsqueeze(dim=-1),
        s_t_m_m.unsqueeze(dim=-1),
        s_t_l_m.unsqueeze(dim=-1),
        sp_m.unsqueeze(dim=-1),
        t_m.unsqueeze(dim=-1),
        st_m.unsqueeze(dim=-1),
    ).squeeze(-1)
    all_preds = seq_and_cat(p_s_t_h, p_s_t_m, p_s_t_l, p_sp, p_t, p_st)
    all_targets = seq_and_cat(t_s_t_h, t_s_t_m, t_s_t_l, t_sp, t_t, t_st)

    pred = all_preds[all_masks == 2].unsqueeze(dim=0)
    target = all_targets[all_masks == 2].unsqueeze(dim=0)

    bs, nt, d = pred.shape

    if pred2unit:
        pred_mu = pred.mean(1, keepdims=True)
        pred_std = pred.std(1, keepdims=True)
        pred = (pred - pred_mu) / (pred_std + 1e-4)

    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    scores = torch.einsum("npd,nqd->npq", pred, target) / tau
    count = (all_masks == 2).sum(dim=-1)

    if mask_other_samples:
        logit_mask = torch.full_like(scores, -torch.finfo(scores.dtype).max)
        start = 0
        for c in count:
            end = start + c
            logit_mask[:, start:end, start:end] = 0
            start += c

        scores = scores + logit_mask

    labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(bs, 1)
    loss = F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1), reduction="none") * (
        tau * 2
    )

    # emulate averaging across the batch dimension
    loss_multiplier = expand_and_reciprocate(count)
    loss = (loss * loss_multiplier).sum() / t_s_t_h.shape[0]
    return loss


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
    patch_size,
    max_patch_size,
):
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

    pixel_s_t_h_m = torch.repeat_interleave(s_t_h_m, repeats=SPACE_TIME_HIGH_RES_BAND_EXPANSION, dim=-1)
    pixel_s_t_m_m = torch.repeat_interleave(s_t_m_m, repeats=SPACE_TIME_MED_RES_BAND_EXPANSION, dim=-1)
    pixel_s_t_l_m = torch.repeat_interleave(s_t_l_m, repeats=SPACE_TIME_LOW_RES_BAND_EXPANSION, dim=-1)
    pixel_sp_m = torch.repeat_interleave(sp_m, repeats=SPACE_BAND_EXPANSION, dim=-1)
    pixel_st_m = torch.repeat_interleave(st_m, repeats=STATIC_BAND_EXPANSION, dim=-1)
    pixel_t_m = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION, dim=-1)

    output_p_s_t_h = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX.items()):
        channel_group_p_s_t_h = p_s_t_h[:, :, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_s_t_h = rearrange(
            channel_group_p_s_t_h,
            "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            assert s_t_h_x.shape[1] > 0 and s_t_h_x.shape[2] > 0, "s_t_h_x h and w are not > 0!"
            channel_group_p_s_t_h = rearrange(
                resize(
                    rearrange(channel_group_p_s_t_h, "b h w t d -> b (t d) h w"),
                    size=(s_t_h_x.shape[1], s_t_h_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=s_t_h_x.shape[3],
                d=len(c_g),
            )

        output_p_s_t_h.append(channel_group_p_s_t_h)

    output_p_s_t_m = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_MED_RES_BANDS_GROUPS_IDX.items()):
        channel_group_p_s_t_m = p_s_t_m[:, :, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_s_t_m = rearrange(
            channel_group_p_s_t_m,
            "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            assert s_t_m_x.shape[1] > 0 and s_t_m_x.shape[2] > 0, "s_t_m_x h and w are not > 0!"
            channel_group_p_s_t_m = rearrange(
                resize(
                    rearrange(channel_group_p_s_t_m, "b h w t d -> b (t d) h w"),
                    size=(s_t_m_x.shape[1], s_t_m_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=s_t_m_x.shape[3],
                d=len(c_g),
            )

        output_p_s_t_m.append(channel_group_p_s_t_m)

    output_p_s_t_l = []
    for idx, (_, c_g) in enumerate(SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX.items()):
        channel_group_p_s_t_l = p_s_t_l[:, :, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_s_t_l = rearrange(
            channel_group_p_s_t_l,
            "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            assert s_t_l_x.shape[1] > 0 and s_t_l_x.shape[2] > 0, "s_t_l_x h and w are not > 0!"
            channel_group_p_s_t_l = rearrange(
                resize(
                    rearrange(channel_group_p_s_t_l, "b h w t d -> b (t d) h w"),
                    size=(s_t_l_x.shape[1], s_t_l_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=s_t_l_x.shape[3],
                d=len(c_g),
            )

        output_p_s_t_l.append(channel_group_p_s_t_l)

    output_p_sp = []
    for idx, (_, c_g) in enumerate(SPACE_BAND_GROUPS_IDX.items()):
        channel_group_p_sp = p_sp[:, :, :, idx, : ((max_patch_size**2) * len(c_g))]
        channel_group_p_sp = rearrange(
            channel_group_p_sp,
            "b t_h t_w (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) c_g",
            c_g=len(c_g),
            p_w=max_patch_size,
            p_h=max_patch_size,
        )
        if patch_size < max_patch_size:
            channel_group_p_sp = rearrange(
                resize(
                    rearrange(channel_group_p_sp, "b h w d -> b d h w"),
                    size=(s_t_h_x.shape[1], s_t_h_x.shape[2]),
                ),
                "b d h w -> b h w d",
                d=len(c_g),
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

    # these now have the same shape as s_t_x, etc.
    p_s_t_h = torch.cat(output_p_s_t_h, dim=-1)
    p_s_t_m = torch.cat(output_p_s_t_m, dim=-1)
    p_s_t_l = torch.cat(output_p_s_t_l, dim=-1)
    p_sp = torch.cat(output_p_sp, dim=-1)
    p_st = torch.cat(output_p_st, dim=-1)
    p_t = torch.cat(output_p_t, dim=-1)

    print("p_s_t_h min/max:", p_s_t_h.min().item(), p_s_t_h.max().item())
    print("p_s_t_m min/max:", p_s_t_m.min().item(), p_s_t_m.max().item())
    print("p_s_t_l min/max:", p_s_t_l.min().item(), p_s_t_l.max().item())
    print("s_t_h_x min/max:", s_t_h_x.min().item(), s_t_h_x.max().item())
    print("s_t_m_x min/max:", s_t_m_x.min().item(), s_t_m_x.max().item())
    print("s_t_l_x min/max:", s_t_l_x.min().item(), s_t_l_x.max().item())
    print("p_sp min/max:", p_sp.min().item(), p_sp.max().item())
    print("sp_x min/max:", sp_x.min().item(), sp_x.max().item())
    print("p_t min/max:", p_t.min().item(), p_t.max().item())
    print("t_x min/max:", t_x.min().item(), t_x.max().item())
    print("p_st min/max:", p_st.min().item(), p_st.max().item())
    print("st_x min/max:", st_x.min().item(), st_x.max().item())
    print("s_t_h_m min/max:", s_t_h_m.min().item(), s_t_h_m.max().item())
    print("s_t_m_m min/max:", s_t_m_m.min().item(), s_t_m_m.max().item())
    print("s_t_l_m min/max:", s_t_l_m.min().item(), s_t_l_m.max().item())
    print("sp_m min/max:", sp_m.min().item(), sp_m.max().item())
    print("t_m min/max:", t_m.min().item(), t_m.max().item())
    print("st_m min/max:", st_m.min().item(), st_m.max().item())

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
    if config["loss_type"] == "patch_disc":
        loss = patch_disc_loss(
            *loss_inputs,
            mask_other_samples=config["loss_mask_other_samples"],
            pred2unit=config["pred2unit"],
            tau=config["tau"],
        )
    elif config["loss_type"] == "mse":
        loss = mse_loss(*loss_inputs)
    elif config["loss_type"] == "MAE":
        loss = mae_loss(*loss_inputs)
    else:
        raise f"loss_type must be patch_disc, MAE or mse, not {config['loss_type']}"

    return loss