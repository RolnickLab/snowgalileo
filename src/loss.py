import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def mse_loss(
    t_s_t,
    t_sp,
    t_t,
    t_st,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    s_t_m,
    sp_m,
    t_m,
    st_m,
):
    encoder_size = t_s_t.shape[-1]
    expanded_s_t_m = repeat(s_t_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_sp_m = repeat(sp_m, "b h w c_g -> b h w c_g d", d=encoder_size)
    expanded_t_m = repeat(t_m, "b t c_g -> b t c_g d", d=encoder_size)
    expanded_st_m = repeat(st_m, "b c_g -> b c_g d", d=encoder_size)
    return F.mse_loss(
        torch.concat(
            [
                p_s_t[expanded_s_t_m == 2],
                p_sp[expanded_sp_m == 2],
                p_t[expanded_t_m == 2],
                p_st[expanded_st_m == 2],
            ]
        ),
        torch.concat(
            [
                t_s_t[expanded_s_t_m == 2],
                t_sp[expanded_sp_m == 2],
                t_t[expanded_t_m == 2],
                t_st[expanded_st_m == 2],
            ]
        ).float(),
    )


def seq_and_cat(s_t, sp, t, st):
    s_t = rearrange(s_t, "b h w t c_g d -> b (h w t c_g) d")
    sp = rearrange(sp, "b h w c_g d -> b (h w c_g) d")
    t = rearrange(t, "b t c_g d -> b (t c_g) d")
    # st is already a sequence
    return torch.cat([s_t, sp, t, st], dim=1)


def expand_and_reciprocate(t):
    reciprocals = torch.reciprocal(t.float())
    return torch.repeat_interleave(reciprocals, t)


def patch_disc_loss(
    t_s_t,
    t_sp,
    t_t,
    t_st,
    p_s_t,
    p_sp,
    p_t,
    p_st,
    s_t_m,
    sp_m,
    t_m,
    st_m,
    mask_other_samples: bool,
    pred2unit: bool = True,
    tau: float = 0.2,
):
    # create tensors of shape (bsz, seq_len, dim)
    all_masks = seq_and_cat(
        s_t_m.unsqueeze(dim=-1),
        sp_m.unsqueeze(dim=-1),
        t_m.unsqueeze(dim=-1),
        st_m.unsqueeze(dim=-1),
    ).squeeze(-1)
    all_preds = seq_and_cat(p_s_t, p_sp, p_t, p_st)
    all_targets = seq_and_cat(t_s_t, t_sp, t_t, t_st)

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
    loss = (loss * loss_multiplier).sum() / t_s_t.shape[0]
    return loss


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
    else:
        raise f"loss_type must be patch_disc or mse, not {config["loss_type"]}"

    return loss
