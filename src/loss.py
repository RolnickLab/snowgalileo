import torch
import torch.nn.functional as F
from einops import repeat, rearrange


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


def debug_print(name, tensor):
    print(f"{name} - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}, contains NaN: {torch.isnan(tensor).any().item()}")


def seq_and_cat(s_t, sp, t, st):
    s_t = rearrange(s_t, "b h w t c_g d -> b (h w t c_g) d")
    sp = rearrange(sp, "b h w c_g d -> b (h w c_g) d")
    t = rearrange(t, "b t c_g d -> b (t c_g) d")
    # st is already a sequence
    return torch.cat([s_t, sp, t, st], dim=1)

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
    pred2unit=True,
    tau=0.2,
):
    # make targets 1, everything else 0
    s_t_m = (s_t_m == 2).int().unsqueeze(dim=-1)
    sp_m = (sp_m == 2).int().unsqueeze(dim=-1)
    t_m = (t_m == 2).int().unsqueeze(dim=-1)
    st_m = (st_m == 2).int().unsqueeze(dim=-1)

    # create tensors of shape (bsz, seq_len, dim)
    all_masks = seq_and_cat(s_t_m, sp_m, t_m, st_m)
    all_preds = seq_and_cat(p_s_t, p_sp, p_t, p_st)
    all_targets = seq_and_cat(t_s_t, t_sp, t_t, t_st)

    # create logit masks
    all_masks_squeezed = all_masks.squeeze(-1)  # (bsz, seq_len)
    row_mask = all_masks_squeezed.unsqueeze(2)  # (bsz, seq_len, 1)
    col_mask = all_masks_squeezed.unsqueeze(1)  #  (bsz, 1, seq_len)
    mask_intersection = row_mask & col_mask  #  (bsz, seq_len, seq_len)
    logit_mask = ((mask_intersection)*-1 + 1) * -1_000_000

    counts = torch.sqrt(mask_intersection.sum(dim=-1).sum(dim=-1))

    bs, nt, d = all_preds.shape

    all_preds = F.normalize(all_preds, p=2, dim=-1)
    all_targets = F.normalize(all_targets, p=2, dim=-1)

    scores = torch.einsum('npd,nqd->npq', all_preds, all_targets) / tau
    scores_sm = (scores + logit_mask).softmax(dim=-1)

    eyye = torch.eye(nt, dtype=torch.float, device=scores.device)[None].repeat(bs, 1, 1)

    neg_log = -torch.log(scores_sm + 0.000001) * mask_intersection
    custom_ce_loss = (tau * 2) * (neg_log * eyye).sum(dim=-1).sum(dim=-1) / counts
    custom_ce_loss = custom_ce_loss.mean()
    return custom_ce_loss



def remove_masks_and_cat(
    batch_idx,
    s_t,
    sp,
    t,
    st,
    s_t_m,
    sp_m,
    t_m,
    st_m,
):
    s_t = rearrange(s_t[batch_idx], 'h w t c_g d -> 1 (h w t c_g) d')
    sp = rearrange(sp[batch_idx], 'h w c_g d -> 1 (h w c_g) d')
    t = rearrange(t[batch_idx], 't c_g d -> 1 (t c_g) d')
    st = rearrange(st[batch_idx], 'c_g d -> 1 c_g d')

    s_t_m = rearrange(s_t_m[batch_idx], "h w t c_g -> (h w t c_g)")
    sp_m = rearrange(sp_m[batch_idx], "h w c_g -> (h w c_g)")
    t_m = rearrange(t_m[batch_idx], "t c_g -> (t c_g)")
    st_m = st_m[batch_idx]

    return torch.cat(
                [
                    s_t[:, s_t_m == 2, :],
                    sp[:, sp_m == 2, :],
                    t[:, t_m == 2, :],
                    st[:, st_m == 2, :],
                ],
            dim=1
            )
    
    


def patch_disc_loss_slow(
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
    pred2unit=True,
    tau=0.2,
):
    bsz = t_s_t.shape[0]

    loss = 0
    for batch_idx in range(bsz):
        pred = remove_masks_and_cat(
            batch_idx,
            p_s_t,
            p_sp,
            p_t,
            p_st,
            s_t_m,
            sp_m,
            t_m,
            st_m,
        )

        target = remove_masks_and_cat(
            batch_idx,
            t_s_t,
            t_sp,
            t_t,
            t_st,
            s_t_m,
            sp_m,
            t_m,
            st_m,
        )

        bs, nt, d = pred.shape
        
        # if pred2unit:
        #     pred_mu = pred.mean(1, keepdims=True)
        #     pred_std = pred.std(1, keepdims=True)
        #     pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        scores = torch.einsum('npd,nqd->npq', pred, target) / tau

        labels = torch.arange(nt, dtype=torch.long, device=pred.device)[None].repeat(bs, 1)
        loss += F.cross_entropy(scores.flatten(0, 1), labels.flatten(0, 1)) * (tau * 2)
    
    return loss / bsz