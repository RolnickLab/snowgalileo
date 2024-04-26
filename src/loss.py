import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def patchify_and_concat(space_time_array, space_only_array, time_only_array, patch_size):
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
    # repeat time only array for each spatial patch
    time_only_array = repeat(
        (rearrange(time_only_array, "b t c -> b (t c)")),
        "b n -> b t_h t_w n",
        t_h=space_time_array.shape[1],
        t_w=space_time_array.shape[2],
    )
    return torch.concat([space_time_array, space_only_array, time_only_array], dim=-1)


def mae_loss(
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
    norm_pix_loss=False,
):
    """
    If true, returns norm pix loss
    If false, returns MSE loss
    Inspired by: https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py#L198
    """
    if norm_pix_loss:
        target = patchify_and_concat(expanded_s_t_x, expanded_s_x, t_x, patch_size)
        pred = patchify_and_concat(p_s_t, p_s, p_t, patch_size)
        mask = patchify_and_concat(expanded_s_t, expanded_s, expanded_t, patch_size)

        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred[mask] - target[mask]) ** 2
        loss = loss.mean(dim=-1)  # mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    else:
        loss = F.mse_loss(
            torch.concat([p_s_t[expanded_s_t], p_s[expanded_s], p_t[expanded_t]]),
            torch.concat(
                [expanded_s_t_x[expanded_s_t], expanded_s_x[expanded_s], t_x[expanded_t]]
            ).float(),
        )
    return loss
