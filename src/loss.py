import torch
import torch.nn.functional as F
from einops import repeat


def mse_loss(
    t_s_x,
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
    encoder_size = t_s_x.shape[-1]
    expanded_s_t_m = repeat(s_t_m, "b h w t c_g -> b h w t c_g d", d=encoder_size)
    expanded_sp_m = repeat(sp_m, "b h w c_g -> b h w c_g d", d=encoder_size)
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
                expanded_s_t_x[expanded_s_t_m == 2],
                expanded_sp_x[expanded_sp_m == 2],
                t_x[expanded_t_m == 2],
                st_x[expanded_st_m == 2],
            ]
        ).float(),
    )
