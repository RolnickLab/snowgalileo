import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

from .data import DYNAMIC_BANDS_GROUPS_IDX, STATIC_BAND_GROUPS_IDX


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                # a value of True indicates that the element should take part in attention
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p,
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, attn_mask):
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        temporal_depth=2,
        spatial_depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.dynamic_groups = DYNAMIC_BANDS_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size

        self.dynamic_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.dynamic_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.static_groups.items()
            }
        )

        self.temporal_blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(temporal_depth)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(spatial_depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = embedding_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_sinusoid_encoding_table(self.pos_embed.shape[1], self.pos_embed.shape[-1])
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_linear_projection(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
    ):
        """
        Given a [H, W, (T), B] inputs, returns a [H, W, (T), B_G, D] output.
        """
        d_i, d_m, s_i, s_m = [], [], [], []
        for channel_group, channel_idxs in self.dynamic_groups.items():
            d_i.append(self.dynamic_embed[channel_group](dynamic_x[:, :, :, :, channel_idxs]))
            d_m.append(dynamic_mask[:, :, :, channel_idxs[0]])
        for channel_group, channel_idxs in self.static_groups.items():
            s_i.append(self.static_embed[channel_group](static_x[:, :, :, channel_idxs]))
            s_m.append(static_mask[:, :, :, channel_idxs[0]])
        return (
            torch.stack(d_i, dim=3),
            torch.stack(s_i, dim=3),
            torch.stack(d_m, dim=3),
            torch.stack(s_m, dim=3),
        )

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
    ):
        b, h, w = dynamic_x.shape[0], dynamic_x.shape[1], dynamic_x.shape[2]
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_linear_projection(
            dynamic_x, static_x, dynamic_mask, static_mask
        )
        d_t, d_c, s_c = dynamic_x.shape[3], dynamic_x.shape[4], static_x.shape[3]

        # apply temporal Transformer blocks
        dynamic_x = rearrange(dynamic_x, "b h w t c d -> (b h w) (t c) d")
        dynamic_mask = rearrange(dynamic_mask, "b h w t c -> (b h w) (t c)")
        static_x = rearrange(static_x, "b h w c d -> (b h w) c d")
        static_mask = rearrange(static_mask, "b h w c -> (b h w) c")
        num_dynamic_tokens = dynamic_x.shape[1]
        joined_x = torch.cat([dynamic_x, static_x], dim=1)
        joined_mask = torch.cat([dynamic_mask, static_mask], dim=1)
        for blk in self.temporal_blocks:
            joined_x = blk(joined_x, attn_mask=~joined_mask.bool())

        # apply spatial Transformer blocks
        dynamic_x = rearrange(
            joined_x[:, :num_dynamic_tokens, :],
            "(b h w) (t c) d -> b (t c) (h w) d",
            h=h,
            w=w,
            b=b,
            t=d_t,
            c=d_c,
        )
        static_x = rearrange(
            joined_x[:, num_dynamic_tokens:, :], "(b h w) c d -> b c (h w) d", b=b, h=h, w=w, c=s_c
        )
        dynamic_mask = rearrange(
            joined_mask[:, :num_dynamic_tokens],
            "(b h w) (t c) -> b (t c) (h w)",
            h=h,
            w=w,
            b=b,
            t=d_t,
            c=d_c,
        )
        static_mask = rearrange(
            joined_mask[:, num_dynamic_tokens:], "(b h w) c -> b c (h w)", b=b, h=h, w=w, c=s_c
        )
        joined_x = torch.cat([dynamic_x, static_x], dim=1)
        joined_mask = torch.cat([dynamic_mask, static_mask], dim=1)
        num_dynamic_tokens, j_t = dynamic_x.shape[1], joined_x.shape[1]
        joined_x = rearrange(joined_x, "b t hw d -> (b t) hw d")
        joined_mask = rearrange(joined_mask, "b t hw -> (b t) hw")
        for blk in self.temporal_blocks:
            joined_x = blk(joined_x, attn_mask=~joined_mask.bool())

        joined_x = rearrange(joined_x, "(b t) hw d -> b t hw d", b=b, t=j_t)
        joined_mask = rearrange(joined_mask, "(b t) hw -> b t hw", b=b, t=j_t)
        dynamic_x = rearrange(
            joined_x[:, :num_dynamic_tokens, :],
            "b (t c) (h w) d -> b h w t c d",
            h=h,
            w=w,
            b=b,
            t=d_t,
            c=d_c,
        )
        static_x = rearrange(
            joined_x[:, num_dynamic_tokens:, :], "b c (h w) d -> b h w c d", h=h, w=w
        )
        dynamic_mask = rearrange(
            joined_mask[:, :num_dynamic_tokens],
            "b (t c) (h w) -> b h w t c",
            h=h,
            w=w,
            b=b,
            t=d_t,
            c=d_c,
        )
        static_mask = rearrange(
            joined_mask[:, num_dynamic_tokens:], "b c (h w) -> b h w c", h=h, w=w
        )

        return dynamic_x, static_x, dynamic_mask, static_mask
