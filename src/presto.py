import numpy as np
import torch
import logging
from einops import rearrange, repeat
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

from .config import PRESTO_INPUT_SIZE
from .data import DYNAMIC_BANDS_GROUPS_IDX, STATIC_BAND_GROUPS_IDX
from .embeddings import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_month_encoding_table,
)

logger = logging.getLogger("__main__")

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
                # attn_mask will have shape [batch, num_heads, N, N]
                # we want to make sure the trace is unmasked; otherwise
                # we get NaNs
                diagonal = repeat(
                    torch.eye(N, device=attn_mask.device).bool(),
                    "s1 s2 -> b h s1 s2",
                    b=B,
                    h=self.num_heads,
                )
                attn_mask = attn_mask + diagonal
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


class PrestoAttn(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        temporal_depth=2,
        spatial_depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
    ):
        super().__init__()
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
        # embeddings for the (T, C) attention
        pos_embed_dim = int(embedding_size * (1 - (channel_embed_ratio + month_embed_ratio)))
        channel_embed_dim = int(embedding_size * channel_embed_ratio)
        month_embed_dim = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(pos_embed_dim, np.arange(max_sequence_length))
            ).float(),
            requires_grad=False,
        )
        month_tab = torch.from_numpy(get_month_encoding_table(month_embed_dim)).float()
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.d_channel_embed = nn.Parameter(
            torch.zeros(len(DYNAMIC_BANDS_GROUPS_IDX), channel_embed_dim)
        )
        self.s_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), channel_embed_dim)
        )
        self.pos_embed_2d = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embedding_size, PRESTO_INPUT_SIZE)).float(),
            requires_grad=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
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

    def construct_temporal_channel_embeddings(self, b: int, h: int, w: int, months: torch.Tensor):
        t = months.shape[1]
        d_channel = repeat(self.d_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        c_g = d_channel.shape[-2]
        d_pos = repeat(self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=c_g)
        m_embed = repeat(self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=c_g)

        d_embed = torch.cat([d_channel, d_pos, m_embed], dim=-1)
        s_channel = repeat(self.s_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)
        s_zeros = torch.zeros(
            b,
            h,
            w,
            s_channel.shape[-2],
            d_embed.shape[-1] - s_channel.shape[-1],
            device=s_channel.device,
        ).float()
        s_embed = torch.cat([s_channel, s_zeros], dim=-1)
        return d_embed, s_embed

    def construct_spatial_channel_embeddings(self, b: int, t: int, d_c_g: int, s_c_g: int):
        d_pos_embed_2d = repeat(self.pos_embed_2d, "hw d -> b tc hw d", b=b, tc=t * d_c_g)
        s_pos_embed_2d = repeat(self.pos_embed_2d, "hw d -> b s_c hw d", b=b, s_c=s_c_g)
        return d_pos_embed_2d, s_pos_embed_2d

    def apply_temporal_channel_attention(self, d_x, s_x, d_m, s_m, months):
        """
        Attn requires a (B, N, D) input.
        Move H, W into the B dimension, and T, C into the N dimension so
        that the model attends over the time and channel dimensions
        """
        b, h, w = d_x.shape[0], d_x.shape[1], d_x.shape[2]
        d_t, d_c, s_c = d_x.shape[3], d_x.shape[4], s_x.shape[3]

        d_embed, s_embed = self.construct_temporal_channel_embeddings(b, h, w, months)

        # apply temporal Transformer blocks
        d_x = d_x + d_embed
        s_x = s_x + s_embed
        d_x = rearrange(d_x, "b h w t c d -> (b h w) (t c) d")
        d_m = rearrange(d_m, "b h w t c -> (b h w) (t c)")
        s_x = rearrange(s_x, "b h w c d -> (b h w) c d")
        s_m = rearrange(s_m, "b h w c -> (b h w) c")
        num_d_t = d_x.shape[1]
        x = torch.cat([d_x, s_x], dim=1)
        m = torch.cat([d_m, s_m], dim=1)
        for blk in self.temporal_blocks:
            x = blk(x, attn_mask=~m.bool())

        d_x = rearrange(
            x[:, :num_d_t, :], "(b h w) (t c) d -> b h w t c d", h=h, w=w, b=b, t=d_t, c=d_c
        )
        s_x = rearrange(x[:, num_d_t:, :], "(b h w) c d -> b h w c d", b=b, h=h, w=w, c=s_c)
        d_m = rearrange(m[:, :num_d_t], "(b h w) (t c) -> b h w t c", h=h, w=w, b=b, t=d_t, c=d_c)
        s_m = rearrange(m[:, num_d_t:], "(b h w) c -> b h w c", b=b, h=h, w=w, c=s_c)
        return d_x, s_x, d_m, s_m

    def apply_spatial_attention(self, d_x, s_x, d_m, s_m):
        """
        Attn requires a (B, N, D) input.
        Move T, C into the B dimension, and H, W into the N dimension so
        that the model attends over the height and width dimensions
        """
        b, h, w, d_t, d_c = d_x.shape[0], d_x.shape[1], d_x.shape[2], d_x.shape[3], d_x.shape[4]
        s_c = s_x.shape[3]
        d_x = rearrange(d_x, "b h w t c d -> b (t c) (h w) d")
        d_m = rearrange(d_m, "b h w t c -> b (t c) (h w)")
        s_x = rearrange(s_x, "b h w c d -> b c (h w) d")
        s_m = rearrange(s_m, "b h w c -> b c (h w)")

        d_embed, s_embed = self.construct_spatial_channel_embeddings(b, d_t, d_c, s_c)
        d_x = d_x + d_embed
        s_x = s_x + s_embed
        num_d_t = d_t * d_c
        x = torch.cat([d_x, s_x], dim=1)
        m = torch.cat([d_m, s_m], dim=1)
        j_t = x.shape[1]
        x = rearrange(x, "b t hw d -> (b t) hw d")
        m = rearrange(m, "b t hw -> (b t) hw")
        for blk in self.temporal_blocks:
            x = blk(x, attn_mask=~m.bool())
        x = rearrange(x, "(b t) hw d -> b t hw d", b=b, t=j_t)
        m = rearrange(m, "(b t) hw -> b t hw", b=b, t=j_t)
        d_x = rearrange(
            x[:, :num_d_t, :], "b (t c) (h w) d -> b h w t c d", h=h, w=w, b=b, t=d_t, c=d_c
        )
        s_x = rearrange(x[:, num_d_t:, :], "b c (h w) d -> b h w c d", h=h, w=w)
        d_m = rearrange(m[:, :num_d_t], "b (t c) (h w) -> b h w t c", h=h, w=w, b=b, t=d_t, c=d_c)
        s_m = rearrange(m[:, num_d_t:], "b c (h w) -> b h w c", h=h, w=w)
        return d_x, s_x, d_m, s_m

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
        months: torch.Tensor,
    ):
        # apply temporal Transformer blocks
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_temporal_channel_attention(
            dynamic_x, static_x, dynamic_mask, static_mask, months
        )
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_spatial_attention(
            dynamic_x, static_x, dynamic_mask, static_mask
        )
        return dynamic_x, static_x, dynamic_mask, static_mask


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
        self.presto_attn = PrestoAttn(
            embedding_size=embedding_size,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def apply_linear_projection(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
    ):
        """
        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), B_G, D] output.
        B_G = number of band groups, D = embedding dimension.
        Masks keep their shape.
        """
        d_i, d_m, s_i, s_m = [], [], [], []
        for idx, (channel_group, channel_idxs) in enumerate(self.dynamic_groups.items()):
            d_i.append(self.dynamic_embed[channel_group](dynamic_x[:, :, :, :, channel_idxs]))
            d_m.append(dynamic_mask[:, :, :, :, idx])
        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            s_i.append(self.static_embed[channel_group](static_x[:, :, :, channel_idxs]))
            s_m.append(static_mask[:, :, :, idx])
        return (
            torch.stack(d_i, dim=-2),
            torch.stack(s_i, dim=-2),
            torch.stack(d_m, dim=-1),
            torch.stack(s_m, dim=-1),
        )

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
        months: torch.Tensor,
    ):
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_linear_projection(
            dynamic_x, static_x, dynamic_mask, static_mask
        )
        dynamic_x, static_x, dynamic_mask, static_mask = self.presto_attn(
            dynamic_x, static_x, dynamic_mask, static_mask, months
        )
        return dynamic_x, static_x, dynamic_mask, static_mask, months


class PrestoDecoder(nn.Module):
    def __init__(
        self,
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        temporal_depth=2,
        spatial_depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))
        self.decoder_embed = nn.Linear(encoder_embedding_size, decoder_embedding_size, bias=True)

        self.presto_attn = PrestoAttn(
            embedding_size=decoder_embedding_size,
            temporal_depth=temporal_depth,
            spatial_depth=spatial_depth,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            max_sequence_length=max_sequence_length,
        )

    def add_masks(self, d_x: torch.Tensor, d_m: torch.Tensor):
        # we make an assumption here that mask_by_presto_pixels_time
        # was used to make the masks. This means we only have masked
        # timesteps, which simplifies the mask addition
        d_x = d_x * (1 - d_m).unsqueeze(-1)
        B, H, W, T, C = d_x.shape[0], d_x.shape[1], d_x.shape[2], d_x.shape[3], d_x.shape[4]
        mask_reshaped = repeat(self.mask_token, "d -> b h w t c d", b=B, h=H, w=W, t=T, c=C)
        masks_to_add = mask_reshaped * d_m.unsqueeze(-1)
        d_m = d_m * 0  # all values are unmasked now
        return d_x + masks_to_add, d_m

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
        months: torch.Tensor,
    ):
        dynamic_x = self.decoder_embed(dynamic_x)
        static_x = self.decoder_embed(static_x)
        dynamic_x, dynamic_mask = self.add_masks(dynamic_x, dynamic_mask)
        dynamic_x, static_x, dynamic_mask, static_mask = self.presto_attn(
            dynamic_x, static_x, dynamic_mask, static_mask, months
        )
        return dynamic_x, static_x, dynamic_mask, static_mask
