import collections.abc
import itertools
import math
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, vmap
from torch.jit import Final

from .config import BASE_GSD
from .data import DYNAMIC_BANDS_GROUPS_IDX, STATIC_BAND_GROUPS_IDX
from .embeddings import (
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed_with_resolution,
    get_month_encoding_table,
)


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, start_lr, max_lr, min_lr):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = start_lr + (max_lr * epoch / warmup_epochs)
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# thanks to https://github.com/bwconrad/flexivit/ for this nice implementation
# of the FlexiPatchEmbed module
def to_2tuple(x: Any) -> Tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 4,
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24
        by https://github.com/bwconrad/flexivit/

        Args:
            patch_size: Base patch size. i.e the size of the parameter buffer
            in_chans: Number of input image channels
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            bias: Whether to use bias in convolution
            patch_size_seq: List of patch sizes to randomly sample from
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Flexi specific attributes
        self.interpolation = interpolation
        self.antialias = antialias

        self.patch_size_seq = patch_size_seq

        # Pre-calculate pinvs
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        """Pre-calculate all pinv matrices"""
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: Tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: Tuple[int, int], new_shape: Tuple[int, int]) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: Tuple[int, int]):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if self.patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        # x has input shape [b, h, w, (t), c]
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0  # ignored if has_time_dimension is False
        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) h w c")
        x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            # During evaluation use base patch size if not specified
            patch_size = self.patch_size

        patch_size = to_2tuple(patch_size)

        # Resize conv weights
        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)

        # Apply conv with resized weights
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)

        return x


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


class FlexiPrestoBase(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        base_patch_size: int = 4,
    ):
        super().__init__()

        self.dynamic_groups = DYNAMIC_BANDS_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.dynamic_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group), embed_dim=embedding_size, patch_size=base_patch_size
                )
                for group_name, group in self.dynamic_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group), embed_dim=embedding_size, patch_size=base_patch_size
                )
                for group_name, group in self.static_groups.items()
            }
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.max_sequence_length = max_sequence_length
        # we have 4 embeddings (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension. This will change soon anyway
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = torch.from_numpy(get_month_encoding_table(int(embedding_size * 0.25))).float()
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.d_channel_embed = nn.Parameter(
            torch.zeros(len(DYNAMIC_BANDS_GROUPS_IDX), int(embedding_size * 0.25))
        )
        self.s_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25))
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def collapse_hwtc(d_x: torch.Tensor, s_x: torch.Tensor, d_m: torch.Tensor, s_m: torch.Tensor):
        d_x = rearrange(d_x, "b h w t c_g d -> b (h w t c_g) d")
        s_x = rearrange(s_x, "b h w c_g d -> b (h w c_g) d")
        d_m = rearrange(d_m, "b h w t c_g-> b (h w t c_g)")
        s_m = rearrange(s_m, "b h w c_g-> b (h w c_g)")
        return d_x, s_x, d_m, s_m

    @staticmethod
    def split_and_expand_hwtc(
        x: torch.Tensor, m: torch.Tensor, h: int, w: int, t: int, d_c_g: int, s_c_g: int
    ):
        n_d_t = h * w * t * d_c_g
        d_x = rearrange(x[:, :n_d_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=d_c_g)
        s_x = rearrange(x[:, n_d_t:], "b (h w c) d -> b h w c d", h=h, w=w, c=s_c_g)
        d_m = rearrange(m[:, :n_d_t], "b (h w t c) -> b h w t c", h=h, w=w, t=t, c=d_c_g)
        s_m = rearrange(m[:, n_d_t:], "b (h w c) -> b h w c", h=h, w=w, c=s_c_g)
        return d_x, s_x, d_m, s_m

    def apply_encodings(self, d_x, s_x, months, patch_size, input_res):
        b, h, w, t, c_g_d, d = d_x.shape
        c_g_s = s_x.shape[-2]
        d_channel = repeat(self.d_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        d_pos = repeat(self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=c_g_d)
        m_embed = repeat(self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=c_g_d)

        s_channel = repeat(self.s_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)
        s_zeros = torch.zeros(
            b,
            h,
            w,
            c_g_s,
            s_channel.shape[-1] * 2,
            device=s_channel.device,
        ).float()

        # find the resolution that each token represents, which will be
        # the number of pixels in a patch * the resolution of each pixel
        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w, "get_2d_sincos_pos_embed_with_resolution currently requires that h==w"
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25),
            h,
            torch.ones(b).to(d_x.device) * gsd_ratio,
            device=d_x.device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_d = repeat(
            spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=c_g_d
        )
        spatial_embed_s = repeat(spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=c_g_s)

        d_embed = torch.cat([d_channel, d_pos, m_embed, spatial_embed_d], dim=-1)
        s_embed = torch.cat([s_channel, s_zeros, spatial_embed_s], dim=-1)
        return d_x + d_embed, s_x + s_embed

    def apply_attn(self, d_x, s_x, d_m, s_m, m, patch_size, input_res):
        # todo - add encodings
        _, h, w, t, d_c_g, _ = d_x.shape
        s_c_g = s_x.shape[3]
        d_x, s_x = self.apply_encodings(d_x, s_x, m, patch_size, input_res)
        d_x, s_x, d_m, s_m = self.collapse_hwtc(d_x, s_x, d_m, s_m)
        x = torch.cat([d_x, s_x], dim=1)
        m = torch.cat([d_m, s_m], dim=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=~m.bool())
        return self.split_and_expand_hwtc(x, m, h, w, t, d_c_g, s_c_g)


class Encoder(FlexiPrestoBase):
    def __init__(
        self,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        num_inputs_per_spatial_dim=4,
    ):
        super().__init__(
            embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            num_inputs_per_spatial_dim,
        )

        self.dynamic_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embedding_size)
                for group_name, group in self.dynamic_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embedding_size)
                for group_name, group in self.static_groups.items()
            }
        )

        self.norm = nn.LayerNorm(embedding_size)

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
        patch_size: Optional[int],
    ):
        """
        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), C_G, D] output.
        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        d_i, s_i, d_m, s_m = [], [], [], []
        for idx, (channel_group, channel_idxs) in enumerate(self.dynamic_groups.items()):
            d_i.append(
                self.dynamic_embed[channel_group](
                    dynamic_x[:, :, :, :, channel_idxs], patch_size=patch_size
                )
            )
            d_m.append(dynamic_mask[:, 0::patch_size, 0::patch_size, :, idx])
        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            s_i.append(
                self.static_embed[channel_group](
                    static_x[:, :, :, channel_idxs], patch_size=patch_size
                )
            )
            s_m.append(static_mask[:, 0::patch_size, 0::patch_size, idx])
        return (
            torch.stack(d_i, dim=-2),
            torch.stack(s_i, dim=-2),
            torch.stack(d_m, dim=-1),
            torch.stack(s_m, dim=-1),
        )

    @classmethod
    def average_tokens(cls, d_x, s_x, d_m, s_m):
        d_x, s_x, d_m, s_m = cls.collapse_hwtc(d_x, s_x, d_m, s_m)
        x = torch.cat([d_x, s_x], dim=1)  # B, N, D
        m = torch.cat([d_m, s_m], dim=1)  # B, N
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    def forward(
        self,
        dynamic_x: torch.Tensor,
        static_x: torch.Tensor,
        dynamic_mask: torch.Tensor,
        static_mask: torch.Tensor,
        months: torch.Tensor,
        patch_size: Optional[int] = None,
        input_resolution_m: Optional[int] = BASE_GSD,
    ):
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_linear_projection(
            dynamic_x, static_x, dynamic_mask, static_mask, patch_size
        )
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_attn(
            dynamic_x, static_x, dynamic_mask, static_mask, months, patch_size, input_resolution_m
        )

        return self.norm(dynamic_x), self.norm(static_x), dynamic_mask, static_mask, months


class PrestoPixelDecoder(FlexiPrestoBase):
    def __init__(
        self,
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        num_inputs_per_spatial_dim=4,
        max_patch_size: int = 8,
    ):
        super().__init__(
            decoder_embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            num_inputs_per_spatial_dim,
        )
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))

        self.max_patch_size = max_patch_size
        self.dynamic_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embedding_size, len(group) * max_patch_size**2)
                for group_name, group in self.dynamic_groups.items()
            }
        )
        self.static_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embedding_size, len(group) * max_patch_size**2)
                for group_name, group in self.static_groups.items()
            }
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
        patch_size: Optional[int] = None,
        input_resolution_m: Optional[int] = BASE_GSD,
    ):
        dynamic_x = self.encoder_to_decoder_embed(dynamic_x)
        static_x = self.encoder_to_decoder_embed(static_x)
        dynamic_x, dynamic_mask = self.add_masks(dynamic_x, dynamic_mask)
        dynamic_x, static_x, _, _ = self.apply_attn(
            dynamic_x, static_x, dynamic_mask, static_mask, months, patch_size, input_resolution_m
        )
        output_d, output_s = [], []
        for idx, (group_name, c_g) in enumerate(self.dynamic_groups.items()):
            # decoded has shape [b, h, w, t, len(c_g) * patch_size ** 2]
            decoded = self.dynamic_embed[group_name](dynamic_x[:, :, :, :, idx])
            output_d.append(
                rearrange(
                    decoded,
                    "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
                    c_g=len(c_g),
                    p_w=self.max_patch_size,
                    p_h=self.max_patch_size,
                )
            )

        for idx, (group_name, c_g) in enumerate(self.static_groups.items()):
            # decoded has shape [b, h, w, len(c_g) * patch_size ** 2]
            decoded = self.static_embed[group_name](static_x[:, :, :, idx])
            output_s.append(
                rearrange(
                    decoded,
                    "b t_h t_w (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) c_g",
                    c_g=len(c_g),
                    p_w=self.max_patch_size,
                    p_h=self.max_patch_size,
                )
            )

        return torch.cat(output_d, dim=-1), torch.cat(output_s, dim=-1)


class PrestoRepresentationDecoder(FlexiPrestoBase):
    def __init__(
        self,
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        num_inputs_per_spatial_dim=4,
    ):
        super().__init__(
            decoder_embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            num_inputs_per_spatial_dim,
        )

        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )
        self.decoder_to_encoder_embed = nn.Linear(
            decoder_embedding_size, encoder_embedding_size, bias=True
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
        patch_size: Optional[int] = None,
        input_resolution_m: Optional[int] = BASE_GSD,
    ):
        dynamic_x = self.encoder_to_decoder_embed(dynamic_x)
        static_x = self.encoder_to_decoder_embed(static_x)
        dynamic_x, dynamic_mask = self.add_masks(dynamic_x, dynamic_mask)
        dynamic_x, static_x, dynamic_mask, static_mask = self.apply_attn(
            dynamic_x,
            static_x,
            torch.zeros_like(dynamic_mask, device=dynamic_mask.device),
            torch.zeros_like(static_mask, device=static_mask.device),
            months,
            patch_size,
            input_resolution_m,
        )
        return (
            self.decoder_to_encoder_embed(dynamic_x),
            self.decoder_to_encoder_embed(static_x),
            dynamic_mask,
            static_mask,
        )
