import collections.abc
import itertools
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, vmap
from torch.jit import Final

from .config import BASE_GSD
from .data import SPACE_BAND_GROUPS_IDX, SPACE_TIME_BANDS_GROUPS_IDX, TIME_BAND_GROUPS_IDX
from .data.config import CONFIG_FILENAME, ENCODER_FILENAME
from .embeddings import (
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed_with_resolution,
    get_month_encoding_table,
)
from .utils import device


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
        patch_size: Union[int, Tuple[int, int]],
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
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
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
        cross_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  # FIXME

        self.cross_attn = cross_attn
        if not cross_attn:
            self.qkv: Optional[nn.Linear] = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.q: Optional[nn.Linear] = None
            self.kv: Optional[nn.Linear] = None
        else:
            self.qkv = None
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None, attn_mask=None):
        if y is None:
            assert not self.cross_attn
            B, N, C = x.shape
            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
        else:
            assert self.cross_attn
            B, N, C = x.shape
            Ny = y.shape[1]
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = (
                self.kv(y)
                .reshape(B, Ny, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            k, v = kv[0], kv[1]

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
        cross_attn: bool = False,
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
            cross_attn=cross_attn,
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

    def forward(self, x, y, attn_mask):
        x = x + self.ls1(self.attn(self.norm1(x), y, attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class FlexiPrestoBase(nn.Module):
    cross_attn: bool

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

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    cross_attn=self.cross_attn,
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
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25))
        )
        self.s_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25))
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25))
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls,
        s_t_x: torch.Tensor,
        s_x: torch.Tensor,
        t_x: torch.Tensor,
        s_t_m: torch.Tensor,
        s_m: torch.Tensor,
        t_m: torch.Tensor,
    ):
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        s_x = rearrange(s_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

        s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
        s_m = rearrange(s_m, "b h w c_g-> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

        x = torch.cat(
            [
                s_t_x,
                s_x,
                t_x,
            ],
            dim=1,
        )
        m = torch.cat([s_t_m, s_m, t_m], dim=1)
        return x, m

    @staticmethod
    def split_x_y(tokens, mask):
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        tokens = tokens.gather(1, indices[:, :, None].expand_as(tokens))

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        min_length = sorted_mask.sum(-1).min()
        y = tokens[:, :max_length]
        x = tokens[:, min_length:]

        x_mask = 1 - sorted_mask[:, min_length:].to(dtype=org_mask_dtype)
        y_mask = 1 - sorted_mask[:, :max_length].to(dtype=org_mask_dtype)
        return x, y, x_mask, y_mask, indices

    @staticmethod
    def combine_x_y(x, y, x_mask, y_mask, indices):
        # multiply by mask to zero out, then add
        B, T = indices.shape[0], indices.shape[1]
        D = x.shape[-1]

        tokens = torch.zeros((B, T, D), dtype=x.dtype, device=x.device)
        full_mask = torch.zeros((B, T), dtype=x_mask.dtype, device=x_mask.device)
        tokens[:, : y.shape[1]] = y * (1 - y_mask).unsqueeze(-1)
        tokens[:, -x.shape[1] :] += x * x_mask.unsqueeze(-1)
        full_mask[:, : y_mask.shape[1]] = y_mask
        full_mask[:, -x_mask.shape[1] :] += x_mask

        tokens = tokens.scatter(1, indices[:, :, None].expand_as(tokens), tokens)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)

        return tokens, torch.clamp(full_mask, max=1)

    @staticmethod
    def remove_masked_tokens(x, mask):
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        # https://stackoverflow.com/a/68621610/2332296
        # move all non-masked values to the front of their rows
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        # set masked values to 0 (not really necessary since we'll ignore them anyway)
        x = x * sorted_mask.unsqueeze(-1)

        # cut off to the length of the longest sequence
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]

        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(x, indices, mask):
        masked_tokens = repeat(
            torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1]
        )
        full_mask = torch.cat(
            (
                mask,
                torch.ones(
                    (x.shape[0], indices.shape[1] - x.shape[1]), device=x.device, dtype=mask.dtype
                ),
            ),
            dim=-1,
        )
        # can't set value on leaf variable
        out = masked_tokens.clone()
        # put tokens in full masked tensor (at the first N positions in every row)
        out[~full_mask.bool()] = x[~mask.bool()]
        # then move them to their original positions
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    @classmethod
    def split_and_expand_hwtc(
        cls,
        x: torch.Tensor,
        m: torch.Tensor,
        h: int,
        w: int,
        t: int,
        s_t_c_g: int,
        s_c_g: int,
        t_c_g: int,
    ):
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g)
        s_x = rearrange(x[:, n_s_t_t:-n_t_t], "b (h w c) d -> b h w c d", h=h, w=w, c=s_c_g)
        t_x = rearrange(x[:, -n_t_t:], "b (t c) d -> b t c d", t=t, c=t_c_g)

        s_t_m = rearrange(m[:, :n_s_t_t], "b (h w t c) -> b h w t c", h=h, w=w, t=t, c=s_t_c_g)
        s_m = rearrange(m[:, n_s_t_t:-n_t_t], "b (h w c) -> b h w c", h=h, w=w, c=s_c_g)
        t_m = rearrange(m[:, -n_t_t:], "b (t c) -> b t c", t=t, c=t_c_g)
        return s_t_x, s_x, t_x, s_t_m, s_m, t_m

    def apply_encodings(self, s_t_x, s_x, t_x, months, patch_size, input_res):
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        s_c_g, t_c_g = s_x.shape[-2], t_x.shape[-2]

        s_t_channel = repeat(self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        pos_embed_s_t = repeat(
            self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g
        )
        m_embed_s_t = repeat(
            self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g
        )

        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)
        t_zeros = torch.zeros(b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device)

        s_channel = repeat(self.s_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)
        s_zeros = torch.zeros(
            b,
            h,
            w,
            s_c_g,
            s_channel.shape[-1] * 2,
            device=s_channel.device,
        )

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
            torch.ones(b).to(s_t_x.device) * gsd_ratio,
            device=s_t_x.device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_s_t = repeat(
            spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g
        )
        spatial_embed_s = repeat(spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=s_c_g)

        s_t_embed = torch.cat([s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1)
        s_embed = torch.cat([s_channel, s_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        return s_t_x + s_t_embed, s_x + s_embed, t_x + t_embed

    def apply_attn(self, s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size, input_res):
        # todo - add encodings
        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        s_c_g, t_c_g = s_x.shape[3], t_x.shape[-2]
        s_t_x, s_x, t_x = self.apply_encodings(s_t_x, s_x, t_x, months, patch_size, input_res)
        x, m = self.collapse_and_combine_hwtc(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        if not self.cross_attn:
            x, indices, m = self.remove_masked_tokens(x, m)
            for blk in self.blocks:
                # we take the inverse of the mask because a value
                # of True indicates the value *should* take part in
                # attention
                x = blk(x=x, y=None, attn_mask=~m.bool())
            x, m = self.add_removed_tokens(x, indices, m)
            return self.split_and_expand_hwtc(x, m, h, w, t, s_t_c_g, s_c_g, t_c_g)
        else:
            x, y, x_mask, y_mask, indices = self.split_x_y(x, m)
            for blk in self.blocks:
                x = blk(x=x, y=y, attn_mask=~y_mask.bool())
            x, m = self.combine_x_y(x, y, x_mask, y_mask, indices)
            return self.split_and_expand_hwtc(x, m, h, w, t, s_t_c_g, s_c_g, t_c_g)


class Encoder(FlexiPrestoBase):
    cross_attn = False

    def __init__(
        self,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__(
            embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            max_patch_size,
        )

        self.space_time_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size
                )
                for group_name, group in self.space_time_groups.items()
            }
        )
        self.space_embed = nn.ModuleDict(
            {
                group_name: FlexiPatchEmbed(
                    in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size
                )
                for group_name, group in self.space_groups.items()
            }
        )
        self.time_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(in_features=len(group), out_features=embedding_size)
                for group_name, group in self.time_groups.items()
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
        s_t_x: torch.Tensor,
        s_x: torch.Tensor,
        t_x: torch.Tensor,
        s_t_m: torch.Tensor,
        s_m: torch.Tensor,
        t_m: torch.Tensor,
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
        s_t_l, s_l, t_l, s_t_m_l, s_m_l, t_m_l = [], [], [], [], [], []
        for idx, (channel_group, channel_idxs) in enumerate(self.space_time_groups.items()):
            s_t_l.append(
                self.space_time_embed[channel_group](
                    s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                )
            )
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            s_l.append(
                self.space_embed[channel_group](s_x[:, :, :, channel_idxs], patch_size=patch_size)
            )
            s_m_l.append(s_m[:, 0::patch_size, 0::patch_size, idx])

        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            t_m_l.append(t_m[:, :, idx])

        return (
            torch.stack(s_t_l, dim=-2),
            torch.stack(s_l, dim=-2),
            torch.stack(t_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1),
            torch.stack(s_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1),
        )

    @classmethod
    def average_tokens(cls, s_t_x, s_x, t_x, s_t_m, s_m, t_m):
        x, m = cls.collapse_and_combine_hwtc(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    def forward(
        self,
        s_t_x: torch.Tensor,
        s_x: torch.Tensor,
        t_x: torch.Tensor,
        s_t_m: torch.Tensor,
        s_m: torch.Tensor,
        t_m: torch.Tensor,
        months: torch.Tensor,
        patch_size: Optional[int] = None,
        input_resolution_m: Optional[int] = BASE_GSD,
    ):
        s_t_x, s_x, t_x, s_t_m, s_m, t_m = self.apply_linear_projection(
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, patch_size
        )
        s_t_x, s_x, t_x, s_t_m, s_m, t_m = self.apply_attn(
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size, input_resolution_m
        )

        return self.norm(s_t_x), self.norm(s_x), self.norm(t_x), s_t_m, s_m, t_m, months

    @classmethod
    def load_from_folder(cls, folder: Path):
        assert (folder / CONFIG_FILENAME).exists(), f"Missing {CONFIG_FILENAME}"
        assert (folder / ENCODER_FILENAME).exists(), f"Missing {ENCODER_FILENAME}"

        with (folder / CONFIG_FILENAME).open("r") as f:
            encoder_config = json.load(f)["model"]["encoder"]

        encoder = cls(**encoder_config)
        encoder.load_state_dict(torch.load(folder / ENCODER_FILENAME, map_location=device))
        return encoder


class PrestoPixelDecoder(FlexiPrestoBase):
    cross_attn = True

    def __init__(
        self,
        encoder_embedding_size: int = 128,
        decoder_embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        max_patch_size: int = 8,
    ):
        super().__init__(
            decoder_embedding_size,
            depth,
            mlp_ratio,
            num_heads,
            max_sequence_length,
            max_patch_size,
        )
        self.encoder_to_decoder_embed = nn.Linear(
            encoder_embedding_size, decoder_embedding_size, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(decoder_embedding_size))

        self.max_patch_size = max_patch_size
        self.space_time_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embedding_size, len(group) * max_patch_size**2)
                for group_name, group in self.space_time_groups.items()
            }
        )
        self.space_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embedding_size, len(group) * max_patch_size**2)
                for group_name, group in self.space_groups.items()
            }
        )
        self.time_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embedding_size, len(group))
                for group_name, group in self.time_groups.items()
            }
        )
        self.norm = nn.LayerNorm(decoder_embedding_size)

    def add_masks(self, s_t_x, s_x, t_x, s_t_m, s_m, t_m):
        s_t_x = s_t_x * (1 - s_t_m).unsqueeze(-1)
        B, H, W, T, S_T_C, _ = s_t_x.shape
        s_t_m_reshaped = repeat(self.mask_token, "d -> b h w t c d", b=B, h=H, w=W, t=T, c=S_T_C)
        s_t_m_add = s_t_m_reshaped * s_t_m.unsqueeze(-1)

        s_x = s_x * (1 - s_m).unsqueeze(-1)
        S_C = s_x.shape[-2]
        s_m_reshaped = repeat(self.mask_token, "d -> b h w c d", b=B, h=H, w=W, c=S_C)
        s_m_add = s_m_reshaped * s_m.unsqueeze(-1)

        t_x = t_x * (1 - t_m).unsqueeze(-1)
        T_C = t_x.shape[-2]
        t_m_reshaped = repeat(self.mask_token, "d -> b t c d", b=B, t=T, c=T_C)
        t_m_add = t_m_reshaped * t_m.unsqueeze(-1)

        return s_t_x + s_t_m_add, s_x + s_m_add, t_x + t_m_add

    def forward(
        self,
        s_t_x: torch.Tensor,
        s_x: torch.Tensor,
        t_x: torch.Tensor,
        s_t_m: torch.Tensor,
        s_m: torch.Tensor,
        t_m: torch.Tensor,
        months: torch.Tensor,
        patch_size: Optional[int] = None,
        input_resolution_m: Optional[int] = BASE_GSD,
    ):
        s_t_x = self.encoder_to_decoder_embed(s_t_x)
        s_x = self.encoder_to_decoder_embed(s_x)
        t_x = self.encoder_to_decoder_embed(t_x)

        s_t_x, s_x, t_x = self.add_masks(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
        s_t_x, s_x, t_x, s_t_m, s_m, t_m = self.apply_attn(
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size, input_resolution_m
        )
        output_s_t, output_s, output_t = [], [], []
        for idx, (group_name, c_g) in enumerate(self.space_time_groups.items()):
            # decoded has shape [b, h, w, t, len(c_g) * patch_size ** 2]
            decoded = self.space_time_embed[group_name](self.norm(s_t_x[:, :, :, :, idx]))
            output_s_t.append(
                rearrange(
                    decoded,
                    "b t_h t_w t (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) t c_g",
                    c_g=len(c_g),
                    p_w=self.max_patch_size,
                    p_h=self.max_patch_size,
                )
            )

        for idx, (group_name, c_g) in enumerate(self.space_groups.items()):
            # decoded has shape [b, h, w, len(c_g) * patch_size ** 2]
            decoded = self.space_embed[group_name](self.norm(s_x[:, :, :, idx]))
            output_s.append(
                rearrange(
                    decoded,
                    "b t_h t_w (c_g p_h p_w) -> b (t_h p_h) (t_w p_w) c_g",
                    c_g=len(c_g),
                    p_w=self.max_patch_size,
                    p_h=self.max_patch_size,
                )
            )

        for idx, (group_name, c_g) in enumerate(self.time_groups.items()):
            decoded = self.time_embed[group_name](self.norm(t_x[:, :, idx]))
            output_t.append(decoded)

        return (
            torch.cat(output_s_t, dim=-1),
            torch.cat(output_s, dim=-1),
            torch.cat(output_t, dim=-1),
        )
