import warnings
from typing import List

import torch.nn.init as init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConditionalLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        """
        This design follows nn.Linear as close as possible:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

        Equivalent to nn.Linear if conditional_weights is None or contain all zeroes

        (Renamed from LoRALinear to ConditionalLinear since the Conditioner now supports full-rank)
        """
        super(ConditionalLinear, self).__init__()
        self.backbone = nn.Linear(in_features, out_features, bias=bias)
        self.conditional_weights = None

    def apply_condition(self, conditional_weights):
        self.conditional_weights = conditional_weights
        if self.conditional_weights is not None:
            if self.conditional_weights.shape != self.backbone.weight.shape:
                raise ValueError(
                    f"conditional_weights must have the same shape ({self.conditional_weights.shape}) as backbone.weight ({self.backbone.weight.shape})"
                )

    def forward(self, x):
        if self.conditional_weights is not None:
            return F.linear(x, self.backbone.weight + self.conditional_weights, self.backbone.bias)
        else:
            return F.linear(x, self.backbone.weight, self.backbone.bias)


class TokenConditioner(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        time_min: int,  # in timesteps
        time_max: int,  # in timesteps
        hw_min: int,  # in pixels per side
        hw_max: int,  # in pixels per side
        patch_size_min: int,  # in pixels per side
        patch_size_max: int,  # in pixels per side
        num_input_channels: int,  # channel *groups*
        num_output_channels: int,  # channel *groups*
        num_recon_objs: int,  # number of reconstructive pretraining objectives
    ):
        super().__init__()

        self.backbone_dim = backbone_dim
        self.time_min = time_min
        self.time_max = time_max
        self.hw_min = hw_min
        self.hw_max = hw_max
        self.patch_size_min = patch_size_min
        self.patch_size_max = patch_size_max
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_recon_objs = num_recon_objs
 
        ##### create conditioner network parameters #####
        self.embedder = nn.Linear(
            3 + num_input_channels + num_output_channels + num_recon_objs, backbone_dim
        )  # 3 is from input shape, i.e., height/width, time, patch size

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def normalize_input_shape(self, hw, patch_size, timesteps, device, dtype):
        if hw < self.hw_min or hw > self.hw_max:
            warnings.warn(
                f"hw ({hw}) is outside the expected range [{self.hw_min}, {self.hw_max}]"
            )
        if patch_size < self.patch_size_min or patch_size > self.patch_size_max:
            warnings.warn(
                f"patch_size ({patch_size}) is outside the expected range [{self.patch_size_min}, {self.patch_size_max}]"
            )
        if timesteps < self.time_min or timesteps > self.time_max:
            warnings.warn(
                f"timesteps ({timesteps}) is outside the expected range [{self.time_min}, {self.time_max}]"
            )

        hw_normalized = (hw - self.hw_min) / (self.hw_max - self.hw_min)
        patch_size_normalized = (patch_size - self.patch_size_min) / (
            self.patch_size_max - self.patch_size_min
        )
        timesteps_normalized = (timesteps - self.time_min) / (self.time_max - self.time_min)

        return torch.tensor(
            [hw_normalized, patch_size_normalized, timesteps_normalized], device=device, dtype=dtype
        )
    
    def forward(
        self,
        hw: int,
        patch_size: int,
        timesteps: int,
        input_channels: torch.Tensor,  # multihot encoding
        output_channels: torch.Tensor,  # multihot encoding
        recon_objs: torch.Tensor,  # multihot encoding
    ):
        normalized_input_shape = self.normalize_input_shape(
            hw, patch_size, timesteps, input_channels.device, input_channels.dtype
        )
        condition = torch.cat(
            [normalized_input_shape, input_channels, output_channels, recon_objs]
        )  # shape (3 + num_input_channels + num_output_channels + num_recon_objs)
        condition = rearrange(condition, 'd -> 1 1 d')

        return self.embedder(condition)  # shape (1, 1, dim)