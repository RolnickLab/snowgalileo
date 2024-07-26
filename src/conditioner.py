from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import functional as F


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
            return F.linear(x, (self.backbone.weight + self.conditional_weights) / 2, self.backbone.bias)
        else:
            return F.linear(x, self.backbone.weight, self.backbone.bias)


class LearnedMixture(nn.Module):
    def __init__(self, num_output_channels: int):
        super().__init__()
        self.num_templates = num_output_channels
        self.e_templates: nn.ModuleList = nn.ModuleList()

    def add_templates(self, template: nn.Module):
        self.e_templates = nn.ModuleList([deepcopy(template) for _ in range(self.num_templates)])
        for t in self.e_templates:
            t.apply(t._init_weights)

    def forward(self, output_channels: torch.Tensor):
        assert sum(output_channels) == 1, f"Expected one hot encoding got {output_channels}"
        assert len(output_channels) == self.num_templates
        return {
            key: val
            for key, val in self.e_templates[
                torch.argwhere(output_channels).item()
            ].named_parameters()
        }
