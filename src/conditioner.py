from copy import deepcopy
from typing import List

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
        self.conditional_bias = None

    def apply_condition(self, conditional_weights, conditional_bias):
        self.conditional_weights = conditional_weights
        self.conditional_bias = conditional_bias
        if self.conditional_weights is not None:
            if self.conditional_weights.shape != self.backbone.weight.shape:
                raise ValueError(
                    f"conditional_weights must have the same shape ({self.conditional_weights.shape}) as backbone.weight ({self.backbone.weight.shape})"
                )
            assert self.conditional_bias is not None
            if self.conditional_bias.shape != self.backbone.bias.shape:
                raise ValueError(
                    f"conditional_bias must have the same shape ({self.conditional_bias.shape}) as backbone.weight ({self.backbone.bias.shape})"
                )

    def forward(self, x):
        if self.conditional_weights is not None:
            assert self.conditional_bias is not None
            return F.linear(
                x,
                (self.backbone.weight + self.conditional_weights) / 2,
                (self.backbone.bias + self.conditional_bias) / 2,
            )
        else:
            return F.linear(x, self.backbone.weight, self.backbone.bias)


class LearnedMixture(nn.Module):
    def __init__(self, num_output_channels: int):
        super().__init__()
        self.num_templates = num_output_channels
        self.templates: nn.ModuleList = nn.ModuleList()

    def add_templates(self, template: nn.Module):
        self.templates = nn.ModuleList([deepcopy(template) for _ in range(self.num_templates)])
        # for t in self.e_templates:
        #     t.apply(t._init_weights)

    @staticmethod
    def average_modules(templates: List[nn.Module]):
        output_dict = {}
        weight = 1 / len(templates)
        for ts in zip(*[t.named_parameters() for t in templates]):
            name = ts[0][0]
            new_weight = sum([weight * ts[i][1] for i in range(len(ts))])
            assert new_weight is not None, f"{name} is None"
            output_dict[name] = new_weight
        return output_dict

    def forward(self, output_channels: torch.Tensor):
        assert len(output_channels) == self.num_templates
        selected_templates = []
        for i in torch.argwhere(output_channels):
            selected_templates.append(self.templates[i])
        return self.average_modules(selected_templates)
