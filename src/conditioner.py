from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn import functional as F


class NonConditionalMlp(nn.Module):
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

    def apply_condition(self, conditional_weights, conditional_bias, mode):
        self.mode = mode
        self.conditional_weights = conditional_weights
        self.conditional_bias = conditional_bias
        if self.conditional_weights is not None:
            if self.conditional_weights.shape != self.backbone.weight.shape:
                raise ValueError(
                    f"conditional_weights must have the same shape ({self.conditional_weights.shape}) as backbone.weight ({self.backbone.weight.shape})"
                )

            if self.mode == "moe":
                assert self.conditional_bias is not None
                if self.conditional_bias.shape != self.backbone.bias.shape:
                    raise ValueError(
                        f"conditional_bias must have the same shape ({self.conditional_bias.shape}) as backbone.weight ({self.backbone.bias.shape})"
                    )

    def forward(self, x):
        if self.conditional_weights is not None:
            if self.mode == "moe":
                assert self.conditional_bias is not None
                return F.linear(
                    x,
                    (self.backbone.weight + self.conditional_weights) / 2,
                    (self.backbone.bias + self.conditional_bias) / 2,
                )
            elif self.mode == "lora":
                return F.linear(
                    x,
                    self.backbone.weight + self.conditional_weights,
                    self.backbone.bias,
                )
            else:
                raise f"mode must be moe or lora, not {self.mode}"

        else:
            return F.linear(x, self.backbone.weight, self.backbone.bias)


class LearnedMixture(nn.Module):
    def __init__(self, num_output_channels: int):
        super().__init__()
        self.mode = "moe"
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

    def forward(self, c_i):
        output_channels = c_i["output_channels"]
        assert len(output_channels) == self.num_templates
        selected_templates = []
        for i in torch.argwhere(output_channels):
            selected_templates.append(self.templates[i])
        return self.average_modules(selected_templates)


class LoRAGenerator(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        backbone_depth: int,
        rank: int,
        num_output_channels: int,
        mlp_ratio: int,
        param_types: List[str] = ["q", "k", "v"],
    ):
        super().__init__()

        self.mode = "lora"
        self.mlp_ratio = mlp_ratio
        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.rank = rank
        self.num_channels = num_output_channels
        self.param_types = param_types

        self.loras = nn.ParameterDict()
        for idx in range(num_output_channels):
            for dim in range(self.backbone_dim):
                for param_type in param_types:
                    in_dim, out_dim = (
                        self.get_param_type_dims(param_type)["input_dim"],
                        self.get_param_type_dims(param_type)["output_dim"],
                    )
                    self.loras[f"{idx}_{dim}_{param_type}_a"] = nn.Parameter(
                        torch.zeros((2 * rank, in_dim))
                    )
                    self.loras[f"{idx}_{dim}_{param_type}_b"] = nn.Parameter(
                        torch.zeros((2 * rank, out_dim))
                    )

    def get_param_type_dims(self, param_type):
        if param_type in ["q", "k", "v", "proj"]:
            in_dim = self.backbone_dim
            out_dim = self.backbone_dim
        elif param_type == "fc1":
            in_dim = self.backbone_dim
            out_dim = int(self.backbone_dim * 4)  # Using MLP ratio of 4
        elif param_type == "fc2":
            in_dim = int(self.backbone_dim * 4)
            out_dim = self.backbone_dim
        else:
            raise ValueError(
                f"Invalid param_type {param_type}. Must be q, k, v, proj, fc1, or fc2."
            )
        return {"input_dim": in_dim, "output_dim": out_dim}

    def get_lora_weights(self, channel_idx, backbone_dim, param_type):
        # Project input weights
        a = self.loras[f"{channel_idx}_{backbone_dim}_{param_type}_a"]
        b = self.loras[f"{channel_idx}_{backbone_dim}_{param_type}_b"]
        # Compute the low-rank weight matrix
        return torch.matmul(a, b) / (self.rank**0.5)

    @staticmethod
    def average_loras(weights: List[Dict]):
        output_dict = {}
        weight = 1 / len(weights)
        for key in weights[0].keys():
            new_weight = sum([weight * weights[i][key] for i in range(len(weights))])
            assert new_weight is not None, f"{key} is None"
            output_dict[key] = new_weight

    def forward(self, c_i):
        output_channels = c_i["output_channels"]
        assert (
            len(self.loras) == 2 * len(output_channels) * len(self.param_types) * self.backbone_dim
        )
        output_loras = []
        for idx, channel_idx in enumerate(torch.argwhere(output_channels)):
            output_loras.append(dict())
            for dim in range(self.backbone_dim):
                for param_type in self.param_types:
                    output_loras[idx][f"{dim}_{param_type}"] = self.get_lora_weights(
                        channel_idx, dim, param_type
                    )
        return self.average_loras(output_loras)
