from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from timm.layers import Mlp


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

    def forward(self, output_channels: torch.Tensor):
        assert len(output_channels) == self.num_templates
        selected_templates = []
        for i in torch.argwhere(output_channels):
            selected_templates.append(self.templates[i])
        return self.average_modules(selected_templates)


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-4,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
    

class MLPBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm = norm_layer(dim)
        self.ls = LayerScale(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls(self.mlp(self.norm(x)))
        return x


class LoRAGenerator(nn.Module):
    def __init__(
        self,
        dim: int, 
        backbone_dim: int,
        rank: int,
        num_output_channels: int,
        param_types: List[str] = ["q", "k", "v"],
    ):
        super().__init__()

        self.mode = "lora"
        self.dim = dim
        self.backbone_dim = backbone_dim
        self.rank = rank
        self.num_channels = num_output_channels
        self.param_types = param_types
        self.sequence_length = sum([2 * rank for _ in param_types])
 
        ##### create conditioner network parameters #####
        self.target_embedder = nn.Linear(self.num_channels, backbone_dim)
        self.embedding = nn.Embedding(self.sequence_length, dim)
        self.blocks = nn.Sequential(*[MLPBlock(dim=dim) for _ in range(4)])
        
        # Create output projections
        self.output_projections = nn.ModuleDict()
        for param_type in param_types:
            in_dim, out_dim = self.get_param_type_dims(param_type)["input_dim"], self.get_param_type_dims(param_type)["output_dim"]
            self.output_projections[f"{param_type}_in"] = nn.Linear(dim, in_dim)
            self.output_projections[f"{param_type}_out"] = nn.Linear(dim, out_dim)

            # Initialize weights to 0 for the "_out" projections
            nn.init.zeros_(self.output_projections[f"{param_type}_out"].weight)
            nn.init.zeros_(self.output_projections[f"{param_type}_out"].bias)
            nn.init.xavier_uniform_(self.output_projections[f"{param_type}_in"].weight, gain=0.1)
            nn.init.zeros_(self.output_projections[f"{param_type}_in"].bias)

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
            raise ValueError(f"Invalid param_type {param_type}. Must be q, k, v, proj, fc1, or fc2.")
        return {"input_dim": in_dim, "output_dim": out_dim}

    def forward(self, output_channels: torch.Tensor):
        output_channels = rearrange(output_channels, 'c -> 1 1 c')  # (1, 1, c_g)
        condition = self.target_embedder(output_channels)  # (1, 1, dim)
        condition = condition.repeat(1, self.sequence_length, 1)  # (1, self.sequence_length, dim)

        position_ids = torch.arange(self.sequence_length, device=output_channels.device).unsqueeze(0)
        embeddings = self.embedding(position_ids)  # (1, self.sequence_length, dim)

        x = condition + embeddings
        x = self.blocks(x)

        # generate lora_weights
        lora_weights = {}
        start_idx = 0
        for param_type in self.param_types:               
            # Project input weights
            input_weights = self.output_projections[f"{param_type}_in"](x[:, start_idx:start_idx+self.rank, :])
            input_weights = input_weights.squeeze(0)  # Shape: (rank, in_dim)
            
            # Project output weights
            output_weights = self.output_projections[f"{param_type}_out"](x[:, start_idx+self.rank:start_idx+2*self.rank, :])
            output_weights = output_weights.squeeze(0).t()  # Shape: (out_dim, rank)
            
            # Compute the low-rank weight matrix
            lora_weights[param_type] = torch.matmul(output_weights, input_weights) / (self.rank ** 0.5)

            start_idx += 2 * self.rank

        return lora_weights