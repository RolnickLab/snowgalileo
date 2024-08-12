from copy import deepcopy
from typing import List
import warnings

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

    def forward(self, c_i):
        output_channels = c_i["output_channels"]
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
        backbone_depth: int,
        do_input_condition: bool,
        rank: int,
        num_output_channels: int,
        param_types: List[str] = ["q", "k", "v"],
    ):
        super().__init__()

        self.mode = "lora"
        self.dim = dim
        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.rank = rank
        self.num_channels = num_output_channels
        self.param_types = param_types
        self.do_input_condition = do_input_condition
        self.sequence_length = sum([2 * rank for _ in param_types])
 
        ##### create conditioner network parameters #####
        if do_input_condition:
            condition_length = 2 * self.num_channels + 3
            self.hw_min = 1
            self.hw_max = 20
            self.patch_size_min = 1
            self.patch_size_max = 8
            self.time_min = 1
            self.time_max = 12

        else:
            condition_length = self.num_channels

        self.condition_proj = nn.Linear(condition_length, backbone_dim)
        self.embedding = nn.Embedding(self.sequence_length, dim)
        self.blocks_before = nn.Sequential(*[MLPBlock(dim=dim) for _ in range(2)])
        self.blocks_during = nn.Sequential(*[MLPBlock(dim=dim) for _ in range(backbone_depth)])
        
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
    
    def get_lora_weights(self, x):
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

    def forward(self, c_i):
        if self.do_input_condition:
            normalized_input_shape = self.normalize_input_shape(
                c_i["hw"], c_i["patch_size"], c_i["timesteps"], c_i["input_channels"].device, c_i["input_channels"].dtype
            )
            condition = torch.cat(
                [normalized_input_shape, c_i["input_channels"], c_i["output_channels"]]
            )
        else:
            condition = c_i["output_channels"]

        condition = rearrange(condition, 'c -> 1 1 c')  # (1, 1, N)
        condition = self.condition_proj(condition)  # (1, 1, dim)
        condition = condition.repeat(1, self.sequence_length, 1)  # (1, self.sequence_length, dim)

        position_ids = torch.arange(self.sequence_length, device=c_i["output_channels"].device).unsqueeze(0)
        embeddings = self.embedding(position_ids)  # (1, self.sequence_length, dim)

        x = condition + embeddings
        x = self.blocks_before(x)

        all_lora_weights = {}
        for block_idx, block in enumerate(self.blocks_during):
            x = block(x)
            all_lora_weights[block_idx] = self.get_lora_weights(x)
            
        return all_lora_weights