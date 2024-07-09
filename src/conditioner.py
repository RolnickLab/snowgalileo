import warnings
from typing import List

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


class LearnedMixture(nn.Module):
    def __init__(
        self,
        dim: int,
        backbone_dim: int,
        backbone_depth: int,
        num_templates: int,
        proj_share_depth: bool,
        rank: int,
        time_min: int,  # in timesteps
        time_max: int,  # in timesteps
        hw_min: int,  # in pixels per side
        hw_max: int,  # in pixels per side
        patch_size_min: int,  # in pixels per side
        patch_size_max: int,  # in pixels per side
        num_input_channels: int,  # channel *groups*
        num_output_channels: int,  # channel *groups*
        num_recon_objs: int,  # number of reconstructive pretraining objectives
        init_scale: float = 0.001,
        template_init_std: float = 0.2,
        softmax_temp: float = 1.0,
        param_types: List[str] = ["q", "k", "v", "proj", "fc1", "fc2"],
    ):
        super().__init__()
        if rank == backbone_dim:
            self.use_lora = False
            print("rank equal to backbone dim, no LoRAs!")
        else:
            self.use_lora = True
            print("rank *not* equal to backbone dim, yes LoRAs!")

        assert (
            rank <= backbone_dim
        ), f"rank ({rank}) must be less than or equal to backbone_dim ({backbone_dim})"

        self.dim = dim
        self.backbone_dim = backbone_dim
        self.backbone_depth = backbone_depth
        self.num_templates = num_templates
        self.proj_share_depth = proj_share_depth
        self.init_scale = init_scale
        self.rank = rank
        self.time_min = time_min
        self.time_max = time_max
        self.hw_min = hw_min
        self.hw_max = hw_max
        self.patch_size_min = patch_size_min
        self.patch_size_max = patch_size_max
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_recon_objs = num_recon_objs
        self.softmax_temp = softmax_temp
        self.param_types = param_types

        ##### create conditioner network parameters #####
        self.embedder = nn.Linear(
            3 + num_input_channels + num_output_channels + num_recon_objs, dim
        )  # 3 is from input shape, i.e., height/width, time, patch size
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, int(4 * dim)),
                    nn.GELU(),
                    nn.Linear(int(4 * dim), dim),
                )
                for _ in range(backbone_depth)
            ]
        )

        ##### create template projections that determine mixtures #####
        self.template_projections = nn.ModuleDict(
            {f"layer_{i}": nn.ModuleDict() for i in range(backbone_depth)}
        )

        for param_type in param_types:
            if proj_share_depth:
                # create a single shared linear layer for all depths
                shared_linear = nn.Linear(dim, num_templates)
                for layer_idx in range(backbone_depth):
                    self.template_projections[f"layer_{layer_idx}"][param_type] = shared_linear
            else:
                # create a separate linear layer for each depth
                for layer_idx in range(backbone_depth):
                    self.template_projections[f"layer_{layer_idx}"][param_type] = nn.Linear(
                        dim, num_templates
                    )

        ##### create templates #####
        self.init_std = template_init_std
        # check if we need attention templates
        if any(param in self.param_types for param in ["q", "k", "v", "proj"]):
            if self.use_lora:
                self.attention_templates_A = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.backbone_dim, self.rank),
                    )
                )
                self.attention_templates_B = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.rank, self.backbone_dim),
                    )
                )
            else:
                self.attention_templates = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.backbone_dim, self.backbone_dim),
                    )
                )
        else:
            self.attention_templates = None
            self.attention_templates_A = None
            self.attention_templates_B = None

        # check if we need FFN templates
        if any(param in self.param_types for param in ["fc1", "fc2"]):
            if self.use_lora:
                self.ffn_templates_A = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.backbone_dim * 4, self.rank),
                    )
                )
                self.ffn_templates_B = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.rank, self.backbone_dim),
                    )
                )
            else:
                self.ffn_templates = nn.Parameter(
                    torch.normal(
                        mean=0,
                        std=self.init_std,
                        size=(self.num_templates, self.backbone_dim * 4, self.backbone_dim),
                    )
                )
        else:
            self.ffn_templates = None
            self.ffn_templates_A = None
            self.ffn_templates_B = None

        if self.use_lora:
            assert (
                (self.ffn_templates_A is not None and self.ffn_templates_B is not None)
                or (
                    self.attention_templates_A is not None
                    and self.attention_templates_B is not None
                )
            ), f"No LoRA templates initialized. Check if param_types {self.param_types} includes any of q, k, v, proj, fc1, or fc2."
        else:
            assert (
                self.ffn_templates is not None or self.attention_templates is not None
            ), f"No full-rank templates initialized. Check if param_types {self.param_types} includes any of q, k, v, proj, fc1, or fc2."

        # Initialize scales with very low values allowing us to initialize all conditional parameters very close to zero
        self.scales = nn.ParameterDict()
        for param_type in self.param_types:
            if param_type in ["q", "k", "v", "proj"]:
                self.scales[param_type] = nn.Parameter(
                    torch.full((self.backbone_dim, self.backbone_dim), init_scale)
                )
            elif param_type == "fc1":
                self.scales[param_type] = nn.Parameter(
                    torch.full((self.backbone_dim * 4, self.backbone_dim), init_scale)
                )
            elif param_type == "fc2":
                self.scales[param_type] = nn.Parameter(
                    torch.full((self.backbone_dim, self.backbone_dim * 4), init_scale)
                )

        self.apply(self._init_weights)
        self.last_mean = 0.0
        self.last_std = 0.0

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
    
    def get_last_stats(self):
        return self.last_mean, self.last_std

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
        ).unsqueeze(
            dim=0
        )  # shape (1, 3 + num_input_channels + num_output_channels + num_recon_objs)

        x = self.embedder(condition)  # shape (1, dim)

        mixed_templates = {}
        for layer_idx, block in enumerate(self.blocks):
            x = block(x) + x

            # Compute mixture values for each parameter type
            mixture_values = {}
            for param_type in self.param_types:
                mixture_logits = self.template_projections[f"layer_{layer_idx}"][param_type](x)
                mixture_logits = rearrange(
                    mixture_logits, "1 l -> l 1 1"
                )  # prepare for broadcasting
                mixture_values[param_type] = torch.softmax(
                    mixture_logits / self.softmax_temp, dim=0
                )

            # Apply mixtures to templates
            layer_results = {}
            for param_type in self.param_types:
                if param_type in ["q", "k", "v", "proj"]:
                    if self.use_lora:
                        templates = torch.bmm(
                            self.attention_templates_A, self.attention_templates_B
                        )
                    else:
                        templates = self.attention_templates

                elif param_type in ["fc1", "fc2"]:
                    if self.use_lora:
                        templates = torch.bmm(self.ffn_templates_A, self.ffn_templates_B)
                    else:
                        templates = self.ffn_templates

                    if param_type == "fc2":
                        # transpose templates for down projection
                        templates = rearrange(templates, "n i o -> n o i")

                layer_results[param_type] = (mixture_values[param_type] * templates).sum(
                    dim=0
                ) * self.scales[param_type]  # weighted sum and scale

            mixed_templates[layer_idx] = layer_results
        
        # compute the mean and std of the conditional weights and set an attribute so we have access during training
        all_values = []
        for layer_results in mixed_templates.values():
            for param_value in layer_results.values():
                all_values.append(param_value.flatten())
        
        all_values = torch.cat(all_values)
        self.last_mean = all_values.mean().item()
        self.last_std = all_values.std().item()

        return mixed_templates