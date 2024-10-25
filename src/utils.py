import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from .config import DEFAULT_SEED
from .data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
)
from .masking import MASKING_MODES, UNMASKING_CHANNEL_GROUPS, MaskedOutput

data_dir = Path(__file__).parent.parent / "data"
logging_dir = Path(__file__).parent.parent / "logs"
config_dir = Path(__file__).parent.parent / "config"

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def will_cause_nans(x: torch.Tensor):
    return torch.isnan(x).any() or torch.isinf(x).any()


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def masked_output_np_to_tensor(
    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, month
) -> MaskedOutput:
    """converts eval task"""
    return MaskedOutput(
        torch.as_tensor(s_t_x, dtype=torch.float32),
        torch.as_tensor(sp_x, dtype=torch.float32),
        torch.as_tensor(t_x, dtype=torch.float32),
        torch.as_tensor(st_x, dtype=torch.float32),
        torch.as_tensor(s_t_m, dtype=torch.float32),
        torch.as_tensor(sp_m, dtype=torch.float32),
        torch.as_tensor(t_m, dtype=torch.float32),
        torch.as_tensor(st_m, dtype=torch.float32),
        torch.as_tensor(month, dtype=torch.long),
    )


class AverageMeter:
    """computes and stores the average and current value"""

    average: float
    sum: float
    count: int

    def __init__(self):
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def check_config(config):
    expected_training_keys_type = {
        "num_epochs": int,
        "batch_size": int,
        "effective_batch_size": int,
        "encode_ratio": float,
        "decode_ratio": float,
        "patch_sizes": list,
        "max_lr": float,
        "final_lr": float,
        "conditioner_multiplier": float,
        "warmup_epochs": (int, float),
        "eval_eurosat_every_n_epochs": int,
        "shape_time_combinations": list,
        "augmentation": dict,
        "masking_probabilities": list,
        "grad_clip": bool,
        "target_condition": bool,
        "target_exit_after": (int, str),
        "conditioner_mode": str,
        "normalization": str,
        "random_masking": str,
    }
    training_dict = config["training"]

    for key, val in expected_training_keys_type.items():
        assert key in training_dict, f"Expected {key} in training dict"
        assert isinstance(
            training_dict[key],
            val,  # type: ignore
        ), f"Expected {key} to be {val}, got {type(training_dict[key])}"

    if isinstance(training_dict["warmup_epochs"], float):
        training_dict["warmup_epochs"] = int(
            training_dict["warmup_epochs"] * training_dict["num_epochs"]
        )
    assert isinstance(training_dict["warmup_epochs"], int)
    assert training_dict["num_epochs"] > training_dict["warmup_epochs"]
    assert training_dict["normalization"] in ["std", "scaling"]
    assert training_dict["random_masking"] in ["half", "full", "none"]

    assert len(training_dict["masking_probabilities"]) == len(
        MASKING_MODES
    ), f"Expected {len(MASKING_MODES)}, got {len(training_dict['masking_probabilities'])}"

    for combination in training_dict["shape_time_combinations"]:
        assert "timesteps" in combination.keys()
        assert "size" in combination.keys()
        assert combination["timesteps"] >= 3

    expected_encoder_decoder_keys_type = {
        "embedding_size": int,
        "depth": int,
        "mlp_ratio": int,
        "num_heads": int,
        "max_sequence_length": int,
    }

    expected_encoder_only_keys_type = {"freeze_projections": bool, "drop_path": float}
    expected_decoder_only_keys_type = {"learnable_channel_embeddings": bool}

    model_dict = config["model"]
    for model in ["encoder", "decoder"]:
        assert model in model_dict
        for key, val in expected_encoder_decoder_keys_type.items():
            assert key in model_dict[model], f"Expected {key} in {model} dict"
            assert isinstance(model_dict[model][key], val)
        if model == "encoder":
            for key, val in expected_encoder_only_keys_type.items():
                assert key in model_dict[model], f"Expected {key} in {model} dict"
                assert isinstance(model_dict[model][key], val)
        elif model == "decoder":
            for key, val in expected_decoder_only_keys_type.items():
                assert key in model_dict[model], f"Expected {key} in {model} dict"
                assert isinstance(model_dict[model][key], val)

    config["model"]["encoder"]["max_patch_size"] = max(config["training"]["patch_sizes"])
    config["model"]["decoder"]["max_patch_size"] = max(config["training"]["patch_sizes"])
    config["model"]["decoder"]["encoder_embedding_size"] = config["model"]["encoder"][
        "embedding_size"
    ]
    config["model"]["decoder"]["decoder_embedding_size"] = config["model"]["decoder"].pop(
        "embedding_size"
    )

    if config["training"]["loss_type"] == "MAE":
        max_patch_size = max(config["training"]["patch_sizes"])
        max_group_length = max(
            [
                max([len(k) for k, v in SPACE_TIME_BANDS_GROUPS_IDX]),
                max([len(k) for k, v in TIME_BAND_GROUPS_IDX]),
                max([len(k) for k, v in SPACE_BAND_GROUPS_IDX]),
                max([len(k) for k, v in STATIC_BAND_GROUPS_IDX]),
            ]
        )
        config["model"]["encoder"]["output_embedding_size"] = (
            max_patch_size**2
        ) * max_group_length

    assert config["training"]["conditioner_mode"] in [
        "moe",
        "lora-t",
        "lora-g",
        "no_cond",
        "token",
    ]

    if config["training"]["conditioner_mode"] == "moe":
        config["model"]["conditioner"] = {"num_output_channels": len(UNMASKING_CHANNEL_GROUPS)}

    elif "lora" in config["training"]["conditioner_mode"]:
        config["model"]["conditioner"]["num_output_channels"] = len(UNMASKING_CHANNEL_GROUPS)
        config["model"]["conditioner"]["backbone_dim"] = config["model"]["encoder"][
            "embedding_size"
        ]
        config["model"]["conditioner"]["backbone_depth"] = config["model"]["encoder"]["depth"]
        config["model"]["conditioner"]["mlp_ratio"] = config["model"]["encoder"]["mlp_ratio"]
    elif config["training"]["conditioner_mode"] == "token":
        config["model"]["conditioner"] = {
            "num_output_channels": len(UNMASKING_CHANNEL_GROUPS),
            "backbone_dim": config["model"]["encoder"]["embedding_size"],
        }

    return config


def load_check_config(name: str) -> Dict:
    with (config_dir / "mae" / name).open("r") as f:
        config = json.load(f)
    config = check_config(config)

    return config


@torch.no_grad()
def plot_space_time_predictions(
    epoch,
    encoder,
    predictor,
    training_config,
    prepared_image,
    image_id,
):
    """
    Plots MAE input images, masks, MAE predictions, and difference of input and predictions.
    Number of timesteps to plot are defined in the training config.
    """
    (
        s_t_x,
        sp_x,
        t_x,
        st_x,
        s_t_m,
        sp_m,
        t_m,
        st_m,
        months,
        expanded_s_t_x,
        _,
        expanded_s_t,
        _,
        _,
        _,
        patch_size,
    ) = prepared_image

    # get predictions with current model
    (p_s_t, _, _, _) = predictor(
        *encoder(
            s_t_x.float(),
            sp_x.float(),
            t_x.float(),
            st_x.float(),
            s_t_m.float(),
            sp_m.float(),
            t_m.float(),
            st_m.float(),
            months.long(),
            patch_size=patch_size,
        ),
        patch_size=patch_size,
    )

    subplot_titles = []

    for band_list in SPACE_TIME_BANDS_GROUPS_IDX.values():
        for band in band_list:
            subplot_titles.append(SPACE_TIME_BANDS[band])

    plot_dict = {}

    for t in training_config["timesteps_to_wandb_plot"]:
        # figure columns: input, mask, prediction, error
        # figure rows: bands
        fig, axs = plt.subplots(len(subplot_titles), 4, figsize=(20, 45))

        # get min and max values for the error colorbar independent of the channel
        error_min = (
            (abs(expanded_s_t_x[:, :, :, t, :] - p_s_t[:, :, :, t, :]))
            * expanded_s_t[:, :, :, t, :]
        ).min()
        error_max = (
            (abs(expanded_s_t_x[:, :, :, t, :] - p_s_t[:, :, :, t, :]))
            * expanded_s_t[:, :, :, t, :]
        ).max()

        for i, band in enumerate(subplot_titles):
            x_to_plot = expanded_s_t_x[0, :, :, t, i].squeeze(0).cpu()
            pred_to_plot = p_s_t[0, :, :, t, i].squeeze(0).cpu()
            mask_to_plot = expanded_s_t[0, :, :, t, i].squeeze(0).cpu()

            x_plot = axs[i, 0].imshow(
                x_to_plot.numpy(), cmap="gray", vmin=x_to_plot.min(), vmax=x_to_plot.max()
            )
            axs[i, 0].set_title(f"Input {band}")
            fig.colorbar(x_plot, ax=axs[i, 0])
            mask_plot = axs[i, 1].imshow(mask_to_plot.numpy(), cmap="gray")
            axs[i, 1].set_title(f"Mask {band}")
            fig.colorbar(mask_plot, ax=axs[i, 1])
            pred_plot = axs[i, 2].imshow(
                (pred_to_plot * mask_to_plot).numpy(),
                cmap="gray",
                vmin=pred_to_plot.min(),
                vmax=pred_to_plot.max(),
            )
            axs[i, 2].set_title(f"Output {band}")
            fig.colorbar(pred_plot, ax=axs[i, 2])
            error = axs[i, 3].imshow(
                (abs(x_to_plot.numpy() - pred_to_plot.numpy())) * mask_to_plot.numpy(),
                cmap="coolwarm",
                vmin=error_min,
                vmax=error_max,
            )
            axs[i, 3].set_title(f"Input - Output {band}")
            fig.colorbar(error, ax=axs[i, 3])

        fig.suptitle(
            f"Plot image: {image_id}, epoch: {epoch}, patch size: {patch_size}, timestep: {t}",
            fontsize=20,
            y=1.0001,
        )
        fig.tight_layout()

        plot = wandb.Image(
            fig, caption=f"plot_image{image_id}_epoch{epoch}_patch_size{patch_size}_timestep{t}"
        )
        plot_dict[patch_size] = plot
    return plot_dict


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now(dateutil.tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts


def is_bf16_available():
    # https://github.com/huggingface/transformers/blob/d91841315aab55cf1347f4eb59332858525fad0f/src/transformers/utils/import_utils.py#L275
    # https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_cuda.py#L51
    # to succeed:
    # 1. the hardware needs to support bf16 (arch >= Ampere)
    # 2. torch >= 1.10 (1.9 should be enough for AMP API has changed in 1.10, so using 1.10 as minimal)
    # 3. CUDA >= 11
    # 4. torch.autocast exists
    # XXX: one problem here is that it may give invalid results on mixed gpus setup, so it's
    # really only correct for the 0th gpu (or currently set default device if different from 0)

    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split(".")[0]) < 11:
        return False
    if not hasattr(torch, "autocast"):
        return False

    return True
