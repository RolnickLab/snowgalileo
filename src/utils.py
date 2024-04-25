import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.transforms.functional import resize

import wandb

from .config import DEFAULT_SEED
from .data.dataset import (
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
)
from .masking import (
    SPACE_TIME_BAND_EXPANSION_T,
    MaskedOutput,
    batch_mask_presto,
    subset_batch_of_images,
)

data_dir = Path(__file__).parent.parent / "data"
logging_dir = Path(__file__).parent.parent / "logs"
config_dir = Path(__file__).parent.parent / "config"

if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def masked_output_np_to_tensor(s_t_x, s_x, t_x, s_t_m, s_m, t_m, month) -> MaskedOutput:
    """converts eval task"""
    return MaskedOutput(
        torch.as_tensor(s_t_x, dtype=torch.float32),
        torch.as_tensor(s_x, dtype=torch.float32),
        torch.as_tensor(t_x, dtype=torch.float32),
        torch.as_tensor(s_t_m, dtype=torch.float32),
        torch.as_tensor(s_m, dtype=torch.float32),
        torch.as_tensor(t_m, dtype=torch.float32),
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


def load_check_config(name: str, mode: str):
    assert mode in ["mae", "jepa"]

    with (config_dir / mode / name).open("r") as f:
        config = json.load(f)

    expected_training_keys_type = {
        "num_epochs": int,
        "batch_size": int,
        "mask_ratio": float,
        "patch_sizes": list,
        "start_lr": float,
        "max_lr": float,
        "final_lr": float,
        "warmup_epochs": (int, float),
        "eval_eurosat_every_n_epochs": int,
        "time_ratio": float,
        "space_ratio": float,
        "spatial_patches_per_dim": int,
        "wandb_plot_every_n_epochs": int,
        "num_images_to_wandb_plot": int,
        "timesteps_to_wandb_plot": list,
        "patch_sizes_to_wandb_plot": list,
    }
    if mode == "jepa":
        expected_training_keys_type["ema"] = list
    training_dict = config["training"]

    for key, val in expected_training_keys_type.items():
        assert key in training_dict, f"Expected {key} in training dict"
        assert isinstance(training_dict[key], val)  # type: ignore

    if isinstance(training_dict["warmup_epochs"], float):
        training_dict["warmup_epochs"] = int(
            training_dict["warmup_epochs"] * training_dict["num_epochs"]
        )
    assert isinstance(training_dict["warmup_epochs"], int)
    assert training_dict["num_epochs"] > training_dict["warmup_epochs"]

    expected_encoder_decoder_keys_type = {
        "embedding_size": int,
        "depth": int,
        "mlp_ratio": int,
        "num_heads": int,
        "max_sequence_length": int,
    }

    model_dict = config["model"]
    for model in ["encoder", "decoder"]:
        assert model in model_dict
        for key, val in expected_encoder_decoder_keys_type.items():
            assert key in model_dict[model], f"Expected {key} in {model} dict"
            assert isinstance(model_dict[model][key], val)

    config["model"]["encoder"]["max_patch_size"] = config["training"]["patch_sizes"][-1]
    config["model"]["decoder"]["max_patch_size"] = config["training"]["patch_sizes"][-1]
    config["model"]["decoder"]["encoder_embedding_size"] = config["model"]["encoder"][
        "embedding_size"
    ]
    config["model"]["decoder"]["decoder_embedding_size"] = config["model"]["decoder"].pop(
        "embedding_size"
    )
    return config


def prepare_batch(batch, training_config, patch_size):
    """
    Prepare batch for training, including masking.
    """
    b = [t.to(device) for t in batch]
    s_t_x, s_x, t_x, months = b

    image_size = patch_size * training_config["spatial_patches_per_dim"]
    s_t_x, s_x = subset_batch_of_images(s_t_x, s_x, image_size)
    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = batch_mask_presto(
        s_t_x,
        s_x,
        t_x,
        months,
        training_config["mask_ratio"],
        patch_size,
        time_ratio=training_config["time_ratio"],
        space_ratio=training_config["space_ratio"],
    )

    return MaskedOutput(s_t_x, s_x, t_x, s_t_m, s_m, t_m, months)


@torch.no_grad()
def plot_space_time_predictions(
    epoch,
    encoder,
    predictor,
    training_config,
    patch_size,
    masked_output,
    expanded_s_t,
    image_id,
):
    """
    Plots MAE input images, MAE predictions, and difference of input and predictions for a random subset of the dataset.
    Patch sizes, number of images, and number of timesteps are defined in the training config.
    """
    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = masked_output

    expanded_s_t = torch.repeat_interleave(
        s_t_m, repeats=SPACE_TIME_BAND_EXPANSION_T.to(device), dim=-1
    ).bool()

    (p_s_t, _, _) = predictor(
        *encoder(
            s_t_x.float(),
            s_x.float(),
            t_x.float(),
            s_t_m.float(),
            s_m.float(),
            t_m.float(),
            months.long(),
            patch_size=patch_size,
        ),
        patch_size=patch_size,
    )

    # p_s_t and p_s always assume the maximum patch size, so we need to
    # resample if its smaller
    if patch_size < training_config["patch_sizes"][-1]:
        t, d = s_t_x.shape[3], s_t_x.shape[4]
        s_t_x = rearrange(
            resize(
                rearrange(s_t_x, "b h w t d -> b (t d) h w"),
                size=(p_s_t.shape[1], p_s_t.shape[2]),
            ),
            "b (t d) h w -> b h w t d",
            t=t,
            d=d,
        )

        # fix the mask too
        expanded_s_t = expanded_s_t[:, 0::patch_size, 0::patch_size]
        expanded_s_t = repeat(
            expanded_s_t,
            "b h w t c -> b (h h2) (w w2) t c",
            h2=training_config["patch_sizes"][-1],
            w2=training_config["patch_sizes"][-1],
        )

    subplot_titles = []

    for band_list in SPACE_TIME_BANDS_GROUPS_IDX.values():
        for band in band_list:
            subplot_titles.append(SPACE_TIME_BANDS[band])

    plot_list = []

    for t in training_config["timesteps_to_wandb_plot"]:
        # figure columns: input, mask, prediction, error
        # figure rows: bands
        fig, axs = plt.subplots(len(subplot_titles), 3, figsize=(15, 45))

        for i, band in enumerate(subplot_titles):
            x_to_plot = s_t_x[:, :, :, t, i].squeeze(0).cpu()
            pred_to_plot = p_s_t[:, :, :, t, i].squeeze(0).cpu()
            mask_to_plot = expanded_s_t[:, :, :, t, i].squeeze(0).cpu()

            loss = F.mse_loss(
                x_to_plot[mask_to_plot].float(),
                pred_to_plot[mask_to_plot].float(),
            )

            x_plot = axs[i, 0].imshow(
                x_to_plot.numpy(), cmap="gray", vmin=x_to_plot.min(), vmax=x_to_plot.max()
            )
            axs[i, 0].set_title(f"Input {band}, loss: {loss:.4f}")
            fig.colorbar(x_plot, ax=axs[i, 0])
            pred_plot = axs[i, 1].imshow(
                (pred_to_plot * mask_to_plot).numpy(),
                cmap="gray",
                vmin=x_to_plot.min(),
                vmax=x_to_plot.max(),
            )
            axs[i, 1].set_title(f"Output {band}")
            fig.colorbar(pred_plot, ax=axs[i, 2])
            error = axs[i, 2].imshow(
                (abs(x_to_plot.numpy() - pred_to_plot.numpy())) * mask_to_plot.numpy(),
                cmap="coolwarm",
                vmin=0,
                vmax=1,
            )
            axs[i, 2].set_title(f"Input - Output {band}")
            fig.colorbar(error, ax=axs[i, 3])

        fig.suptitle(
            f"Plot image: {image_id}, epoch: {epoch}, patch_size{patch_size}, timestep: {t}",
            fontsize=20,
            y=1.0001,
        )
        fig.tight_layout()

        plot = wandb.Image(
            fig, caption=f"plot_image{image_id}_epoch{epoch}_patch_size{patch_size}_timestep{t}"
        )
        plot_list.append(plot)
    return plot_list


def timestamp_dirname(suffix: Optional[str] = None) -> str:
    ts = datetime.now(dateutil.tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S_%f")
    return f"{ts}_{suffix}" if suffix is not None else ts
