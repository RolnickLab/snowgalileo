import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from einops import rearrange

from .config import DEFAULT_SEED
from .masking import MaskedOutput, subset_batch_of_images, batch_mask_presto
from torchvision.transforms.functional import resize

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


def masked_output_np_to_tensor(d_x, s_x, d_m, s_m, month) -> MaskedOutput:
    """converts eval task"""
    d_x_torch = torch.as_tensor(d_x, dtype=torch.float32)
    s_x_torch = torch.as_tensor(s_x, dtype=torch.float32)
    d_m_torch = torch.as_tensor(d_m, dtype=torch.float32)
    s_m_torch = torch.as_tensor(s_m, dtype=torch.float32)
    month_torch = torch.as_tensor(month, dtype=torch.long)
    return MaskedOutput(d_x_torch, s_x_torch, d_m_torch, s_m_torch, month_torch)


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
        "plot_every_n_epochs": int,
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


def plot_masked_bands(y_true: np.ndarray, y_pred: np.ndarray, mask_strategy: str):
    """Plot only the masked bands over time"""
    ncols = len(BANDS_GROUPS_IDX[mask_strategy])
    plt = import_optional_dependency("matplotlib.pyplot")
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 10))
    for i, masked_band_idx in enumerate(BANDS_GROUPS_IDX[mask_strategy]):
        if ncols == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(y_true[:, masked_band_idx], label=f"Actual {mask_strategy} band {i}")
        ax.plot(y_pred[:, masked_band_idx], label=f"Prediced {mask_strategy} band {i}")
        ax.set_title(f"{mask_strategy} band {i}")
        ax.set_ylabel(f"{mask_strategy} band {i}")
        ax.set_xlabel("Time interval")
        ax.legend()
    return fig


def plot_masked_general(example: MaskedExample, y_pred: np.ndarray, dw_pred: np.ndarray):
    """Plot all bands over time"""
    fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20, 30))

    # Reconstruct eo data
    eo_data_actual = example.x_eo.copy()
    eo_data_actual[example.mask_eo == 1] = example.y_eo[example.mask_eo == 1]
    eo_data_predicted = y_pred

    dw_actual = example.x_dw.copy()
    dw_actual[example.mask_dw == 1] = example.y_dw[example.mask_dw == 1]
    dw_predicted = np.argmax(dw_pred, axis=1)

    row_idx = 0
    for band_group, band_indexes in BANDS_GROUPS_IDX.items():
        if row_idx > 6:
            row_idx = 6
        else:
            col_idx = 0
        for b in band_indexes:
            ax = axes[row_idx, col_idx]
            (pred_line,) = ax.plot(eo_data_predicted[:, b], color="orange")
            (actual_line,) = ax.plot(eo_data_actual[:, b], color="blue")
            ax.set_title(NORMED_BANDS[b])
            ax.set_ylabel(band_group)
            col_idx += 1
        row_idx += 1

    dw_ax = axes[0, 4]
    dw_ax.plot(dw_predicted, color="orange")
    dw_ax.plot(dw_actual, color="blue")
    dw_ax.set_title("Dynamic World")
    dw_ax.set_yticks(list(DynamicWorld2020_2021.legend.keys()))
    dw_ax.set_yticklabels((DynamicWorld2020_2021.legend.values()), rotation=60)

    fig.legend([pred_line, actual_line], ["Predicted", "Actual"], loc="upper left")
    return fig


def plot_prediction(example: MaskedExample, eo_pred: np.ndarray, dw_pred: np.ndarray):
    if example.strategy in list(BANDS_GROUPS_IDX.keys()):
        fig = plot_masked_bands(example.y_eo, eo_pred, example.strategy)
    else:
        fig = plot_masked_general(example, eo_pred, dw_pred)
    plt = import_optional_dependency("matplotlib.pyplot")
    plt.suptitle(
        f"Start month: {example.start_month}, "
        + f"Latlon: {example.latlon}"
        + f"\nStrategy: {example.strategy}",
        size=24,
    )
    fig.subplots_adjust(top=0.15)
    fig.tight_layout()
    return fig

def plot_predictions(model, patch_size, image_size, training_config, image):
    # repeat preprocessing and masking procedure
    d_x, s_x, d_m, s_m, months = image
    d_x, s_x, months = image

    d_x, s_x = subset_batch_of_images(d_x, s_x, image_size)
    d_x, s_x, d_m, s_m, months = batch_mask_presto(
        d_x,
        s_x,
        months,
        training_config["mask_ratio"],
        patch_size,
        time_ratio=training_config["time_ratio"],
        space_ratio=training_config["space_ratio"],
    )

    target = d_x[:, :, :, 0, 0].squeeze(0).cpu().numpy()
    assert(target.shape == (image_size, image_size))

    with torch.no_grad():
        p_d, _ = model(
            d_x.float(),
            s_x.float(),
            d_m.float(),
            s_m.float(),
            months.long(),
            patch_size=patch_size,
        )

    output = p_d[:, :, :, 0, 0].squeeze(0).cpu().numpy()
    assert(output.shape == (image_size, image_size))

    if patch_size < training_config["patch_sizes"][-1]:
        t, d = d_x.shape[3], d_x.shape[4]
        p_d = rearrange(
            resize(
                rearrange(p_d, "b h w t d -> b (t d) h w"), size=(d_x.shape[1], d_x.shape[2])
            ),
            "b (t d) h w -> b h w t d",
            t=t,
            d=d,
        )
    
    interpolated = p_d[:, :, :, 0, 0].squeeze(0).cpu().numpy()
    assert(interpolated.shape == (image_size, image_size))

    name_plots_list = []
    for i, example in enumerate(examples):
        if i < wandb_plots:
            title = f"plot_train_{i}_{example.strategy}"
        else:
            title = f"plot_val_{i}_{example.strategy}"
        fig = plot_masked(
            example=example,
            output=p_d[i].cpu().numpy(),
            interpolated=dw_preds[i].cpu().numpy(),
        )
        name_plots_list.append((title, wandb.Image(fig)))
    return name_plots_list