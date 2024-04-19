import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from einops import rearrange
from copy import deepcopy

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

def plot_predictions(
        encoder,
        predictor, 
        training_config, 
        examples_to_plot):
    
    encoder = deepcopy(encoder).requires_grad_(False).eval()
    predictor = deepcopy(predictor).requires_grad_(False).eval()

    for example in examples_to_plot:
        # repeat preprocessing and masking procedure for image to plot
        example = [torch.from_numpy(ex).to(device) for ex in example]
        s_t_x, s_x, t_x, months = example

        # randomly sample a patch size, and a corresponding image size
        patch_size = np.random.choice(training_config["patch_sizes"])
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

        target = s_t_x[:, :, 0, 0].squeeze(0).cpu().numpy()
        print(target.shape)
        assert(target.shape == (image_size, image_size, 1))

        """
        masked = d_x[:, :, :, 0, :].squeeze(0).cpu().numpy()
        assert(masked.shape == (image_size, image_size))

        with torch.no_grad():
            p_d, _ = model(
                s_t_x.float(),
                s_x.float(),
                t_x.float(),
                s_t_m.float(),
                s_m.float(),
                t_m.float(),
                months.long(),
                patch_size=patch_size,
            )

    output = p_d[:, :, :, 0, :].squeeze(0).cpu().numpy()
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
    
    interpolated = p_d[:, :, :, 0, :].squeeze(0).cpu().numpy()
    assert(interpolated.shape == (image_size, image_size))

    # design choices: which bands to plot / RGB (because visual), S1, ERA5, etc.
    # target, masked, prediction, interpolated
    # S2 RGB
    # S1
    # ERA5
    # DW

    # iterate over timesteps to plot, squeeze timedim

    name_plots_list = []
    for i, example in enumerate(training_config["nr_timesteps_to_plot"]):
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
    """