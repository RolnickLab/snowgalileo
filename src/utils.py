import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_SEED
from .masking import MaskedOutput

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
        "num_timesteps_to_wandb_plot": int,
        "band_indeces_to_wandb_plot": list,
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
