import argparse
import json
import os
from pathlib import Path
from typing import List, cast

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import DATA_FOLDER, EE_PROJECT
from src.eval import EuroSatEval, So2SatEval, TreeSatEval
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.masking import (
    SPACE_BAND_EXPANSION,
    SPACE_TIME_BAND_EXPANSION,
    TIME_BAND_EXPANSION,
    batch_mask_presto,
    subset_batch_of_images,
)
from src.utils import AverageMeter, data_dir, device, load_check_config, seed_everything, plot_predictions

seed_everything(DEFAULT_SEED)
process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# test:
# https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="small.json")
args = argparser.parse_args().__dict__

config = load_check_config(args["config_file"], "mae")
training_config = config["training"]

run_id = None
wandb_enabled = True
wandb_org = "nasa-harvest"
output_dir = Path(__file__).parent

print("Loading dataset and dataloader")
dataset = Dataset(
    DATA_FOLDER / "tifs", download=False, cache_folder=DATA_FOLDER / "npys_spacetime"
)
dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=Hyperparams.num_workers,
)
print("Loading models")
encoder = Encoder(**config["model"]["encoder"]).to(device)
predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
print("Loading validation task")
val_task = EuroSatEval(rgb=True)

SPACE_TIME_BAND_EXPANSION_T = torch.tensor(SPACE_TIME_BAND_EXPANSION, device=device).long()
SPACE_BAND_EXPANSION_T = torch.tensor(SPACE_BAND_EXPANSION, device=device).long()
TIME_BAND_EXPANSION_T = torch.tensor(TIME_BAND_EXPANSION, device=device).long()


###############################################################
if training_config["wandb_plot_every_n_epochs"] > 0:
    examples_to_plot = []
    plot_indeces = np.random.choice(len(dataset), training_config["num_images_to_wandb_plot"])

    for i in plot_indeces:
        examples_to_plot.append(dataset[i])

plot_predictions(encoder=encoder, predictor=predictor, training_config=training_config, examples_to_plot = examples_to_plot)
