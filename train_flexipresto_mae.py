import argparse
import json
import os
from pathlib import Path
from typing import List, cast

import codecarbon
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
from src.eval import EuroSatEval, PastisEval, So2SatEval, TreeSatEval
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.masking import (
    DYNAMIC_BAND_EXPANSION,
    STATIC_BAND_EXPANSION,
    batch_mask_presto,
    subset_batch_of_images,
)
from src.utils import AverageMeter, data_dir, device, load_check_config, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

tracker = codecarbon.EmissionsTracker(
    project_name="flexipresto",
    experiment_name="train_flexipresto.py",
    save_to_api=False,
    output_dir=data_dir,
)

# test:
# https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
torch.backends.cuda.matmul.allow_tf32 = True

tracker.start()

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
dataset = Dataset(DATA_FOLDER / "tifs", download=False, cache_folder=DATA_FOLDER / "npys")
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

DYNAMIC_BAND_EXPANSION_T = torch.tensor(DYNAMIC_BAND_EXPANSION, device=device).long()
STATIC_BAND_EXPANSION_T = torch.tensor(STATIC_BAND_EXPANSION, device=device).long()


if wandb_enabled:
    import wandb

    run = wandb.init(
        entity=wandb_org,
        project="flexipresto",
        dir=output_dir,
    )
    run_id = cast(Run, run).id
    config["training"]["training_samples"] = len(dataset)
    wandb.config.update(config)


param_groups = [{"params": encoder.parameters()}, {"params": predictor.parameters()}]

optimizer = torch.optim.AdamW(param_groups, lr=training_config["start_lr"])  # type: ignore
iterations_per_epoch = len(dataset)

for e in tqdm(range(training_config["num_epochs"])):
    train_loss = AverageMeter()
    for i, b in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        b = [t.to(device) for t in b]
        d_x, s_x, months = b

        # randomly sample a patch size, and a corresponding image size
        patch_size = np.random.choice(training_config["patch_sizes"])
        image_size = patch_size * training_config["spatial_patches_per_dim"]
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

        # also transform to pixel
        reversed_d = torch.repeat_interleave(d_m, repeats=DYNAMIC_BAND_EXPANSION_T, dim=-1).bool()
        reversed_s = torch.repeat_interleave(s_m, repeats=STATIC_BAND_EXPANSION_T, dim=-1).bool()

        optimizer.zero_grad()
        adjust_learning_rate(
            optimizer,
            epoch=i / len(dataloader) + e,
            warmup_epochs=training_config["warmup_epochs"],
            total_epochs=training_config["num_epochs"],
            max_lr=training_config["max_lr"],
            start_lr=training_config["start_lr"],
            min_lr=training_config["final_lr"],
        )

        # generate the predictions. TODO: add layer norm
        (
            p_d,
            p_s,
        ) = predictor(
            *encoder(
                d_x.float(),
                s_x.float(),
                d_m.float(),
                s_m.float(),
                months.long(),
                patch_size=patch_size,
            ),
            patch_size=patch_size,
        )

        # p_d and p_s always assume the maximum patch size, so we need to
        # resample if its smaller
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
            p_s = rearrange(
                resize(rearrange(p_s, "b h w d -> b d h w"), size=(d_x.shape[1], d_x.shape[2])),
                "b d h w -> b h w d",
            )
        loss = F.smooth_l1_loss(
            torch.concat([p_d[reversed_d], p_s[reversed_s]]),
            torch.concat([d_x[reversed_d], s_x[reversed_s]]),
        )
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {e}, iteration {i}: loss = {loss.item()}, memory used: {process.memory_info().rss}",
            flush=True,
        )
        train_loss.update(loss.item(), n=d_x.shape[0])

    if wandb_enabled:
        wandb.log({"train_loss": train_loss.average})

    if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
        e % training_config["eval_eurosat_every_n_epochs"] == 0
    ):
        results = val_task.evaluate_model_on_task(encoder, model_modes=["KNNat5"])
        if wandb_enabled:
            wandb.log(results)


eval_tasks: List[EvalTask] = [
    *[PastisEval()],
    *[TreeSatEval(mode, patch_size) for mode in ["s1", "s2", "combined"] for patch_size in [6, 3]],
    *[EuroSatEval(rgb) for rgb in [True, False]],
    *[So2SatEval()],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder)
    print(json.dumps(results, indent=2), flush=True)
    if wandb_enabled:
        wandb.log(results)
tracker.stop()
