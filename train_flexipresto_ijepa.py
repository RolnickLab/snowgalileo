import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, cast

import codecarbon
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import DATA_FOLDER, EE_PROJECT
from src.eval import EuroSatEval, PastisEval, So2SatEval, TreeSatEval
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoRepresentationDecoder, adjust_learning_rate
from src.masking import batch_mask_presto, subset_batch_of_images
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
argparser.add_argument("--config_file", type=str, default="default.json")
args = argparser.parse_args().__dict__

config = load_check_config(args["config_file"], "jepa")
training_config = config["training"]

# this too
run_id = None
wandb_enabled = True
wandb_org = "nasa-harvest"
output_dir = Path(__file__).parent

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
predictor = PrestoRepresentationDecoder(**config["model"]["decoder"]).to(device)
target_encoder = deepcopy(encoder)
print("Loading validation task")
val_task = EuroSatEval(rgb=True)

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
ema = training_config["ema"]
momentum_scheduler = (
    ema[0] + i * (ema[1] - ema[0]) / (iterations_per_epoch * training_config["num_epochs"])
    for i in range(int(iterations_per_epoch * training_config["num_epochs"]) + 1)
)

for e in tqdm(range(training_config["num_epochs"])):
    train_loss = AverageMeter()
    for i, b in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        b = [t.to(device) for t in b]
        s_t_x, s_x, t_x, months = b

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

        # also transform to patch-space
        patch_s_t = s_t_m[:, 0::patch_size, 0::patch_size].bool()
        patch_s = s_m[:, 0::patch_size, 0::patch_size].bool()

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
        p_s_t, p_s, p_t, _, _, _ = predictor(
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
        # generate the targets
        with torch.no_grad():
            t_s_t, t_s, t_t, _, _, _, _ = target_encoder(
                s_t_x.float(),
                s_x.float(),
                t_x.float(),
                torch.zeros_like(s_t_m),
                torch.zeros_like(s_m),
                torch.zeros_like(t_m),
                months.long(),
                patch_size=patch_size,
            )

        loss = F.smooth_l1_loss(
            torch.concat([p_s_t[patch_s_t], p_s[patch_s], p_t[t_m]]),
            torch.concat([t_s_t[patch_s_t], t_s[patch_s], t_t[t_m]]),
        )
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {e}, iteration {i}: loss = {loss.item()}, memory used: {process.memory_info().rss}",
            flush=True,
        )
        train_loss.update(loss.item(), n=s_t_x.shape[0])
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    if wandb_enabled:
        wandb.log({"train_loss": train_loss.average})

    if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
        e % training_config["eval_eurosat_every_n_epochs"] == 0
    ):
        results = val_task.evaluate_model_on_task(encoder, model_modes=["KNNat5"])
        if wandb_enabled:
            wandb.log(results)


eval_tasks: List[EvalTask] = [
    *[
        PastisEval(
            average_s2_over_month=average_s2_over_month,
            num_subtiles_per_image=num_subtiles_per_image,
        )
        for average_s2_over_month in [True, False]
        for num_subtiles_per_image in [4, 16]
    ],
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
