import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import List, cast

import codecarbon
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from tqdm import tqdm

from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import DATA_FOLDER, EE_PROJECT, NUM_TIMESTEPS
from src.data.dataset import SPACE_TIME_BANDS
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
from src.utils import (
    AverageMeter,
    data_dir,
    device,
    load_check_config,
    seed_everything,
)
from wandb.sdk.wandb_run import Run

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


##################################################################################################
def plot_space_time_predictions(encoder, predictor, training_config, plot_dataloader):
    """
    Plots MAE input images, MAE predictions and differences for a random subset of the dataset.
    The number of timesteps, number of images, and bands to plot are defined in the training config.
    """
    c = 0

    encoder = deepcopy(encoder).requires_grad_(False).eval()
    predictor = deepcopy(predictor).requires_grad_(False).eval()

    examples_to_plot = []

    for _ in range(training_config["num_images_to_wandb_plot"]):
        # extract batches of images to be able to apply batch masking
        examples_to_plot.append(next(iter(plot_dataloader)))

    for idx, example in enumerate(examples_to_plot):
        # repeat preprocessing and masking procedure for image to plot
        example = [ex.to(device) for ex in example]
        s_t_x, s_x, t_x, months = example

        # patch size will be the last one because random is seeded
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

        input_to_plot = s_t_x[:, :, :, :, :].squeeze(0).cpu().numpy()
        assert input_to_plot.shape == (
            image_size,
            image_size,
            NUM_TIMESTEPS,
            len(SPACE_TIME_BANDS),
        )

        with torch.no_grad():
            output, _, _ = predictor(
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

        output_to_plot = output[:, :, :, :, :].squeeze(0).cpu().numpy()
        assert output_to_plot.shape == (
            training_config["patch_sizes"][-1] * training_config["spatial_patches_per_dim"],
            training_config["patch_sizes"][-1] * training_config["spatial_patches_per_dim"],
            NUM_TIMESTEPS,
            len(SPACE_TIME_BANDS),
        )

        if patch_size < training_config["patch_sizes"][-1]:
            t, d = s_t_x.shape[3], s_t_x.shape[4]
            interpolated = rearrange(
                resize(
                    rearrange(output, "b h w t d -> b (t d) h w"),
                    size=(s_t_x.shape[1], s_t_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=t,
                d=d,
            )
            interpolated_to_plot = interpolated[:, :, :, :, :].squeeze(0).cpu().numpy()
            assert interpolated_to_plot.shape == (
                image_size,
                image_size,
                NUM_TIMESTEPS,
                len(SPACE_TIME_BANDS),
            )

    plot_list = []
    for c in training_config["band_indeces_to_wandb_plot"]:
        for t in range(training_config["num_timesteps_to_wandb_plot"]):
            input = input_to_plot[:, :, t, c]
            output = output_to_plot[:, :, t, c]
            if patch_size < training_config["patch_sizes"][-1]:
                interpolated = interpolated_to_plot[:, :, t, c]

            # plot target, masked, prediction, interpolated
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].imshow(input, cmap="gray")
            axs[0, 0].set_title(f"Target_image{idx}_timestep{t}_channel{c}")
            axs[0, 1].imshow(output, cmap="gray")
            axs[0, 1].set_title(f"Prediction_image{idx}_timestep{t}_channel{c}")
            if patch_size < training_config["patch_sizes"][-1]:
                axs[1, 0].imshow(interpolated, cmap="gray")
                axs[1, 0].set_title(f"Interpolated_image{idx}_timestep{t}_band{c}")
                axs[1, 1].imshow(input - interpolated, cmap="coolwarm")
                axs[1, 1].set_title(f"Difference_image{idx}_timestep{t}_channel{c}")
            else:
                axs[1, 0].imshow(input - output, cmap="coolwarm")
                axs[1, 0].set_title(f"Difference_image{idx}_timestep{t}_channel{c}")

            fig.tight_layout()

            title = f"plot_{idx}_{t}_{c}"
            plot = wandb.Image(fig)
            plot_list.append((title, plot))
    return plot_list


##################################################################################################

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

    # choose random images to plot during training
    if training_config["wandb_plot_every_n_epochs"] > 0:
        plot_dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )

param_groups = [{"params": encoder.parameters()}, {"params": predictor.parameters()}]

optimizer = torch.optim.AdamW(param_groups, lr=training_config["start_lr"])  # type: ignore
iterations_per_epoch = len(dataset)

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

        # also transform to pixel
        expanded_s_t = torch.repeat_interleave(
            s_t_m, repeats=SPACE_TIME_BAND_EXPANSION_T, dim=-1
        ).bool()
        expanded_s = torch.repeat_interleave(s_m, repeats=SPACE_BAND_EXPANSION_T, dim=-1).bool()
        expanded_t = torch.repeat_interleave(t_m, repeats=TIME_BAND_EXPANSION_T, dim=-1).bool()

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
        (p_s_t, p_s, p_t) = predictor(
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
            p_s_t = rearrange(
                resize(
                    rearrange(p_s_t, "b h w t d -> b (t d) h w"),
                    size=(s_t_x.shape[1], s_t_x.shape[2]),
                ),
                "b (t d) h w -> b h w t d",
                t=t,
                d=d,
            )
            p_s = rearrange(
                resize(
                    rearrange(p_s, "b h w d -> b d h w"), size=(s_t_x.shape[1], s_t_x.shape[2])
                ),
                "b d h w -> b h w d",
            )
        loss = F.mse_loss(
            torch.concat([p_s_t[expanded_s_t], p_s[expanded_s], p_t[expanded_t]]),
            torch.concat([s_t_x[expanded_s_t], s_x[expanded_s], t_x[expanded_t]]).float(),
        )
        loss.backward()
        optimizer.step()
        print(
            f"Epoch {e}, iteration {i}: loss = {loss.item()}, memory used: {process.memory_info().rss}",
            flush=True,
        )
        train_loss.update(loss.item(), n=s_t_x.shape[0])

    if wandb_enabled:
        wandb.log({"train_loss": train_loss.average})

        if (training_config["wandb_plot_every_n_epochs"] != 0) and (
            e % training_config["wandb_plot_every_n_epochs"] == 0
        ):
            plot_list = plot_space_time_predictions(
                encoder=encoder,
                predictor=predictor,
                training_config=training_config,
                plot_dataloader=plot_dataloader,
            )
            for title, plot in plot_list:
                wandb.log({"plot": plot})

    if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
        e % training_config["eval_eurosat_every_n_epochs"] == 0
    ):
        results = val_task.evaluate_model_on_task(encoder, model_modes=["KNNat5"])
        if wandb_enabled:
            wandb.log(results)


eval_tasks: List[EvalTask] = [
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


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
