import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import List, cast

import codecarbon
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.collate_fns import mae_collate_fn
from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import DATA_FOLDER, EE_PROJECT, OUTPUT_FOLDER
from src.eval import EuroSatEval, So2SatEval, TreeSatEval
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.utils import (
    AverageMeter,
    data_dir,
    device,
    is_bf16_available,
    load_check_config,
    plot_space_time_predictions,
    seed_everything,
    timestamp_dirname,
)

seed_everything(DEFAULT_SEED)
process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

tracker = codecarbon.EmissionsTracker(
    project_name="flexipresto",
    experiment_name="train_flexipresto.py",
    save_to_api=False,
    output_dir=data_dir,
)

torch.backends.cuda.matmul.allow_tf32 = True
autocast_device = torch.bfloat16 if is_bf16_available() else torch.float32

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
dataset = Dataset(
    DATA_FOLDER / "tifs", download=False, cache_folder=DATA_FOLDER / "npys_spacetime_16"
)
dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=Hyperparams.num_workers,
    collate_fn=partial(
        mae_collate_fn,
        patch_sizes=training_config["patch_sizes"],
        spatial_patches_per_dim=training_config["spatial_patches_per_dim"],
        mask_ratio=training_config["mask_ratio"],
        time_ratio=training_config["time_ratio"],
        space_ratio=training_config["space_ratio"],
        channel_ratio=training_config["channel_ratio"],
    ),
    pin_memory=True,
)
print("Loading models")
encoder = Encoder(**config["model"]["encoder"]).to(device)
predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
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

    # prepare random images to plot during training
    if training_config["wandb_plot_every_n_epochs"] > 0:
        assert training_config["num_images_to_wandb_plot"] > 0
        assert len(training_config["patch_sizes_to_wandb_plot"]) > 0
        assert len(training_config["timesteps_to_wandb_plot"]) > 0

        examples_to_plot = {}

        for p in training_config["patch_sizes_to_wandb_plot"]:
            # call the collate function with current patch size
            plot_dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_sampler=BatchSampler([1, 2, 3], batch_size=1, drop_last=False),
                collate_fn=partial(
                    mae_collate_fn,
                    patch_sizes=training_config["patch_sizes"],
                    spatial_patches_per_dim=training_config["spatial_patches_per_dim"],
                    mask_ratio=training_config["mask_ratio"],
                    time_ratio=training_config["time_ratio"],
                    space_ratio=training_config["space_ratio"],
                    channel_ratio=training_config["channel_ratio"],
                    fixed_patch_size=p,
                ),
            )

            prepared_image_to_plot = {}
            for image_id, b in enumerate(plot_dataloader):
                b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
                prepared_image_to_plot[image_id] = b
                if len(prepared_image_to_plot) >= training_config["num_images_to_wandb_plot"]:
                    break

            examples_to_plot[p] = prepared_image_to_plot
            print(f"Prepared {len(prepared_image_to_plot)} images for patch size {p}")
        print(f"all {len(examples_to_plot)} images for patch size")

param_groups = [{"params": encoder.parameters()}, {"params": predictor.parameters()}]

optimizer = torch.optim.AdamW(param_groups, lr=training_config["start_lr"])  # type: ignore
iterations_per_epoch = len(dataset)

for e in tqdm(range(training_config["num_epochs"])):
    train_loss = AverageMeter()
    for i, b in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
        (
            s_t_x,
            s_x,
            t_x,
            s_t_m,
            s_m,
            t_m,
            months,
            expanded_s_t_x,
            expanded_s_x,
            s_t_m_p,
            s_m_p,
            t_m_p,
            patch_size,
        ) = b

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

        with torch.autocast(device_type=device.type, dtype=autocast_device):
            (p_s_t, p_s, p_t) = predictor(
                *encoder(
                    s_t_x,
                    s_x,
                    t_x,
                    s_t_m,
                    s_m,
                    t_m,
                    months.long(),
                    patch_size=patch_size,
                ),
                patch_size=patch_size,
            )

            loss = F.mse_loss(
                torch.concat([p_s_t[s_t_m_p], p_s[s_m_p], p_t[t_m_p]]),
                torch.concat([expanded_s_t_x[s_t_m_p], expanded_s_x[s_m_p], t_x[t_m_p]]),
            )
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), n=s_t_x.shape[0])

    if wandb_enabled:
        wandb.log({"train_loss": train_loss.average})

        if (training_config["wandb_plot_every_n_epochs"] != 0) and (
            e % training_config["wandb_plot_every_n_epochs"] == 0
        ):
            plot_list = []
            for patch_size, patch_size_dict in examples_to_plot.items():
                for image_id, prepared_image in patch_size_dict.items():
                    plot_list.append(
                        plot_space_time_predictions(
                            epoch=e,
                            encoder=encoder,
                            predictor=predictor,
                            training_config=training_config,
                            prepared_image=prepared_image,
                            image_id=image_id,
                        )
                    )
            for patch_size, plot in [
                (patch_size, plot)
                for plot_dict in plot_list
                for patch_size, plot in plot_dict.items()
            ]:
                wandb.log({f"plot_mae_patch_size_{patch_size}": plot})

    if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
        e % training_config["eval_eurosat_every_n_epochs"] == 0
    ):
        results = val_task.evaluate_model_on_task(encoder, model_modes=["KNNat5"])
        if wandb_enabled:
            wandb.log(results)

model_path = OUTPUT_FOLDER / timestamp_dirname(run_id)
model_path.mkdir()
torch.save(encoder.state_dict(), model_path / "encoder.pt")
torch.save(predictor.state_dict(), model_path / "predictor.pt")

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
