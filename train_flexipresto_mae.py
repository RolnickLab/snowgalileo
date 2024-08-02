import argparse
import json
import os
from functools import partial
from pathlib import Path
from typing import List, cast

import codecarbon
import matplotlib.pyplot as plt
import psutil
import torch
from torch.utils.data import BatchSampler, DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.collate_fns import MaskingFunctions, mae_collate_fn
from src.conditioner import LearnedMixture
from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import (
    CONFIG_FILENAME,
    DATA_FOLDER,
    DECODER_FILENAME,
    EE_PROJECT,
    ENCODER_FILENAME,
    OUTPUT_FOLDER,
    TIFS_FOLDER,
)
from src.eval import (
    BigEarthNetEval,
    BinaryCropHarvestEval,
    BrickKilnEval,
    CashewPlantEval,
    EuroSatEval,
    PastisPatchEval,
    PastisPixelEval,
    SACropEval,
    So2SatEval,
    TreeSatEval,
)
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.loss import masked_autoencoder_loss
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
argparser.add_argument("--cache_folder", type=str, default="")
args = argparser.parse_args().__dict__

if args["cache_folder"] == "":
    cache_folder = DATA_FOLDER
else:
    cache_folder = Path(args["cache_folder"])

config = load_check_config(args["config_file"], "mae")
training_config = config["training"]

run_id = None
wandb_enabled = True
wandb_org = "nasa-harvest"
output_dir = Path(__file__).parent


print("Loading dataset and dataloader")
dataset = Dataset(TIFS_FOLDER, download=False, h5py_folder=cache_folder / "h5pys", h5pys_only=True)
dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=Hyperparams.num_workers,
    collate_fn=partial(
        mae_collate_fn,
        patch_sizes=training_config["patch_sizes"],
        shape_time_combinations=training_config["shape_time_combinations"],
        mask_ratio=training_config["mask_ratio"],
        decoder_unmask_ratio=training_config["decoder_unmask_ratio"],
        augmentation_strategies=training_config["augmentation"],
        masking_probabilities=training_config["masking_probabilities"],
    ),
    pin_memory=True,
)
print("Loading models")
predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
if "conditioner" in config["model"]:
    eval_w_condition = True
    conditioner = LearnedMixture(**config["model"]["conditioner"]).to(device)
    decoder_conditioner = LearnedMixture(**config["model"]["conditioner"])
    encoder = Encoder(**config["model"]["encoder"], conditioner=conditioner).to(device)
    predictor = PrestoPixelDecoder(
        **config["model"]["decoder"], conditioner=decoder_conditioner
    ).to(device)
    param_groups = [
        {
            "params": [p for n, p in encoder.named_parameters() if "conditioner" not in n],
            "name": "encoder",
            "weight_decay": training_config["weight_decay"],
        },
        {
            "params": [p for n, p in predictor.named_parameters() if "conditioner" not in n],
            "name": "decoder",
            "weight_decay": training_config["weight_decay"],
        },
        {
            "params": [p for p in encoder.conditioner.parameters()]
            + [p for p in predictor.conditioner.parameters()],
            "name": "conditioner",
            "weight_decay": training_config["weight_decay"],
        },
    ]
else:
    eval_w_condition = False
    encoder = Encoder(**config["model"]["encoder"]).to(device)
    predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
    param_groups = [
        {
            "params": encoder.parameters(),
            "name": "encoder",
            "weight_decay": training_config["weight_decay"],
        },
        {
            "params": predictor.parameters(),
            "name": "decoder",
            "weight_decay": training_config["weight_decay"],
        },
    ]


print("Loading validation task")
val_task_no_latlons = EuroSatEval(
    geobench=True, rgb=False, include_latlons=False, do_condition=eval_w_condition
)

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
                    shape_time_combinations=training_config["shape_time_combinations"],
                    fixed_patch_size=p,
                    fixed_space_time_combination={"size": 4, "timesteps": 12},
                    mask_ratio=training_config["mask_ratio"],
                    decoder_unmask_ratio=training_config["decoder_unmask_ratio"],
                ),
            )

            prepared_image_to_plot = {}
            for image_id, bs in enumerate(plot_dataloader):
                b = bs[0]
                b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
                prepared_image_to_plot[image_id] = b[:-1]  # to remove c_i
                if len(prepared_image_to_plot) >= training_config["num_images_to_wandb_plot"]:
                    break

            examples_to_plot[p] = prepared_image_to_plot
            print(f"Prepared {len(prepared_image_to_plot)} images for patch size {p}")
        print(f"all {len(examples_to_plot)} images for patch size")

optimizer = torch.optim.AdamW(
    param_groups, lr=training_config["start_lr"], weight_decay=training_config["weight_decay"]
)  # type: ignore
iterations_per_epoch = len(dataset)
assert training_config["effective_batch_size"] % training_config["batch_size"] == 0
iters_to_accumulate = training_config["effective_batch_size"] / training_config["batch_size"]

i = 0
for e in tqdm(range(training_config["num_epochs"])):
    train_loss = AverageMeter()
    random_masking_train_loss = AverageMeter()
    task_masking_train_loss = AverageMeter()
    for bs in tqdm(dataloader, total=len(dataloader), leave=False):
        for b in bs:
            i += 1
            b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
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
                expanded_sp_x,
                s_t_m_p,
                sp_m_p,
                t_m_p,
                st_m_p,
                patch_size,
                c_i,
            ) = b

            if c_i is not None:
                # there is probably a better way to do this
                c_i = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in c_i.items()
                }
            else:
                raise ValueError("c_i should not be None")

            with torch.autocast(device_type=device.type, dtype=autocast_device):
                (p_s_t, p_sp, p_t, p_st) = predictor(
                    *encoder(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months.long(),
                        c_i=c_i,
                        patch_size=patch_size,
                    ),
                    patch_size=patch_size,
                    c_i=c_i,
                )

                loss = masked_autoencoder_loss(
                    expanded_s_t_x,
                    expanded_sp_x,
                    t_x,
                    st_x,
                    p_s_t,
                    p_sp,
                    p_t,
                    p_st,
                    s_t_m_p,
                    sp_m_p,
                    t_m_p,
                    st_m_p,
                    patch_size=training_config["patch_sizes"][-1],
                    loss_type=training_config["mae_loss"],
                )

            train_loss.update(loss.item(), n=s_t_x.shape[0])
            if c_i is not None:
                task_masking_train_loss.update(loss.item(), n=s_t_x.shape[0])
            else:
                random_masking_train_loss.update(loss.item(), n=s_t_x.shape[0])

            loss = loss / iters_to_accumulate
            loss.backward()

            if ((i + 1) % iters_to_accumulate == 0) or (i + 1 == len(dataloader)):
                if training_config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                adjust_learning_rate(
                    optimizer,
                    epoch=i / (len(MaskingFunctions) * len(dataloader)) + e,
                    warmup_epochs=training_config["warmup_epochs"],
                    total_epochs=training_config["num_epochs"],
                    max_lr=training_config["max_lr"],
                    start_lr=training_config["start_lr"],
                    min_lr=training_config["final_lr"],
                    conditioner_multiplier=training_config["conditioner_multiplier"],
                )

    if wandb_enabled:
        to_log = {
            "train_loss": train_loss.average,
            "random_masking_train_loss": random_masking_train_loss.average,
            "task_masking_train_loss": task_masking_train_loss.average,
            "epoch": e,
        }
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
                to_log[f"plot_mae_patch_size_{patch_size}"] = plot

        if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
            e % training_config["eval_eurosat_every_n_epochs"] == 0
        ):
            results = val_task_no_latlons.evaluate_model_on_task(
                encoder, model_modes=["KNNat5 Classifier", "KNNat20 Classifier"]
            )
            to_log.update(results)
        wandb.log(to_log)
        plt.close("all")

model_path = OUTPUT_FOLDER / timestamp_dirname(run_id)
model_path.mkdir()
torch.save(encoder.state_dict(), model_path / ENCODER_FILENAME)
torch.save(predictor.state_dict(), model_path / DECODER_FILENAME)
with (model_path / CONFIG_FILENAME).open("w") as f:
    json.dump(config, f)

eval_tasks: List[EvalTask] = [
    *[BinaryCropHarvestEval(country=country) for country in ["Kenya", "Togo", "Brazil"]],
    *[EuroSatEval(rgb=rgb, include_latlons=False, geobench=True) for rgb in [True, False]],
    *[So2SatEval(geobench=geobench) for geobench in [True, False]],
    BrickKilnEval(),
    *[CashewPlantEval(output_mode=output_mode) for output_mode in ["mode", "norm_counts"]],
    *[SACropEval(output_mode=output_mode) for output_mode in ["mode", "norm_counts"]],
    *[
        EuroSatEval(rgb=rgb, include_latlons=include_latlons, geobench=False)
        for rgb in [True, False]
        for include_latlons in [True, False]
    ],
    *[
        PastisPatchEval(
            output_mode=output_mode,
            num_subtiles_per_image=num_subtiles_per_image,
            band_mode=band_mode,
        )
        for output_mode in ["mode", "norm_counts"]
        # 4 has input hw 64, 16 has input hw 32
        for num_subtiles_per_image in [4, 16]
        for band_mode in ["combined", "s2"]
    ],
    *[
        TreeSatEval(mode=mode, patch_size=patch_size)
        for mode in ["s1", "s2", "combined"]
        for patch_size in [6, 3]
    ],
    PastisPixelEval(),
    BigEarthNetEval(),
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder)
    print(json.dumps(results, indent=2), flush=True)
    if wandb_enabled:
        wandb.log(results)
tracker.stop()
