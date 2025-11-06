import argparse
import copy
import json
import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import List, cast

import codecarbon
import psutil
import torch
import torch.nn as nn
import wandb
from torch.utils.data import BatchSampler, DataLoader, Subset
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.collate_fns import mae_collate_fn
from src.config import DEFAULT_SEED, get_random_config
from src.data import Dataset, Normalizer
from src.data.config import (
    CONFIG_FILENAME,
    DATA_FOLDER,
    DECODER_FILENAME,
    EE_BUCKET_TIFS,
    EE_PROJECT,
    ENCODER_FILENAME,
    NORMALIZATION_DICT_FILENAME,
    OPTIMIZER_FILENAME,
    OUTPUT_FOLDER,
    TARGET_ENCODER_FILENAME,
)
from src.eval.eval import EvalTask
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.loss import construct_target_encoder_masks, do_loss
from src.utils import (
    AverageMeter,
    check_config,
    config_dir,
    data_dir,
    device,
    is_bf16_available,
    load_check_config,
    seed_everything,
    timestamp_dirname,
    will_cause_nans,
)

process = psutil.Process()

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

tracker = codecarbon.EmissionsTracker(
    project_name="ai4snow",
    experiment_name="ai4snow.py",
    save_to_api=False,
    output_dir=data_dir,
)

torch.backends.cuda.matmul.allow_tf32 = True
autocast_device = torch.bfloat16 if is_bf16_available() else torch.float32

# tracker.start()

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="ai4snow_ps10.json")
argparser.add_argument("--run_name_prefix", type=str, default="")
argparser.add_argument("--h5py_folder", type=str, default="data/h5pys_ps10_5")
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--download", dest="download", action="store_true")
argparser.add_argument("--h5pys_only", dest="h5pys_only", action="store_true")
argparser.add_argument("--num_workers", dest="num_workers", default=0)
argparser.add_argument("--batch_size", dest="batch_size", default="")
argparser.add_argument("--sync_models_from_service_account", action="store_true")
argparser.add_argument("--checkpoint_every_epoch", type=int, default=50)
argparser.add_argument("--tifs_folder", type=str, default="tifs_all_bands_500m")
argparser.add_argument(
    "--dataset_subset_size", type=int, default=0, help="0 for using the entire dataset"
)
argparser.add_argument(
    "--wandb_run_id", type=str, default="", help="If set, will resume the run with this ID"
)
argparser.add_argument("--restart", action="store_true", help="If set, will restart the run")
argparser.add_argument("--path_to_model_checkpoint", type=str, default="")
argparser.add_argument(
    "--ablate",
    type=str,
    choices=["", "time", "space", "high_res", "low_res", "aux", "location", "time", "space"],
    default="",
)

argparser.set_defaults(download=False)
argparser.set_defaults(cache_in_ram=False)
args = argparser.parse_args().__dict__

if args["restart"]:
    assert args["path_to_model_checkpoint"] != "", "Please provide a path to the model checkpoint"
    model_path = Path(args["path_to_model_checkpoint"])
    assert model_path.exists(), f"Model path {model_path} does not exist"
else:
    model_path = None

if args["h5py_folder"] == "":
    cache_folder = None
else:
    cache_folder = Path(args["h5py_folder"])


if args["output_folder"] == "":
    output_folder = OUTPUT_FOLDER
else:
    output_folder = Path(args["output_folder"])

if args["wandb_run_id"] != "":
    run_id = args["wandb_run_id"]
else:
    # if not set, we will create a new run
    run_id = None

start_epoch = 0
wandb_enabled = True
wandb_org = "sea-ice"
wandb_output_dir = Path(__file__).parent

if not args["restart"]:
    if args["config_file"] == "random_tiny":
        config, run_name = get_random_config("tiny")
        config = check_config(config)
    elif args["config_file"] == "random_vitb-tiny":
        config, run_name = get_random_config("vitb-tiny")
        config = check_config(config)
    elif args["config_file"] == "random_base":
        config, run_name = get_random_config("base")
        config = check_config(config)
    else:
        config = load_check_config(args["config_file"])
        run_name = f"{args['config_file']} config file"
    if args["run_name_prefix"] != "":
        prefix = args["run_name_prefix"]
        run_name = f"{prefix}_{run_name}"
    config["run_name"] = run_name
    id_dir = None
else:
    # if we are restarting, we load the config from the model path
    assert model_path is not None, "Please provide a path to the model checkpoint"
    # Find the subdirectory ending with run id
    matching_dirs = list((model_path).glob(f"*{run_id}"))

    if not matching_dirs:
        raise FileNotFoundError("No subdirectory ending with run id found in model_path")

    id_dir = matching_dirs[0]

    with (id_dir / CONFIG_FILENAME).open("r") as f:
        config = json.load(f)
    run_name = config["run_name"]
    start_epoch = config.get("cur_epoch", 0)
    print("Restarting from epoch:", start_epoch)

run = wandb.init(
    name=run_name,
    entity=wandb_org,
    project="ai4snow",
    dir=wandb_output_dir,
    id=run_id,
    resume="allow",
)
run_id = cast(Run, run).id
config["wandb_run_id"] = run_id

training_config = config["training"]

if args["batch_size"] != "":
    warnings.warn(
        f"Overriding batch size from {training_config['batch_size']} to {args['batch_size']}"
    )
    training_config["batch_size"] = int(args["batch_size"])
    config["training"]["batch_size"] = int(args["batch_size"])

# we seed everything after we call get_random_config(), since
# we want this to differ between runs
seed_everything(DEFAULT_SEED)

print("Loading dataset and dataloader")

dataset = Dataset(
    data_folder=DATA_FOLDER / args["tifs_folder"],
    download=args["download"],
    h5py_folder=cache_folder,
    h5pys_only=args["h5pys_only"],
)
config["training"]["training_samples"] = len(dataset)

if not args["restart"]:
    # we can't reset these values without wandb
    # complaining
    wandb.config.update(config)

if training_config["normalization"] == "std":
    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(normalizing_dict, flush=True)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer
else:
    normalizer = Normalizer(std=False)
    dataset.normalizer = normalizer

# use a subset of the dataset for training
if args["dataset_subset_size"] > 0:
    subset_size = args["dataset_subset_size"]
    indices = random.sample(range(len(dataset)), subset_size)
    dataset = Subset(dataset, indices)  # type: ignore

dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=int(args["num_workers"]),
    collate_fn=partial(
        mae_collate_fn,
        patch_sizes_high_res=training_config["patch_sizes_high_res"],
        patch_sizes_med_res=training_config["patch_sizes_med_res"],
        patch_sizes_low_res=training_config["patch_sizes_low_res"],
        shape_time_combinations=training_config["shape_time_combinations"],
        encode_ratio=training_config["encode_ratio"],
        decode_ratio=training_config["decode_ratio"],
        augmentation_strategies=training_config["augmentation"],
        masking_probabilities=training_config["masking_probabilities"],
        max_unmasking_channels=training_config["max_unmasking_channels"],
        random_masking=training_config["random_masking"],
        unmasking_channels_combo=training_config["unmasking_channels_combo"],
        ablate=args["ablate"],
    ),
    pin_memory=True,
)

# prepare random images to plot during training
if training_config["wandb_plot_every_n_epochs"] > 0:
    assert training_config["num_images_to_wandb_plot"] > 0
    assert len(training_config["timesteps_to_wandb_plot"]) > 0

    plot_dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_sampler=BatchSampler([1, 2, 3], batch_size=1, drop_last=False),
        collate_fn=partial(
            mae_collate_fn,
            patch_sizes_high_res=training_config["patch_sizes_high_res"],
            patch_sizes_med_res=training_config["patch_sizes_med_res"],
            patch_sizes_low_res=training_config["patch_sizes_low_res"],
            shape_time_combinations=training_config["shape_time_combinations"],
            encode_ratio=training_config["encode_ratio"],
            decode_ratio=training_config["decode_ratio"],
            augmentation_strategies=training_config["augmentation"],
            masking_probabilities=training_config["masking_probabilities"],
            max_unmasking_channels=training_config["max_unmasking_channels"],
            random_masking=training_config["random_masking"],
            unmasking_channels_combo=training_config["unmasking_channels_combo"],
            ablate=args["ablate"],
        ),
    )
    prepared_image_to_plot = {}
    for image_id, b in enumerate(plot_dataloader):
        b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
        prepared_image_to_plot[image_id] = b
        if len(prepared_image_to_plot) >= training_config["num_images_to_wandb_plot"]:
            break
    print(f"Prepared {len(prepared_image_to_plot)} images")


print("Loading models")
predictor = PrestoPixelDecoder(**config["model"]["decoder"])
if torch.cuda.device_count() > 1:
    print("Transforming predictor to use multiple GPUs")
    predictor = nn.DataParallel(predictor)
predictor.to(device)
param_groups = [
    {
        "params": predictor.parameters(),
        "name": "decoder",
        "weight_decay": training_config["weight_decay"],
    }
]
encoder = Encoder(**config["model"]["encoder"])
if torch.cuda.device_count() > 1:
    print("Transforming encoder to use multiple GPUs")
    encoder = nn.DataParallel(encoder)
encoder.to(device)
param_groups.append(
    {
        "params": encoder.parameters(),
        "name": "encoder",
        "weight_decay": training_config["weight_decay"],
    }
)

if args["restart"]:
    assert model_path is not None
    print(f"Loading checkpoint for epoch {start_epoch} from {model_path}", flush=True)
    encoder.load_state_dict(torch.load(id_dir / ENCODER_FILENAME, map_location=device))
    predictor.load_state_dict(torch.load(id_dir / DECODER_FILENAME, map_location=device))

# print("Loading validation task")
# val_task_no_latlons = EuroSatEval(
#    normalization=dataset.normalizer,
#    geobench=True,
#    rgb=True,
#    include_latlons=False,
# )

optimizer = torch.optim.AdamW(
    param_groups,
    lr=0,
    weight_decay=training_config["weight_decay"],
    betas=(training_config["betas"][0], training_config["betas"][1]),
)  # type: ignore
if args["restart"]:
    assert model_path is not None
    print(f"Loading optimizer state from {model_path}", flush=True)
    optimizer.load_state_dict(torch.load(id_dir / OPTIMIZER_FILENAME, map_location=device))

assert training_config["effective_batch_size"] % training_config["batch_size"] == 0
iters_to_accumulate = training_config["effective_batch_size"] / training_config["batch_size"]

# setup target encoder and momentum from: https://github.com/facebookresearch/ijepa/blob/main/src/train.py
repeat_aug = 4
steps_per_epoch = len(dataloader) * repeat_aug / iters_to_accumulate
momentum_scheduler = (
    training_config["ema"][0]
    + i
    * (training_config["ema"][1] - training_config["ema"][0])
    / (steps_per_epoch * training_config["num_epochs"])
    for i in range(int(steps_per_epoch * training_config["num_epochs"]) + 1)
)
target_encoder = copy.deepcopy(encoder)
target_encoder.eval()
if args["restart"]:
    assert model_path is not None
    target_encoder.load_state_dict(
        torch.load(id_dir / TARGET_ENCODER_FILENAME, map_location=device)
    )
    # we also want to step through the momentum scheduler since we are going to fast forward training
    for momentum_epoch in range(start_epoch):
        for i in range(int(steps_per_epoch)):
            _ = next(momentum_scheduler)

for p in target_encoder.parameters():
    p.requires_grad = False

skipped_batches = 0
for e in tqdm(range(start_epoch, training_config["num_epochs"])):
    print(f"Epoch {e + 1}")
    i = 0
    train_loss = AverageMeter()
    random_masking_train_loss = AverageMeter()
    task_masking_train_loss = AverageMeter()
    for bs in tqdm(dataloader, total=len(dataloader), leave=False):
        for b in bs:
            i += 1
            b = [t.to(device) if isinstance(t, torch.Tensor) else t for t in b]
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size_high_res,
                patch_size_med_res,
                patch_size_low_res,
            ) = b

            print("t shape: " + str(t_x.shape))

            if (
                will_cause_nans(s_t_h_x)
                or will_cause_nans(s_t_m_x)
                or will_cause_nans(s_t_l_x)
                or will_cause_nans(sp_x)
                or will_cause_nans(t_x)
                or will_cause_nans(st_x)
            ):
                skipped_batches += 1
                warnings.warn(f"Skipping batch with NaNs, {skipped_batches}")
                continue

            with torch.autocast(device_type=device.type, dtype=autocast_device):
                (p_s_t_h, p_s_t_m, p_s_t_l, p_sp, p_t, p_st) = predictor(
                    *encoder(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_h_m,
                        s_t_m_m,
                        s_t_l_m,
                        sp_m,
                        t_m,
                        st_m,
                        months.long(),
                        patch_size_high_res=patch_size_high_res,
                        patch_size_med_res=patch_size_med_res,
                        patch_size_low_res=patch_size_low_res,
                    ),
                    patch_size_high_res=patch_size_high_res,
                    patch_size_med_res=patch_size_med_res,
                    patch_size_low_res=patch_size_low_res,
                )

                # handle nans introduced after processing
                if (
                    will_cause_nans(p_s_t_h)
                    or will_cause_nans(p_s_t_m)
                    or will_cause_nans(p_s_t_l)
                    or will_cause_nans(p_sp)
                    or will_cause_nans(p_t)
                    or will_cause_nans(p_st)
                ):
                    skipped_batches += 1
                    warnings.warn(f"Skipping batch with NaNs after processing, {skipped_batches}")
                    continue

                if training_config["loss_type"] != "MAE":
                    with torch.no_grad():
                        t_s_t_h, t_s_t_m, t_s_t_l, t_sp, t_t, t_st, _, _, _, _, _ = target_encoder(
                            s_t_h_x,
                            s_t_m_x,
                            s_t_l_x,
                            sp_x,
                            t_x,
                            st_x,
                            *construct_target_encoder_masks(
                                s_t_h_m,
                                s_t_m_m,
                                s_t_l_m,
                                sp_m,
                                t_m,
                                st_m,
                                config["training"]["target_masking"],
                            ),
                            months.long(),
                            patch_size_high_res=patch_size_high_res,
                            patch_size_med_res=patch_size_med_res,
                            patch_size_low_res=patch_size_low_res,
                            exit_after=config["training"]["target_exit_after"],
                            token_exit_cfg=config["training"]["token_exit_cfg"],
                        )

                    loss = do_loss(
                        training_config,
                        (
                            t_s_t_h,
                            t_s_t_m,
                            t_s_t_l,
                            t_sp,
                            t_t,
                            t_st,
                            p_s_t_h,
                            p_s_t_m,
                            p_s_t_l,
                            p_sp,
                            p_t,
                            p_st,
                            s_t_h_m[:, 0::patch_size_high_res, 0::patch_size_high_res],
                            s_t_m_m[:, 0::patch_size_med_res, 0::patch_size_med_res],
                            s_t_l_m[:, 0::patch_size_low_res, 0::patch_size_low_res],
                            sp_m[:, 0::patch_size_high_res, 0::patch_size_high_res],
                            t_m,
                            st_m,
                        ),
                    )
                else:
                    loss = do_loss(
                        training_config,
                        (
                            p_s_t_h,
                            p_s_t_m,
                            p_s_t_l,
                            p_sp,
                            p_t,
                            p_st,
                            s_t_h_x,
                            s_t_m_x,
                            s_t_l_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_h_m,
                            s_t_m_m,
                            s_t_l_m,
                            sp_m,
                            t_m,
                            st_m,
                            patch_size_high_res,
                            patch_size_med_res,
                            patch_size_low_res,
                            max(training_config["patch_sizes_high_res"]),
                        ),
                    )
                assert not torch.isnan(loss).any(), "NaNs in loss"
                print("Got through one loss calc w/o assertion error - yay!")
            train_loss.update(loss.item(), n=s_t_h_x.shape[0])
            random_masking_train_loss.update(loss.item(), n=s_t_h_x.shape[0])

            loss = loss / iters_to_accumulate
            loss.backward()

            if ((i + 1) % iters_to_accumulate == 0) or (i + 1 == len(dataloader)):
                if training_config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                current_lr = adjust_learning_rate(
                    optimizer,
                    epoch=i / (repeat_aug * len(dataloader)) + e,
                    warmup_epochs=training_config["warmup_epochs"],
                    total_epochs=training_config["num_epochs"],
                    max_lr=training_config["max_lr"],
                    min_lr=training_config["final_lr"],
                )

                with torch.no_grad():
                    try:
                        m = next(momentum_scheduler)
                    except StopIteration:
                        m = training_config["ema"][1]
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
                if wandb_enabled:
                    to_log = {
                        "train_loss": train_loss.average,
                        "random_masking_train_loss": random_masking_train_loss.average,
                        "task_masking_train_loss": task_masking_train_loss.average,
                        "epoch": e,
                        "momentum": m,
                        "lr": current_lr,
                    }
                    # wandb.log(to_log, step=e)
                    """  
                    if (training_config["wandb_plot_every_n_epochs"] != 0) and (e % training_config["wandb_plot_every_n_epochs"] == 0):
                        plot_list_nested = []
                        for image_id, prepared_image in prepared_image_to_plot.items():
                            plot_list_nested.append(
                                plot_space_time_predictions(
                                    epoch=e,
                                    encoder=encoder,
                                    predictor=predictor,
                                    training_config=training_config,
                                    prepared_image=prepared_image,
                                    image_id=image_id,
                                )
                            )

                        for plot in [
                            plot_list
                            for plot_list in plot_list_nested
                        ]:
                            wandb.log({f"plot": plot})
                    """

    """
    if wandb_enabled:
        to_log = {
            "train_loss": train_loss.average,
            "random_masking_train_loss": random_masking_train_loss.average,
            "task_masking_train_loss": task_masking_train_loss.average,
            "epoch": e,
            "momentum": m,
            "lr": current_lr,
        }
    """

    # if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
    #    e % training_config["eval_eurosat_every_n_epochs"] == 0
    # ):
    # to_log.update(
    #    val_task_no_latlons.evaluate_model_on_task(
    #        encoder, model_modes=["KNNat5 Classifier", "KNNat20 Classifier"]
    #    )
    # )
    # to_log.update(
    #    val_task_ts.evaluate_model_on_task(
    #        encoder, model_modes=["KNNat5 Classifier", "Logistic Regression"]
    #    )
    # )
    wandb.log(to_log, step=e)

    if args["checkpoint_every_epoch"] > 0:
        if e % args["checkpoint_every_epoch"] == 0:
            if model_path is None:
                model_path = output_folder
            if not model_path.exists():
                model_path.mkdir()
            if id_dir is None:
                id_dir = timestamp_dirname(run_id)
                id_dir = Path(model_path / id_dir)
                id_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpointing to {model_path}")
            torch.save(encoder.state_dict(), id_dir / ENCODER_FILENAME)
            torch.save(predictor.state_dict(), id_dir / DECODER_FILENAME)
            torch.save(target_encoder.state_dict(), id_dir / TARGET_ENCODER_FILENAME)
            torch.save(optimizer.state_dict(), id_dir / OPTIMIZER_FILENAME)
            config["cur_epoch"] = e + 1
            with (id_dir / CONFIG_FILENAME).open("w") as f:
                json.dump(config, f)


if model_path is None:
    if model_path is None:
        model_path = output_folder
        if not model_path.exists():
            model_path.mkdir()
    if id_dir is None:
        id_dir = timestamp_dirname(run_id)
        id_dir = Path(model_path / id_dir)
        id_dir.mkdir(parents=True, exist_ok=True)
torch.save(encoder.state_dict(), id_dir / ENCODER_FILENAME)
torch.save(predictor.state_dict(), id_dir / DECODER_FILENAME)
torch.save(target_encoder.state_dict(), id_dir / TARGET_ENCODER_FILENAME)
torch.save(optimizer.state_dict(), id_dir / OPTIMIZER_FILENAME)
with (model_path / CONFIG_FILENAME).open("w") as f:
    json.dump(config, f)

# upload the model to google cloud
if args["sync_models_from_service_account"]:
    # authenticate the service account
    os.system(
        "gcloud auth activate-service-account  large-earth-model@appspot.gserviceaccount.com"
    )
os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/outputs {model_path}")

eval_tasks: List[EvalTask] = []
for task in eval_tasks:
    results = task.train_and_evaluate_model_on_task(encoder)
    print(json.dumps(results, indent=2), flush=True)
    if wandb_enabled:
        wandb.log(results, step=training_config["num_epochs"])
# tracker.stop()
