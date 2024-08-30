import argparse
import copy
import json
import os
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union, cast

import codecarbon
import psutil
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.beaker import is_beaker_job, maybe_get_beaker_config
from src.collate_fns import mae_collate_fn
from src.conditioner import LearnedMixture, LoRAGenerator
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
    TIFS_FOLDER,
)
from src.eval import (
    BinaryCropHarvestEval,
    EuroSatEval,
)
from src.eval.eval import EvalTask, Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.loss import do_loss
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
argparser.add_argument("--h5py_folder", type=str, default="")
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--download", dest="download", action="store_true")
argparser.add_argument("--h5pys_only", dest="h5pys_only", action="store_true")
argparser.add_argument("--num_workers", dest="num_workers", default=Hyperparams.num_workers)
argparser.add_argument("--batch_size", dest="batch_size", default="")
argparser.add_argument("--sync_models_from_service_account", action="store_true")
argparser.add_argument("--checkpoint_every_epoch", type=int, default=-1)

argparser.set_defaults(download=False)
argparser.set_defaults(cache_in_ram=False)
args = argparser.parse_args().__dict__

if args["h5py_folder"] == "":
    cache_folder = DATA_FOLDER / "h5pys"
else:
    cache_folder = Path(args["h5py_folder"])


if args["output_folder"] == "":
    output_folder = OUTPUT_FOLDER
else:
    output_folder = Path(args["output_folder"])

restart = False
model_path: Optional[Path] = None
start_epoch = 0
run_id = None
wandb_enabled = True
wandb_org = "nasa-harvest"
wandb_output_dir = Path(__file__).parent

if is_beaker_job():
    # see if the output folder exists. If so, there
    # was an existing job
    # "when a job is preempted, it gets a new result dataset,
    # which starts as a copy of the previous job's results."
    output_dirs = [o for o in output_folder.glob("*") if o.is_dir()]
    if len(output_dirs) > 0:
        assert len(output_dirs) == 1, f"Got more than one output dir: {output_dirs}"
        restart = True
        model_path = output_dirs[0]
        print(f"Restarting run using {model_path}")
        with (model_path / CONFIG_FILENAME).open("r") as f:
            config = json.load(f)
        run_name = config["run_name"]
        start_epoch = config["cur_epoch"]
        run_id = config["wandb_run_id"]

if not restart:
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
    config["run_name"] = run_name

run = wandb.init(
    name=run_name, entity=wandb_org, project="flexipresto", dir=wandb_output_dir, id=run_id
)
run_id = cast(Run, run).id
config["wandb_run_id"] = run_id
if is_beaker_job():
    beaker_config = maybe_get_beaker_config()
    config.update(vars(beaker_config))
wandb.config.update(config)

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
    TIFS_FOLDER,
    download=args["download"],
    h5py_folder=cache_folder,
    h5pys_only=args["h5pys_only"],
)
config["training"]["training_samples"] = len(dataset)

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

dataloader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=int(args["num_workers"]),
    collate_fn=partial(
        mae_collate_fn,
        patch_sizes=training_config["patch_sizes"],
        shape_time_combinations=training_config["shape_time_combinations"],
        encode_ratio=training_config["encode_ratio"],
        decode_ratio=training_config["decode_ratio"],
        augmentation_strategies=training_config["augmentation"],
        masking_probabilities=training_config["masking_probabilities"],
        unmasking_probabilities=training_config["unmasking_probabilities"],
    ),
    pin_memory=True,
)

print("Loading models")
predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
param_groups = [
    {
        "params": predictor.parameters(),
        "name": "decoder",
        "weight_decay": training_config["weight_decay"],
    }
]
eval_w_condition = False
if "conditioner" in config["model"]:
    eval_w_condition = True
    if training_config["conditioner_mode"] == "moe":
        encoder_conditioner: Union[LearnedMixture, LoRAGenerator] = LearnedMixture(
            **config["model"]["conditioner"]
        ).to(device)
    elif training_config["conditioner_mode"] == "lora":
        encoder_conditioner = LoRAGenerator(**config["model"]["conditioner"]).to(device)

    encoder = Encoder(**config["model"]["encoder"], conditioner=encoder_conditioner).to(device)
    param_groups.extend(
        [
            {
                "params": [p for n, p in encoder.named_parameters() if "conditioner" not in n],
                "name": "encoder",
                "weight_decay": training_config["weight_decay"],
            },
            {
                "params": encoder.conditioner.parameters(),
                "name": "conditioner",
                "weight_decay": training_config["conditioner_weight_decay"],
            },
        ]
    )
else:
    encoder = Encoder(**config["model"]["encoder"]).to(device)
    param_groups.append(
        {
            "params": encoder.parameters(),
            "name": "encoder",
            "weight_decay": training_config["weight_decay"],
        }
    )

if restart:
    assert model_path is not None
    encoder.load_state_dict(torch.load(model_path / ENCODER_FILENAME, map_location=device))
    predictor.load_state_dict(torch.load(model_path / DECODER_FILENAME, map_location=device))


print("Loading validation task")
val_task_no_latlons = EuroSatEval(
    normalization=training_config["normalization"],
    geobench=True,
    rgb=False,
    include_latlons=False,
    do_condition=eval_w_condition,
)

optimizer = torch.optim.AdamW(
    param_groups,
    lr=0,
    weight_decay=training_config["weight_decay"],
    betas=(training_config["betas"][0], training_config["betas"][1]),
)  # type: ignore
if restart:
    assert model_path is not None
    optimizer.load_state_dict(torch.load(model_path / OPTIMIZER_FILENAME, map_location=device))

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
if restart:
    assert model_path is not None
    target_encoder.load_state_dict(
        torch.load(model_path / TARGET_ENCODER_FILENAME, map_location=device)
    )
    # we also want to step through the momentum scheduler since we are going to fast forward training
    for momentum_epoch in range(start_epoch):
        for i in steps_per_epoch:
            _ = next(momentum_scheduler)

for p in target_encoder.parameters():
    p.requires_grad = False

skipped_batches = 0
for e in tqdm(range(training_config["num_epochs"])):
    if e >= start_epoch:
        i = 0
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
                    patch_size,
                    c_i,
                ) = b

                if (
                    will_cause_nans(s_t_x)
                    or will_cause_nans(sp_x)
                    or will_cause_nans(t_x)
                    or will_cause_nans(st_x)
                ):
                    skipped_batches += 1
                    warnings.warn(f"Skipping batch with NaNs, {skipped_batches}")
                    continue

                if c_i is not None:
                    # there is probably a better way to do this
                    c_i = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in c_i.items()
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
                    )

                    with torch.no_grad():
                        t_s_t, t_sp, t_t, t_st, _, _, _, _, _ = target_encoder(
                            s_t_x,
                            sp_x,
                            t_x,
                            st_x,
                            ~(s_t_m == 2),  # we want 0s where the mask == 2
                            ~(sp_m == 2),
                            ~(t_m == 2),
                            ~(st_m == 2),
                            months.long(),
                            patch_size=patch_size,
                            c_i=c_i if training_config["target_condition"] else None,
                            exit_after=training_config["target_exit_after"],
                        )

                    loss = do_loss(
                        training_config,
                        (
                            t_s_t,
                            t_sp,
                            t_t,
                            t_st,
                            p_s_t,
                            p_sp,
                            p_t,
                            p_st,
                            s_t_m[:, 0::patch_size, 0::patch_size],
                            sp_m[:, 0::patch_size, 0::patch_size],
                            t_m,
                            st_m,
                        ),
                    )
                    assert not torch.isnan(loss).any(), "NaNs in loss"
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
                    current_lr = adjust_learning_rate(
                        optimizer,
                        epoch=i / (repeat_aug * len(dataloader)) + e,
                        warmup_epochs=training_config["warmup_epochs"],
                        total_epochs=training_config["num_epochs"],
                        max_lr=training_config["max_lr"],
                        min_lr=training_config["final_lr"],
                        conditioner_multiplier=training_config["conditioner_multiplier"],
                    )

                    with torch.no_grad():
                        try:
                            m = next(momentum_scheduler)
                        except StopIteration:
                            m = training_config["ema"][1]
                        for param_q, param_k in zip(
                            encoder.parameters(), target_encoder.parameters()
                        ):
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

        if (training_config["eval_eurosat_every_n_epochs"] != 0) and (
            e % training_config["eval_eurosat_every_n_epochs"] == 0
        ):
            results = val_task_no_latlons.evaluate_model_on_task(
                encoder, model_modes=["KNNat5 Classifier", "KNNat20 Classifier"]
            )
            to_log.update(results)
        wandb.log(to_log)
    if args["checkpoint_every_epoch"] != -1:
        if e % args["checkpoint_every_epoch"] == 0:
            if model_path is None:
                model_path = output_folder / timestamp_dirname(run_id)
                model_path.mkdir()
            torch.save(encoder.state_dict(), model_path / ENCODER_FILENAME)
            torch.save(predictor.state_dict(), model_path / DECODER_FILENAME)
            torch.save(target_encoder.state_dict(), model_path / TARGET_ENCODER_FILENAME)
            torch.save(optimizer.state_dict(), model_path / OPTIMIZER_FILENAME)
            config["cur_epoch"] = e
            with (model_path / CONFIG_FILENAME).open("w") as f:
                json.dump(config, f)

if model_path is None:
    model_path = output_folder / timestamp_dirname(run_id)
    model_path.mkdir()
torch.save(encoder.state_dict(), model_path / ENCODER_FILENAME)
torch.save(predictor.state_dict(), model_path / DECODER_FILENAME)
torch.save(target_encoder.state_dict(), model_path / TARGET_ENCODER_FILENAME)
torch.save(optimizer.state_dict(), model_path / OPTIMIZER_FILENAME)
with (model_path / CONFIG_FILENAME).open("w") as f:
    json.dump(config, f)

# upload the model to google cloud
if args["sync_models_from_service_account"]:
    # authenticate the service account
    os.system(
        "gcloud auth activate-service-account  large-earth-model@appspot.gserviceaccount.com"
    )
os.system(f"gcloud storage rsync -r gs://{EE_BUCKET_TIFS}/outputs {model_path}")

eval_tasks: List[EvalTask] = [
    *[
        BinaryCropHarvestEval(
            normalizer=cast(Normalizer, dataset.normalizer), country=country, do_condition=True
        )
        for country in ["Kenya", "Togo", "Brazil"]
    ],
    *[
        EuroSatEval(
            normalization=training_config["normalization"],
            rgb=rgb,
            include_latlons=False,
            geobench=True,
            do_condition=True,
        )
        for rgb in [True, False]
    ],
    *[
        EuroSatEval(
            normalization=training_config["normalization"],
            rgb=rgb,
            include_latlons=include_latlons,
            geobench=False,
        )
        for rgb in [True, False]
        for include_latlons in [True, False]
    ],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(encoder)
    print(json.dumps(results, indent=2), flush=True)
    if wandb_enabled:
        wandb.log(results)
tracker.stop()
