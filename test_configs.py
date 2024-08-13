import argparse
import copy
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.collate_fns import mae_collate_fn
from src.conditioner import LearnedMixture, LoRAGenerator
from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import (
    DATA_FOLDER,
    TIFS_FOLDER,
)
from src.eval.eval import Hyperparams
from src.flexipresto import Encoder, PrestoPixelDecoder, adjust_learning_rate
from src.loss import do_loss
from src.utils import (
    AverageMeter,
    device,
    is_bf16_available,
    check_config,
    seed_everything,
)
from generate_random_config import get_random_config

seed_everything(DEFAULT_SEED)

torch.backends.cuda.matmul.allow_tf32 = True
autocast_device = torch.bfloat16 if is_bf16_available() else torch.float32

argparser = argparse.ArgumentParser()
argparser.add_argument("--config_file", type=str, default="medium.json")
argparser.add_argument("--cache_folder", type=str, default="")
args = argparser.parse_args().__dict__

if args["cache_folder"] == "":
    cache_folder = DATA_FOLDER / "h5pys"
else:
    cache_folder = Path(args["cache_folder"])

for _ in range(1_000):
    config = get_random_config()
    config = check_config(config)
    print(config)
    training_config = config["training"]

    dataset = Dataset(TIFS_FOLDER, download=False, h5py_folder=cache_folder, h5pys_only=True)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=Hyperparams.num_workers,
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

    predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
    param_groups = []
    eval_w_condition = False
    if "encoder_conditioner" in config["model"]:
        eval_w_condition = True
        if training_config["conditioner_mode"] == "moe":
            encoder_conditioner = LearnedMixture(**config["model"]["encoder_conditioner"]).to(device)
        elif training_config["conditioner_mode"] == "lora":
            encoder_conditioner = LoRAGenerator(**config["model"]["encoder_conditioner"]).to(device)
        
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
                    "name": "encoder_conditioner",
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

    predictor = PrestoPixelDecoder(**config["model"]["decoder"]).to(device)
    param_groups.append(
        {
            "params": predictor.parameters(),
            "name": "decoder",
            "weight_decay": training_config["weight_decay"],
        }
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=training_config["start_lr"],
        weight_decay=training_config["weight_decay"],
        betas=(training_config["betas"][0], training_config["betas"][1]),
    )  # type: ignore

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
    for p in target_encoder.parameters():
        p.requires_grad = False

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
                        start_lr=training_config["start_lr"],
                        min_lr=training_config["final_lr"],
                        conditioner_multiplier=training_config["conditioner_multiplier"],
                    )

                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
            if i > 50:
                break
        break
