import argparse
import json
from pathlib import Path
from typing import List

import psutil
import torch

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.eval import (
    TimeseriesAblationsEval,
)
from src.eval.eval import EvalTask
from src.snowgalileo import Encoder
from src.utils import device, load_check_config, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--pretraining_checkpoint_folder",
    type=str,
    default="outputs/checkpoints_tiny/epoch_100",
    help="Path to folder containing pretrained checkpoint.",
)
# TODO: make the choices of naming more descriptive
argparser.add_argument(
    "--decoding_strategy",
    type=str,
    default="attention_probe",
    choices=["finetune", "linear_probe", "attention_probe", "sklearn"],
    help="Decoding strategy to use. 'Finetune' uses a linear decoder and finetunes the entire model. 'Linear_probe' uses a linear decoder and only trains the decoder. 'Attention_probe' uses an attention-based decoder and fine-tunes the entire model. 'sklearn' uses the frozen encoder features for a sklearn model.",
)
argparser.add_argument("--resample", action="store_true", help="Whether to use oversampling.")
argparser.add_argument(
    "--num_finetune_epochs", type=int, default=25, help="Number of epochs to finetune for."
)
argparser.add_argument(
    "--save_final_checkpoint",
    action="store_true",
    help="Whether to save the final checkpoint after finetuning.",
)
argparser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date.",
)
argparser.add_argument(
    "--eval_config",
    type=str,
    default="ablate_timeseries_tiny.json",
    help="Determines the ablation mode. Options are stored in src/eval/eval_configs/",
)
argparser.add_argument(
    "--h5pys_only",
    action="store_true",
    help="Where to only use h5pys (faster, but need to be already stored in this format)",
)
args = argparser.parse_args().__dict__

with (Path("src/eval/eval_configs") / Path(args["eval_config"])).open("r") as f:
    eval_config = json.load(f)

# retrieve model size from config filename
raw_filename = args["eval_config"].split(".")[0]
model_size_from_config = raw_filename.split("_")[-1]

if args["pretraining_checkpoint_folder"] != "":
    checkpoint_folder = args["pretraining_checkpoint_folder"].split("/")[1]
    model_size_from_checkpoint_folder = checkpoint_folder.split("_")[1]
    assert model_size_from_checkpoint_folder == model_size_from_config
    # load pretrained snowgalileo encoder
    encoder = Encoder.load_from_folder(
        Path(DATA_FOLDER / args["pretraining_checkpoint_folder"])
    ).to(device)
    initialization_id = "snowgalileo_pretrained"
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config(f"ai4snow_{model_size_from_config}.json")
    encoder = Encoder(**config["model"]["encoder"]).to(device)
    initialization_id = "snowgalileo_random"

eval_tasks: List[EvalTask] = [
    *[
        TimeseriesAblationsEval(
            exclude_prediction_high_res=args["exclude_prediction_high_res"],
            decoder_mode=args["decoding_strategy"],
            num_finetune_epochs=args["num_finetune_epochs"],
            eval_config=eval_config,
            h5pys_only=args["h5pys_only"],
        )
    ],
]
# TODO: remove the model_mode argument since it is sklearn-specific
# TODO: eventually remove sklearn from eval tasks altogether if we don't use it
for task in eval_tasks:
    results = task.train_and_evaluate_model_on_task(
        pretrained_model=encoder,
        model_modes=["Regression"],
        log_wandb=True,
        initialization_id=initialization_id,
        save_final_checkpoint=args["save_final_checkpoint"],
    )
    print(json.dumps(results, indent=2, default=str), flush=True)
