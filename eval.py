import argparse
import json
from pathlib import Path
from typing import List

import psutil
import torch

from src.config import DEFAULT_SEED
from src.eval import (
    LandsatEval,
)
from src.eval.eval import EvalTask
from src.flexipresto import Encoder
from galileo.src.galileo import Encoder as GalileoEncoder
from src.utils import device, load_check_config, seed_everything
from src.data.config import DATA_FOLDER

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--encoder_type", type=str, default="snowgalileo", choices=["gabis_galileo", "snowgalileo"])
argparser.add_argument("--strategy", type=str, default="attention_probe", choices=["finetune", "linear_probe", "attention_probe", "sklearn"], help="Whether to finetune the model, else probe.")
argparser.add_argument("--eval_mode", type=str, default="evaluate", choices=["evaluate", "visualize_predictions", "visualize_predictions_best_worst"])
argparser.add_argument("--resample", action="store_true", help="Whether to use oversampling.")
argparser.add_argument("--num_finetune_epochs", type=int, default=25, help="Number of epochs to finetune for.")
argparser.add_argument("--save_final_checkpoint", action="store_true", help="Whether to save the final checkpoint after finetuning.")
args = argparser.parse_args().__dict__

if args["encoder_type"] == "gabis_galileo":
    encoder = GalileoEncoder.load_from_folder(Path("galileo/data/models/nano")).to(device)
    initialization_id = "galileo_pretrained"
else:
    if args["output_folder"] != "":
        # load pretrained snowgalileo encoder
        encoder = Encoder.load_from_folder(Path(DATA_FOLDER / args["output_folder"])).to(device)
        initialization_id = "snowgalileo_pretrained"
    else:
        # randomly initialized snowgalileo encoder
        config = load_check_config("ai4snow_ps10.json")
        encoder = Encoder(**config["model"]["encoder"]).to(device)
        initialization_id = "snowgalileo_random"

eval_tasks: List[EvalTask] = [
    # geobench EuroSat only works without latlons
    *[LandsatEval(
        exclude_prediction_high_res=high, 
        evaluation_mode=args["eval_mode"], 
        resample=args["resample"], 
        decoder_mode=args["strategy"],
        num_finetune_epochs=args["num_finetune_epochs"],
        ) for high in [True, False]
    ],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(
        pretrained_model=encoder, model_modes=["Regression"], baseline_galileo=(args["encoder_type"]=="gabis_galileo"), log_wandb=True, initialization_id=initialization_id
    )
    print(json.dumps(results, indent=2, default=str), flush=True)
