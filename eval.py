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
from src.utils import device, load_check_config, seed_everything
from src.data.config import DATA_FOLDER

seed_everything(DEFAULT_SEED)
process = psutil.Process()

eval_mode = "evaluate"  # or "visualize_predictions" or "visualize_predictions_best_worst"
resample = False

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str, default="")
argparser.add_argument("--encoder_type", type=str, default="snowgalileo", choices=["gabis_galileo", "snowgalileo"])
args = argparser.parse_args().__dict__

if args["encoder_type"] == "gabis_galileo":
    encoder = Encoder.load_from_folder("galileo/data/models/nano").to(device)
else:
    if args["output_folder"] != "":
        # load pretrained snowgalileo encoder
        encoder = Encoder.load_from_folder(Path(DATA_FOLDER / args["output_folder"])).to(device)
    else:
        # randomly initialized snowgalileo encoder
        config = load_check_config("ai4snow.json")
        encoder = Encoder(**config["model"]["encoder"])

eval_tasks: List[EvalTask] = [
    # geobench EuroSat only works without latlons
    *[LandsatEval(exclude_prediction_high_res=high, evaluation_mode=eval_mode, resample=resample) for high in [True, False]],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(
        pretrained_model=encoder, model_modes=["Regression"]
    )
    print(json.dumps(results, indent=2, default=str), flush=True)
