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

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str, default="")
args = argparser.parse_args().__dict__

if args["output_folder"] != "":
    encoder = Encoder.load_from_folder(Path(DATA_FOLDER / args["output_folder"])).to(device)
else:
    config = load_check_config("ai4snow.json")
    encoder = Encoder(**config["model"]["encoder"])

eval_tasks: List[EvalTask] = [
    # geobench EuroSat only works without latlons
    *[LandsatEval(exclude_prediction_date=excl) for excl in [True, False]],
]
for task in eval_tasks:
    results = task.evaluate_model_on_task(pretrained_model=encoder)
    print(json.dumps(results, indent=2), flush=True)
