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
from src.flexipresto import Encoder
from src.utils import device, load_check_config, seed_everything
from src.utils import checkpoints_dir
from src.eval.patch_predict import EncoderWithHead

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint_name", type=str, default="")
args = argparser.parse_args().__dict__

# TODO: fix the EncoderWithHead loading pipeline
with (Path("src") / Path("eval") / Path("eval_configs") / Path("landsat_eval_1_99_test.json")).open("r") as f:
    config = json.load(f)
    default_attn_config = config["attention_probe"]

if args["checkpoint_name"] != "":
    # load pretrained snowgalileo encoder
    config = load_check_config("ai4snow_ps10.json")
    encoder = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(encoder, eval_config=default_attn_config).to(device)
    model = Encoder.load_from_folder(Path(checkpoints_dir/ args["checkpoint_name"])).to(device)
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config("ai4snow_ps10.json")
    encoder = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(encoder).to(device)
    raise NotImplementedError("Loading full model from scratch not implemented yet.")

eval_task = LandsatEval()

eval_task.visualize_sample_predictions(model=model, log_wandb=True)
