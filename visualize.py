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
from src.data.config import DATA_FOLDER

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint_name", type=str, default="finetuned_seg_ls_s42_ps10_attn__no_high_res_in_pred_date_final.pth")
argparser.add_argument("--output_folder", type=str, default="outputs/checkpoints_ps10_5/epoch_82/")
args = argparser.parse_args().__dict__

# TODO: fix the EncoderWithHead loading pipeline
# TODO: make sure the eval config matches the training config
with (Path("src") / Path("eval") / Path("eval_configs") / Path("landsat_eval_1_99_test.json")).open("r") as f:
    config = json.load(f)
    default_attn_config = config["attention_probe"]
    sigmoid_slope = config["hyperparams"]["sigmoid_slope"]

if args["checkpoint_name"] != "":
    # load pretrained snowgalileo encoder
    config = load_check_config("ai4snow_ps10.json")
    encoder_random_init = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(encoder_random_init, eval_config=default_attn_config, sigmoid_slope=sigmoid_slope).to(device)
    checkpoint = torch.load(checkpoints_dir() / args["checkpoint_name"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args["output_folder"] != "":
        # for debugging, load the pretrained encoder as well
        encoder_pretrained = Encoder.load_from_folder(Path(DATA_FOLDER / args["output_folder"])).to(device)
        model_debug = EncoderWithHead(encoder_pretrained, eval_config=default_attn_config, sigmoid_slope=sigmoid_slope).to(device)
        model_debug.load_state_dict(checkpoint["model_state_dict"])
        assert all(p1.equal(p2) for p1, p2 in zip(model.parameters(), model_debug.parameters())), "Model parameters do not match between random and pretrained encoder!"
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config("ai4snow_ps10.json")
    encoder = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(encoder, eval_config=default_attn_config, sigmoid_slope=sigmoid_slope).to(device)

eval_task = LandsatEval()

eval_task.visualize_sample_predictions(model=model, log_wandb=True)
