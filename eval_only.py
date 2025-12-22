import argparse
import json
from pathlib import Path

import psutil
import torch

from src.config import DEFAULT_SEED
from src.eval import (
    LandsatEval,
)
from src.eval.patch_predict import EncoderWithHead
from src.snowgalileo import Encoder
from src.utils import checkpoints_dir, device, load_check_config, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--checkpoint_name",
    type=str,
    default="finetuned_seg_ls_s42_ps10_attn__no_high_res_in_pred_date_final.pth",
)
argparser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date. Should match checkpoint training setup.",
)
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_train_tiny.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
)
argparser.add_argument(
    "--h5pys_only",
    action="store_true",
    help="Where to only use h5pys (faster, but need to be already stored in this format)",
)
args = argparser.parse_args().__dict__

# TODO: fix the EncoderWithHead loading pipeline
# TODO: make sure the eval config matches the training config
with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["config_name"])).open(
    "r"
) as f:
    eval_config = json.load(f)
    default_attn_config = eval_config["attention_probe"]
    sigmoid_slope = eval_config["hyperparams"]["sigmoid_slope"]

# retrieve model size from config filename
raw_filename = args["eval_config"].split(".")[0]
model_size_from_config = raw_filename.split("_")[2]

if args["checkpoint_name"] != "":
    checkpoint_folder = args["pretraining_checkpoint_folder"].split("/")[1]
    model_size_from_checkpoint_folder = checkpoint_folder.split("_")[1]
    assert model_size_from_checkpoint_folder == model_size_from_config
    # load pretrained snowgalileo encoder
    config = load_check_config("ai4snow_ps10.json")
    encoder_random_init = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(
        encoder_random_init, eval_config=default_attn_config, sigmoid_slope=sigmoid_slope
    ).to(device)
    checkpoint = torch.load(Path(checkpoints_dir / args["checkpoint_name"]), map_location=device)
    model.load_state_dict(checkpoint)
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config(f"ai4snow_{model_size_from_config}.json")
    encoder_random_init = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(
        encoder_random_init, eval_config=default_attn_config, sigmoid_slope=sigmoid_slope
    ).to(device)

eval_task = LandsatEval(
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    eval_config=eval_config,
    h5pys_only=args["h5pys_only"],
)

eval_task.evaluate_model_on_task(model=model, id=args["checkpoint_name"])
