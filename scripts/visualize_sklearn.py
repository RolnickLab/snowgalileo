import argparse
import json
from pathlib import Path

import psutil
import torch
import joblib

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
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
    "--pretraining_checkpoint_folder",
    type=str,
    default="outputs/checkpoints_tiny/epoch_100",
    help="Path to folder containing pretrained checkpoint.",
)
argparser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date. Should match checkpoint training.",
)
argparser.add_argument("--sklearn_model_path", type=str, default="")
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_test_rockies_tiny.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
)
args = argparser.parse_args().__dict__

# TODO: fix the EncoderWithHead loading pipeline
# TODO: make sure the eval config matches the training config
with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["eval_config_name"])).open(
    "r"
) as f:
    eval_config = json.load(f)
    sigmoid_slope = eval_config["hyperparameters_snowgalileo"]["sigmoid_slope"]

# retrieve model size from config filename
raw_filename = args["eval_config_name"].split(".")[0]
model_size_from_config = raw_filename.split("_")[-1]

if args["pretraining_checkpoint_folder"] != "":
    checkpoint_folder = args["pretraining_checkpoint_folder"].split("/")[1]
    model_size_from_checkpoint_folder = checkpoint_folder.split("_")[1]
    assert model_size_from_checkpoint_folder == model_size_from_config
    # load pretrained snowgalileo encoder
    encoder = Encoder.load_from_folder(
        Path(DATA_FOLDER / args["pretraining_checkpoint_folder"])
    ).to(device).eval()
    initialization_id = "snowgalileo_pretrained"
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config(f"ai4snow_{model_size_from_config}.json")
    encoder = Encoder(**config["model"]["encoder"]).to(device).eval()
    initialization_id = "snowgalileo_random"

# read sklearn checkpoint
model = joblib.load(args["sklearn_model_path"])
sklearn_models = []
sklearn_models.append(model)

eval_task = LandsatEval(
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    eval_config=eval_config,
    h5pys_only=False,
)

eval_task.visualize_sample_predictions(
    model=encoder, log_wandb=True, sklearn=True, sklearn_models=sklearn_models
)
