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
argparser.add_argument(
    "--decoding_strategy",
    type=str,
    default="attention_probe",
    choices=["finetune", "linear_probe", "attention_probe"],
    help="Decoding strategy to use. 'Finetune' uses a linear decoder and finetunes the entire model. 'Linear_probe' uses a linear decoder and only trains the decoder. 'Attention_probe' uses an attention-based decoder and fine-tunes the entire model. 'sklearn' uses the frozen encoder features for a sklearn model.",
)
args = argparser.parse_args().__dict__

decoder_mode = args["decoding_strategy"]

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

if args["checkpoint_name"] != "":
    # load pretrained snowgalileo encoder
    # sigmoid slope is ignored when linear head is used
    config = load_check_config(f"ai4snow_{model_size_from_config}.json")
    encoder_random_init = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(
        encoder_random_init, eval_config=eval_config[decoder_mode], sigmoid_slope=sigmoid_slope
    ).to(device)
    checkpoint = torch.load(Path(checkpoints_dir / args["checkpoint_name"]), map_location=device)
    model.load_state_dict(checkpoint)
else:
    # randomly initialized snowgalileo encoder
    config = load_check_config(f"ai4snow_{model_size_from_config}.json")
    encoder_random_init = Encoder(**config["model"]["encoder"])
    model = EncoderWithHead(
        encoder_random_init, eval_config=eval_config[decoder_mode], sigmoid_slope=sigmoid_slope
    ).to(device)

eval_task = LandsatEval(
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    eval_config=eval_config,
    h5pys_only=args["h5pys_only"],
)

eval_task.evaluate_model_on_task(model=model, id=args["checkpoint_name"])
