import argparse
import json
from pathlib import Path
from typing import List

import psutil
import torch
from sklearn.model_selection import train_test_split

from galileo.src.galileo import Encoder as GalileoEncoder
from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.eval import (
    LandsatEval,
)
from src.eval.eval import EvalTask
from src.flexipresto import Encoder
from src.utils import device, load_check_config, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_folder", type=str, default="outputs/checkpoints_ps10_5/epoch_82/")
argparser.add_argument(
    "--encoder_type", type=str, default="snowgalileo", choices=["orig_galileo", "snowgalileo"]
)
argparser.add_argument(
    "--strategy",
    type=str,
    default="attention_probe",
    choices=["finetune", "linear_probe", "attention_probe", "sklearn"],
    help="Whether to finetune the model, else probe.",
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
    "--eval_config", type=str, default="landsat_eval_5_95.json", help="Which eval config to use."
)
args = argparser.parse_args().__dict__

with (Path(__file__).parents[0] / Path("src/eval/eval_configs") / Path(args["eval_config"])).open(
    "r"
) as f:
    eval_config = json.load(f)

if args["encoder_type"] == "orig_galileo":
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

# TODO: move this somewhere else
# create dataset split on the fly, so we don't have to store multiple copies
# NOTE: assumes that all input files are in h5py folder
if (
    eval_config["data"]["split_type"] in eval_config
    and eval_config["data"]["split_type"] == "train_val_test_random"
):
    eval_config["data"]["train_val_test_split"] = [0.7, 0.15, 0.15]
    input_path = Path(DATA_FOLDER / eval_config["data"]["input_tif_folder"])
    mask_path = Path(DATA_FOLDER / eval_config["data"]["label_folder"])
    h5pys_path = Path(DATA_FOLDER / eval_config["data"]["input_h5py_folder"])

    assert (
        len(list(input_path.glob("*.tif")))
        == len(list(mask_path.glob("*.tif")))
        == len(list(h5pys_path.glob("*.h5py")))
    )

    # Make sure input_files and mask_files are properly matched
    # both should contain the same filenames in corresponding order
    input_files = sorted(Path(input_path).glob("*.h5py"))
    mask_files = sorted(Path(mask_path).glob("*.tif"))

    assert all(f.stem == m.stem for f, m in zip(input_files, mask_files)), (
        "Input and mask files not aligned!"
    )

    # Pair them together before splitting
    pairs = list(zip(input_files, mask_files))

    X_train, X_temp, y_train, y_temp = train_test_split(
        pairs, test_size=0.3, random_state=DEFAULT_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=DEFAULT_SEED
    )

    # zip back to pairs
    train_pairs = list(zip(X_train, y_train))
    val_pairs = list(zip(X_val, y_val))
    test_pairs = list(zip(X_test, y_test))


eval_tasks: List[EvalTask] = [
    # geobench EuroSat only works without latlons
    *[
        LandsatEval(
            exclude_prediction_high_res=args["exclude_prediction_high_res"],
            resample=args["resample"],
            decoder_mode=args["strategy"],
            num_finetune_epochs=args["num_finetune_epochs"],
            eval_config=eval_config,
        )
    ],
]
for task in eval_tasks:
    results = task.train_and_evaluate_model_on_task(
        pretrained_model=encoder,
        model_modes=["Regression"],
        baseline_galileo=(args["encoder_type"] == "orig_galileo"),
        log_wandb=True,
        initialization_id=initialization_id,
        save_final_checkpoint=args["save_final_checkpoint"],
    )
    print(json.dumps(results, indent=2, default=str), flush=True)
