import wandb
import os
from sklearn.model_selection import KFold
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
from sklearn.metrics import confusion_matrix
from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER
from src.flexipresto import Encoder
from galileo.src.galileo import Encoder as GalileoEncoder
from src.eval import LandsatEval
from src.eval.eval import EvalTask
from src.utils import device, load_check_config, seed_everything
import argparse
from typing import List, Optional
from pathlib import Path

seed_everything(DEFAULT_SEED)

parser = argparse.ArgumentParser(description="PyTorch Unet Training")
parser.add_argument(
    "--pretrain", default="none", type=str, choices=["none", "snow", "galileo"]
)
parser.add_argument(
    "--resample", action="store_true"
)
parser.add_argument(
    "--num_finetune_epochs", type=int, default=50
)

args = parser.parse_args() 
pretrain = args.pretrain

# TODO: potentially change from balanced accuracy to OA
# TODO: do something against class imbalance?
sweep_configuration = {
    "name": f"sweep_pretrain_{args.pretrain}_resample_{args.resample}",
    "method": "random",
    "metric": {"goal": "maximize", "name": "balanced_accuracy"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]},
        "lr_schedule": {"values": [True, False]},
        "warmup_fraction": {"values": [0.0, 0.05, 0.1, 0.2]},
        "batch_size": {"values": [16]},
        "optimizer": {"values": ["Adam", "SGD"]},
        "weight_decay": {"values": [0, 1e-5, 1e-3]},
        "num_workers": {"values": [4]},
        "sigmoid_slope": {"values": [0.01, 0.1, 1.0]},
        "loss_fn": {"values": ["MSE"]},
    },
}

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for key in os.environ.keys():
        if key.startswith("WANDB_") and key not in exclude:
            del os.environ[key]

def train_and_validate():
    args=parser.parse_args()

    with wandb.init(project="ai4snow_sweeps") as sweep_run:
        if args.pretrain == "galileo":
            encoder = GalileoEncoder.load_from_folder(Path("galileo/data/models/nano")).to(device)
        elif args.pretrain == "snow":
            # load pretrained snowgalileo encoder
            encoder = Encoder.load_from_folder(Path(DATA_FOLDER / "outputs/checkpoints_ps10_5/epoch_82/")).to(device)
        else:
            # randomly initialized snowgalileo encoder
            config = load_check_config("ai4snow_ps10.json")
            encoder = Encoder(**config["model"]["encoder"]).to(device)

        eval_tasks: List[EvalTask] = [
            # geobench EuroSat only works without latlons
            *[LandsatEval(exclude_prediction_high_res=False, evaluation_mode="evaluate", resample=sweep_run.config["resample"], finetune=True, num_finetune_epochs=args.num_finetune_epochs)],
        ]
        for task in eval_tasks:
            results = task.evaluate_model_on_task(
                pretrained_model=encoder, model_modes=["Regression"], baseline_galileo=(args.pretrain=="galileo"), sklearn=False, hyperparams_config=sweep_run.config, log_wandb=False
            )
        # log metric to sweep run
        sweep_run.log(
            {
                "r2": results.get("r2", -1),
                "rmse": results.get("rmse", -1),
                "overall_accuracy": results.get("overall_accuracy", -1),
                "balanced_accuracy": results.get("balanced_accuracy", -1),
                "recall": results.get("recall", -1),
                "precision": results.get("precision", -1),
                "f1": results.get("f1", -1),
                "miou": results.get("miou", -1),
            }
        )
        sweep_run.finish()


def main():    
    wandb.login()

    sweep_config = sweep_configuration
    # number of runs in the sweep
    count = 100

    sweep_id = wandb.sweep(sweep=sweep_config, project="ai4snow_sweeps", entity="sea-ice")
    wandb.agent(sweep_id, function=train_and_validate, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
