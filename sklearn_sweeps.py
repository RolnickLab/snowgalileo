import argparse
import json
import os
from pathlib import Path

import wandb

from src.config import DEFAULT_SEED
from src.eval.landsat_baselines import LandsatEvalSklearn
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model_type",
    type=str,
    default="rf",
    choices=["rf", "svr", "mlp"],
    help="Type of sklearn model to use: rf (Random Forest), svr (Support Vector Regressor), or mlp (Multi-layer Perceptron).",
)
parser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date.",
)
parser.add_argument(
    "--eval_config_name",
    type=str,
    default="landsat_eval_1_99_test.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
)

args = parser.parse_args()

# TODO: discuss which metric to optimize
rf_sweep_configuration = {
    "name": "sweep_sklearn_rf",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "n_estimators": {"values": [50, 100, 200, 300, 350, 400, 450, 500]},
        "max_depth": {"values": [None, 10, 20, 30, 40, 50]},
        "normalization": {"values": [None, "std"]},
    },
}

svr_sweep_configuration = {
    "name": "sweep_sklearn_svr",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "kernel": {"values": ["linear", "poly", "rbf"]},
        "C_exponent": {"values": [-15, -10, -5, 0, 5, 10, 15]},
        "degree": {"values": [2, 3]},
        "gamma_exponent": {"values": [-5, 0, 5]},
        "normalization": {"values": [None, "std"]},
    },
}

mlp_sweep_configuration = {
    "name": "sweep_sklearn_mlp",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "learning_rate_init": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "normalization": {"values": [None, "std"]},
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
    args = parser.parse_args()

    with wandb.init(project="ai4snow_sweeps_sklearn") as sweep_run:
        with (
            Path("src") / Path("eval") / Path("eval_configs") / Path(args.eval_config_name)
        ).open("r") as f:
            config = json.load(f)

        eval_task = LandsatEvalSklearn(
            normalization="std",
            exclude_prediction_date=False,
            exclude_prediction_high_res=args.exclude_prediction_high_res,
            resample=False,
            eval_config=config,
            model_type=args.model_type,
        )

        sweep_run.config.update(args)

        results = eval_task.fit_sklearn(
            hyperparameters=sweep_run.config,
            save_results=False,
        )

        # log metric to sweep run
        sweep_run.log(
            {
                "r2": results.get("r2", -1),
                "rmse": results.get("rmse", -1),
            }
        )
        sweep_run.finish()


def main():
    wandb.login()

    if args.model_type == "rf":
        sweep_config = rf_sweep_configuration
    else:
        raise NotImplementedError(
            f"Sweep configuration for model type {args.model_type} is not implemented."
        )

    # number of runs in the sweep
    count = 100

    sweep_id = wandb.sweep(sweep=sweep_config, project="ai4snow_sweeps", entity="sea-ice")
    wandb.agent(sweep_id, function=train_and_validate, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
