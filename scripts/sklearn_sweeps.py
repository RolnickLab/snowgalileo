import argparse
import json
import os
from pathlib import Path

import wandb

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset
from src.fsc.landsat_baselines import LandsatEvalSklearn
from src.utils import config_dir, seed_everything

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
    default="fsc_train_balanced_tiny.json",
    help="Config name for evaluation. Options are stored in configs/finetune/",
)
parser.add_argument(
    "--h5pys_only",
    action="store_true",
    help="Where to only use h5pys (faster, but need to be already stored in this format)",
)
args = parser.parse_args()

# Rittger et al.: https://www.sciencedirect.com/science/article/pii/S003442572100328X#s0010 tested n_estimators from 100 - 500 in increments of 100
# Sklearn definitions:
# n_estimators == number of trees
# max_features == mtry = [p / 3], where p==total number of predictor variables (following Kuter et al.)
# min_samples_leaf = 5 (following Kuter et al. & Rittger et al.)
rf_sweep_configuration = {
    "name": "sweep_sklearn_rf",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "n_estimators": {"values": [50, 100, 200, 300, 400, 500]},
        "normalization": {"values": [None, "std"]},
        "max_features": {"values": ["feature_dependent", "sqrt", "log2"]},
        "min_samples_leaf": {"values": [1, 2, 5]},
        "max_depth": {"values": [None, 10, 20, 30]},
        "min_samples_split": {"values": [2, 5, 10]},
    },
}

# Following Kuter et al.: https://www.sciencedirect.com/science/article/pii/S0034425721000122#bb0360
# kernel function and regularization parameter C are the sensitive hyperparameters.
# Polynomial kernel depends on degree d. RBF depends on kernel width gamma.
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
        "epsilon": {"values": [0.01, 0.1, 0.5, 1.0]},
        "max_iter": {"values": [500, 1000, 5000, 10000]},
        "bagging": {"values": [True, False]},
    },
}

# Following Kuter et al. (2018)
mlp_sweep_configuration = {
    "name": "sweep_sklearn_mlp",
    "method": "random",
    "metric": {"goal": "maximize", "name": "r2"},
    "parameters": {
        "learning_rate_init": {"values": [0.0001, 0.001, 0.01, 0.1]},
        "activation": {"values": ["logistic", "tanh", "relu"]},
        "normalization": {"values": [None, "std"]},
        "solver": {"values": ["adam"]},
        "alpha": {"values": [1e-5, 1e-4, 1e-3]},
        "batch_size": {"values": [64, 128, 256, "auto"]},
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

    with wandb.init(project=f"ai4snow_{args.model_type}_sweeps_small_set") as sweep_run:
        with (Path("configs") / Path("finetune") / Path(args.eval_config_name)).open("r") as f:
            config = json.load(f)

        # we use the normalization values for missing data imputation so we load it independently
        normalizing_dict = Dataset.load_normalization_values(
            path=config_dir / NORMALIZATION_DICT_FILENAME
        )

        eval_task = LandsatEvalSklearn(
            normalization="std",
            exclude_prediction_date=False,
            exclude_prediction_high_res=args.exclude_prediction_high_res,
            exclude_prediction_era5=True,
            resample=False,
            eval_config=config,
            model_type=args.model_type,
            h5pys_only=args.h5pys_only,
            normalizing_dict=normalizing_dict,
        )

        sweep_run.config.update(args)

        results = eval_task.fit_sklearn(
            hyperparameters=sweep_run.config,
            save_results=False,
            sweep_run=sweep_run,
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
    elif args.model_type == "svr":
        sweep_config = svr_sweep_configuration
    elif args.model_type == "mlp":
        sweep_config = mlp_sweep_configuration

    # number of runs in the sweep
    count = 100

    sweep_id = wandb.sweep(
        sweep=sweep_config, project=f"ai4snow_{args.model_type}_sweeps_small_set", entity="sea-ice"
    )
    wandb.agent(sweep_id, function=train_and_validate, count=count)

    wandb.finish()


if __name__ == "__main__":
    main()
