import argparse
import json
from pathlib import Path

import joblib

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset
from src.fsc.landsat_baselines import LandsatEvalSklearn
from src.utils import config_dir, seed_everything

seed_everything(DEFAULT_SEED)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date.",
)
argparser.add_argument(
    "--exclude_prediction_sensors",
    action="store_true",
    help="Whether to exclude observational sensors in prediction date.",
)
argparser.add_argument(
    "--exclude_prediction_date",
    action="store_true",
    help="Whether to exclude prediction date.",
)
argparser.add_argument(
    "--include_prediction_era5",
    action="store_true",
    help="Whether to include ERA5 in prediction date.",
)
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_test_rockies_tiny.json",
    help="Config name for evaluation. Options are stored in configs/eval/",
)
argparser.add_argument(
    "--model_type",
    type=str,
    default="rf",
    choices=["rf", "svr", "mlp"],
    help="Type of model to train: rf, svr, or mlp.",
)
argparser.add_argument(
    "--normalization",
    type=str,
    default="std",
    choices=["std", ""],
)
argparser.add_argument(
    "--model_checkpoint_path", default="landsat_rf_model_rf_50est_19012026.joblib"
)
argparser.add_argument(
    "--run_id",
    type=str,
    default="default"
)
args = argparser.parse_args().__dict__

if "rockies" in args["eval_config_name"]:
    id = f"rockies_{args["run_id"]}"
elif "switzerland" in args["eval_config_name"]:
    id = f"switzerland_{args["run_id"]}"
else:
    raise ValueError(f"Unknown eval_config_name {args['eval_config_name']}")

with (Path("configs") / Path("eval") / Path(args["eval_config_name"])).open("r") as f:
    config = json.load(f)

# we use the normalization values for missing data imputation so we load it independently
normalizing_dict = Dataset.load_normalization_values(path=config_dir / NORMALIZATION_DICT_FILENAME)

# read model checkpoint
model = joblib.load(args["model_checkpoint_path"])

rf = LandsatEvalSklearn(
    normalization=args["normalization"],
    exclude_prediction_date=args["exclude_prediction_date"],
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    exclude_prediction_sensors=args["exclude_prediction_sensors"],
    exclude_prediction_era5=not args["include_prediction_era5"],
    resample=False,
    eval_config=config,
    model_type=args["model_type"],
    normalizing_dict=normalizing_dict,
)
rf.predict_only(model=model, id=id, save_results=True, normalization=args["normalization"],)
