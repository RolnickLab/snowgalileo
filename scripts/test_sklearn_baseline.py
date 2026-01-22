import argparse
import json
from pathlib import Path

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset
from src.eval.landsat_baselines import LandsatEvalSklearn
from src.utils import config_dir, seed_everything
import joblib

seed_everything(DEFAULT_SEED)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--exclude_prediction_high_res",
    action="store_true",
    help="Whether to exclude high-res in prediction date. Should match checkpoint training setup.",
)
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_test_rockies_tiny.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
)
argparser.add_argument(
    "--model_type",
    type=str,
    default="rf",
    choices=["rf", "svr", "mlp"],
    help="Type of model to train: rf, svr, or mlp.",
)
argparser.add_argument(
    "--model_checkpoint_path",
    default="landsat_rf_model_rf_50est_19012026.joblib"
)
args = argparser.parse_args().__dict__

if "rockies" in args["eval_config_name"]:
    id = "rockies"
elif "switzerland" in args["eval_config_name"]:
    id = "switzerland"
else:
    raise ValueError(f"Unknown eval_config_name {args['eval_config_name']}")

with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["eval_config_name"])).open(
    "r"
) as f:
    config = json.load(f)

# we use the normalization values for missing data imputation so we load it independently
normalizing_dict = Dataset.load_normalization_values(path=config_dir / NORMALIZATION_DICT_FILENAME)

# read model checkpoint
model = joblib.load(args["model_checkpoint_path"])

rf = LandsatEvalSklearn(
    normalization="std",
    exclude_prediction_date=False,
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    resample=False,
    eval_config=config,
    model_type=args["model_type"],
    normalizing_dict=normalizing_dict,
)
rf.predict_only(id=id, save_results=True)
