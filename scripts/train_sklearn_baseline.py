import argparse
import json
from pathlib import Path

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
    help="Whether to exclude high-res in prediction date. Should match checkpoint training setup.",
)
argparser.add_argument(
    "--exclude_prediction_date",
    action="store_true",
    help="Whether to exclude the prediction date. Should match checkpoint training setup.",
)
argparser.add_argument(
    "--exclude_prediction_sensors",
    action="store_true",
    help="Whether to exclude observational sensors in prediction date.",
)
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_train_balanced_tiny.json",
    help="Config name for evaluation. Options are stored in configs/finetune/",
)
argparser.add_argument(
    "--run_id",
    type=str,
    default="default_run",
    help="Identifier used to store results and model checkpoint.",
)
argparser.add_argument(
    "--model_type",
    type=str,
    default="rf",
    choices=["rf", "svr", "mlp"],
    help="Type of model to train: rf, svr, or mlp.",
)
argparser.add_argument(
    "--h5pys_only",
    action="store_true",
    help="Where to only use h5pys (faster, but need to be already stored in this format)",
)
argparser.add_argument(
    "--dataset_subset_size",
    type=int,
    default=0
)
argparser.add_argument(
    "--bagging",
    action="store_true",
)
args = argparser.parse_args().__dict__


id = args["run_id"]
with (Path("configs") / Path("finetune") / Path(args["eval_config_name"])).open("r") as f:
    config = json.load(f)

# we use the normalization values for missing data imputation so we load it independently
normalizing_dict = Dataset.load_normalization_values(path=config_dir / NORMALIZATION_DICT_FILENAME)

rf = LandsatEvalSklearn(
    normalization="std",
    exclude_prediction_date=args["exclude_prediction_date"],
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    exclude_prediction_sensors=args["exclude_prediction_sensors"],
    resample=False,
    eval_config=config,
    model_type=args["model_type"],
    h5pys_only=args["h5pys_only"],
    normalizing_dict=normalizing_dict,
    bagging=args["bagging"],
)
rf.fit_sklearn(id=args["run_id"], save_results=True, dataset_subset_size=args["dataset_subset_size"])
