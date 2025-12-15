import argparse
import json
from pathlib import Path

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset
from src.eval.landsat_baselines import LandsatEvalSklearn
from src.utils import config_dir, seed_everything

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
    default="landsat_eval_1_99_test.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
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
args = argparser.parse_args().__dict__


id = args["run_id"]
with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["eval_config_name"])).open(
    "r"
) as f:
    config = json.load(f)

# we use the normalization values for missing data imputation so we load it independently
normalizing_dict = Dataset.load_normalization_values(path=config_dir / NORMALIZATION_DICT_FILENAME)

rf = LandsatEvalSklearn(
    normalization="std",
    exclude_prediction_date=False,
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    resample=False,
    eval_config=config,
    model_type=args["model_type"],
    h5pys_only=args["h5pys_only"],
    normalizing_dict=normalizing_dict,
)
rf.fit_sklearn(id=args["run_id"], save_results=True)
