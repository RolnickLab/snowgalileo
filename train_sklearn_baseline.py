import argparse
import json
from pathlib import Path

from src.config import DEFAULT_SEED
from src.eval.landsat_baselines import LandsatEvalSklearn
from src.utils import seed_everything

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

args = argparser.parse_args().__dict__


id = args["run_id"]
with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["eval_config_name"])).open(
    "r"
) as f:
    config = json.load(f)

rf = LandsatEvalSklearn(
    normalization="std",
    exclude_prediction_date=False,
    exclude_prediction_high_res=args["exclude_prediction_high_res"],
    resample=False,
    eval_config=config,
    model_type=args["model_type"],
)
rf.fit_sklearn(id=args["run_id"], save_results=True)
