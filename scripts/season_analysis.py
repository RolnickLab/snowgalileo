import argparse
import json
from pathlib import Path

import pandas as pd
import psutil
import torch

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER, RESULTS_FOLDER
from src.eval.utils import extract_season_from_filename
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_test_rockies_tiny.json",
    help="Config name for evaluation. Options are stored in src/eval/eval_configs/",
)
argparser.add_argument(
    "--results_csv_name",
    type=str,
    default="fsc_test_rockies_tiny",
)
args = argparser.parse_args().__dict__

with (Path("src") / Path("eval") / Path("eval_configs") / Path(args["eval_config_name"])).open(
    "r"
) as f:
    eval_config = json.load(f)
data_config = eval_config["data"]

input_results_csv_path = RESULTS_FOLDER / f"evaluation_results_{args['results_csv_name']}.csv"
output_results_csv_path = (
    RESULTS_FOLDER / f"evaluation_results_{args['results_csv_name']}_with_seasons.csv"
)
output_results_csv_path.touch(exist_ok=True)

tif_data_path = DATA_FOLDER / data_config["input_tif_folder"] / "test"

df = pd.read_csv(input_results_csv_path)
all_files = df["filename"].tolist()

seasons = []

for i in all_files:
    season = extract_season_from_filename(i)
    seasons.append(season)

df["season"] = seasons

df.to_csv(output_results_csv_path, index=False)
