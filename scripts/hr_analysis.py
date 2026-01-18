import argparse
import json
from pathlib import Path

import pandas as pd
import psutil
import torch

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER, RESULTS_FOLDER
from src.eval.hr_eval import HRMetaDataset
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
    RESULTS_FOLDER / f"evaluation_results_{args['results_csv_name']}_with_hr.csv"
)
output_results_csv_path.touch(exist_ok=True)

tif_data_path = DATA_FOLDER / data_config["input_tif_folder"] / "test"

hr_dataset = HRMetaDataset(data_folder=tif_data_path)

df = pd.read_csv(input_results_csv_path)
all_files = df["filename"].tolist()

num_hr_days = []
last_hr_day = []
num_s1_days = []
num_s2_days = []
num_landsat_days = []
last_s1_day = []
last_s2_day = []
last_landsat_day = []
filenames = []

for i in all_files:
    tif_path = Path(tif_data_path / i)
    hr_dict, filename = hr_dataset.return_hr_from_filename(i)
    num_hr_days.append(hr_dict["num_hr_days"])
    last_hr_day.append(hr_dict["last_hr_day"])
    num_s1_days.append(hr_dict["num_s1_days"])
    num_s2_days.append(hr_dict["num_s2_days"])
    num_landsat_days.append(hr_dict["num_landsat_days"])
    last_s1_day.append(hr_dict["last_s1_day"])
    last_s2_day.append(hr_dict["last_s2_day"])
    last_landsat_day.append(hr_dict["last_landsat_day"])
    filenames.append(filename)

assert filenames == all_files, "filenames must match!"

df["num_hr_days"] = num_hr_days
df["last_hr_day"] = last_hr_day
df["num_s1_days"] = num_s1_days
df["num_s2_days"] = num_s2_days
df["num_landsat_days"] = num_landsat_days
df["last_s1_day"] = last_s1_day
df["last_s2_day"] = last_s2_day
df["last_landsat_day"] = last_landsat_day

df.to_csv(output_results_csv_path, index=False)
