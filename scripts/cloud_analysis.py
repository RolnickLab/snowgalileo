import argparse
import json
from pathlib import Path

import pandas as pd
import psutil
import torch

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER, RESULTS_FOLDER
from src.fsc.add_eval.cloud_eval import CloudMetaDataset
from src.utils import seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

torch.backends.cuda.matmul.allow_tf32 = True

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--eval_config_name",
    type=str,
    default="fsc_test_rockies_tiny.json",
    help="Config name for evaluation. Options are stored in configs/eval/",
)
argparser.add_argument(
    "--results_csv_name",
    type=str,
    default="fsc_test_rockies_tiny",
)
args = argparser.parse_args().__dict__

with (Path("configs") / Path("eval") / Path(args["eval_config_name"])).open("r") as f:
    eval_config = json.load(f)
data_config = eval_config["data"]

input_results_csv_path = RESULTS_FOLDER / f"evaluation_results_{args['results_csv_name']}.csv"
output_results_csv_path = (
    RESULTS_FOLDER / f"evaluation_results_{args['results_csv_name']}_with_clouds.csv"
)
output_results_csv_path.touch(exist_ok=True)

tif_data_path = DATA_FOLDER / data_config["input_tif_folder"] / "test"

cloud_dataset = CloudMetaDataset(data_folder=tif_data_path)

df = pd.read_csv(input_results_csv_path)
all_files = df["filename"].tolist()

last_clear_days = []
total_clear_days = []
total_cloudy_days = []
total_cloud_shadow_days = []
total_cirrus_days = []
lats = []
lons = []
total_days = []
filenames = []

for i in all_files:
    tif_path = Path(tif_data_path / i)
    cloud_state, filename = cloud_dataset.return_cloud_state_from_filename(i)
    last_clear_days.append(cloud_state["last_clear_day"])
    total_clear_days.append(cloud_state["total_clear_days"])
    total_cloudy_days.append(cloud_state["total_cloudy_days"])
    total_cloud_shadow_days.append(cloud_state["total_cloud_shadow_days"])
    total_cirrus_days.append(cloud_state["total_cirrus_days"])
    lats.append(cloud_state["lat"])
    lons.append(cloud_state["lon"])
    total_days.append(cloud_state["total_days"])
    filenames.append(filename)

assert filenames == all_files, "filenames must match!"

df["last_clear_day"] = last_clear_days
df["total_clear_days"] = total_clear_days
df["total_cloudy_days"] = total_cloudy_days
df["total_cloud_shadow_days"] = total_cloud_shadow_days
df["total_cirrus_days"] = total_cirrus_days

df.to_csv(output_results_csv_path, index=False)
