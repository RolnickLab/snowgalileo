import argparse
import os
from pathlib import Path

from src.config import DEFAULT_SEED
from src.data import Dataset
from src.data.config import DATA_FOLDER, EE_PROJECT, TIFS_FOLDER
from src.utils import (
    seed_everything,
)

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# we seed everything after we call get_random_config(), since
# we want this to differ between runs
seed_everything(DEFAULT_SEED)

argparser = argparse.ArgumentParser(
    description="Script for computing normalization values using Running Stats (should be executed once before training)"
)
argparser.add_argument("--h5py_folder", type=str, default="")
argparser.add_argument("--tifs_folder", type=str, default="")
argparser.add_argument("--h5pys_only", dest="h5pys_only", action="store_true")
args = argparser.parse_args().__dict__

if args["tifs_folder"] == "":
    tifs_folder = TIFS_FOLDER
else:
    tifs_folder = Path(DATA_FOLDER / args["tifs_folder"])

if args["h5py_folder"] != "":
    args["h5py_folder"] = Path(DATA_FOLDER / args["h5py_folder"])
else:
    args["h5py_folder"] = None

print("Loading dataset and dataloader")

dataset = Dataset(
    data_folder=tifs_folder,
    download=False,
    h5py_folder=args["h5py_folder"],
    h5pys_only=args["h5pys_only"],
)

normalizing_dict = dataset.compute_running_stats(sampled_n=len(dataset))
print(normalizing_dict, flush=True)
