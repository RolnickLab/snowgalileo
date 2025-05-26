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

argparser = argparse.ArgumentParser()
argparser.add_argument("--h5py_folder", type=str, default="")
argparser.add_argument("--tifs_folder", type=str, default="")
argparser.add_argument("--download", dest="download", action="store_true")
argparser.add_argument("--h5pys_only", dest="h5pys_only", action="store_true")
argparser.add_argument("--estimate_from", type=int, default=1000)
argparser.add_argument("--plot_distributions", action="store_true")
argparser.add_argument("--assets_folder_name", type=str, default="assets")
argparser.set_defaults(download=False)
argparser.set_defaults(cache_in_ram=False)
args = argparser.parse_args().__dict__

if args["tifs_folder"] == "":
    tifs_folder = TIFS_FOLDER
else:
    tifs_folder = Path(DATA_FOLDER / args["tifs_folder"])

print("Loading dataset and dataloader")

dataset = Dataset(
    data_folder=tifs_folder,
    download=args["download"],
    h5py_folder=None,
    h5pys_only=args["h5pys_only"],
)

normalizing_dict = dataset.compute_running_stats()
print(normalizing_dict, flush=True)
