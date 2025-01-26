import argparse

from src.data.config import DATA_FOLDER
from src.data.earthengine.utils import copy_files_with_partial_check

argparser = argparse.ArgumentParser()
argparser.add_argument("--src_folder", type=str, default="tifs3000")
argparser.add_argument("--dest_folder", type=str, default="tifs_all")

args = argparser.parse_args().__dict__

src_folder = DATA_FOLDER / args["src_folder"]
dest_folder = DATA_FOLDER / args["dest_folder"]

copy_files_with_partial_check(src_folder, dest_folder)
