import os
import argparse
from src.data.config import DATA_FOLDER
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="landsat_eval_tifs/patches_UTM_1_99_cropped")
argparser.add_argument("--output_folder", type=str, default="landsat_eval_masks/all/patches_UTM_1_99_subset")

args = argparser.parse_args().__dict__

input_path = Path(DATA_FOLDER / args["input_folder"])
output_path = Path(DATA_FOLDER / args["output_folder"])

# count the number of files in the input folder that have the same name as in the output folder
count = 0
non_count = 0
for file in os.listdir(input_path):
    stem = file.split(".")[0]
    # assert that the stem only occurs once in the input folder
    assert sum(1 for f in os.listdir(input_path) if f.startswith(stem)) == 1, f"File {stem} occurs multiple times in input folder"

    if stem in os.listdir(output_path):
        count += 1
    else:
        non_count += 1

print(f"Number of matching files: {count}")
print(f"Number of non-matching files: {non_count}")