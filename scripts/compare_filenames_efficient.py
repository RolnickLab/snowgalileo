import os
import argparse
from pathlib import Path
from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="landsat_eval_tifs/patches_UTM_5_95_cropped/train")
argparser.add_argument("--output_folder", type=str, default="landsat_eval_masks/all/patches_UTM_5_95_subset/train")

args = argparser.parse_args().__dict__

input_path = Path(DATA_FOLDER / args["input_folder"])
output_path = Path(DATA_FOLDER / args["output_folder"])

def compare_filenames(folder1, folder2):
    # Efficiently list filenames (not full paths)
    files1 = {entry.name for entry in os.scandir(folder1) if entry.is_file()}
    files2 = {entry.name for entry in os.scandir(folder2) if entry.is_file()}

    # Compute matches and mismatches
    matching = files1 & files2
    only_in_folder1 = files1 - files2
    only_in_folder2 = files2 - files1

    print(f"Matching files: {len(matching)}")
    print(f"Non-matching files in {folder1}: {len(only_in_folder1)}")
    print(f"Non-matching files in {folder2}: {len(only_in_folder2)}")
    print(f"Total non-matching (unique across both): {len(only_in_folder1) + len(only_in_folder2)}")

    # print the first 10 filenames that are not matching
    print(f"First 10 non-matching files in {folder1}: {list(only_in_folder1)[:10]}")
    print(f"First 10 non-matching files in {folder2}: {list(only_in_folder2)[:10]}")

if __name__ == "__main__":
    compare_filenames(input_path, output_path)
