import argparse
import random
from shutil import copyfile

from src.data.config import DATA_FOLDER

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--src_folder_input", type=str, default="landsat_eval_tifs/patches_UTM_1_99_test_cropped/test/"
)
argparser.add_argument(
    "--dest_folder_input",
    type=str,
    default="landsat_eval_tifs/patches_UTM_1_99_test_cropped/visualize/",
)
argparser.add_argument(
    "--src_folder_label", type=str, default="landsat_eval_masks/all/patches_UTM_1_99_test/test/"
)
argparser.add_argument(
    "--dest_folder_label",
    type=str,
    default="landsat_eval_masks/all/patches_UTM_1_99_test/visualize/",
)
argparser.add_argument("--file_limit", type=int, default=10)

args = argparser.parse_args().__dict__

src_folder_input = DATA_FOLDER / args["src_folder_input"]
dest_folder_input = DATA_FOLDER / args["dest_folder_input"]
src_folder_label = DATA_FOLDER / args["src_folder_label"]
dest_folder_label = DATA_FOLDER / args["dest_folder_label"]

dest_folder_input.mkdir(parents=True, exist_ok=True)
dest_folder_label.mkdir(parents=True, exist_ok=True)

# copy a limited number of files from src_folder to dest_folder. The same filenames should be copied from src_folder_label to dest_folder_label
# the files should be chosen randomly

random.seed(42)
files = list(src_folder_input.glob("*.tif"))
random.shuffle(files)
file_count = 0
for file in files:
    if file_count >= args["file_limit"]:
        break
    copyfile(file, dest_folder_input / file.name)
    label_file = src_folder_label / file.name
    if label_file.exists():
        copyfile(label_file, dest_folder_label / label_file.name)
    else:
        print(f"Label file {label_file} does not exist.")
    file_count += 1

# check that dest_folder_input and dest_folder_label contain the same number of files and that the filenames are matching
assert len(list(dest_folder_input.glob("*.tif"))) == len(list(dest_folder_label.glob("*.tif")))
assert set(f.name for f in dest_folder_input.glob("*.tif")) == set(
    f.name for f in dest_folder_label.glob("*.tif")
)
print(
    f"Copied {file_count} files from {src_folder_input} to {dest_folder_input} and from {src_folder_label} to {dest_folder_label}."
)
