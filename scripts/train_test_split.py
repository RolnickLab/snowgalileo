from src.data.config import DATA_FOLDER
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import DEFAULT_SEED
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="landsat_eval_tifs/100m_tif_global_cropped")
argparser.add_argument("--mask_folder", type=str, default="landsat_eval_masks/all/100m_mask_global_subset")
argparser.add_argument("--holdout_lon_min", type=float, default=None)
argparser.add_argument("--holdout_lon_max", type=float, default=None)

args = argparser.parse_args().__dict__

input_path = Path(DATA_FOLDER / args["input_folder"])
mask_path = Path(DATA_FOLDER / args["mask_folder"])

# assert that input and mask path contain the same number of files
assert len(list(input_path.glob("*.tif"))) == len(list(mask_path.glob("*.tif")))

# create train and test split using sklearn's train_test_split
def create_train_test_split(input_path, mask_path, test_size=0.2, random_state=DEFAULT_SEED):
    input_files = list(input_path.glob("*.tif"))
    mask_files = list(mask_path.glob("*.tif"))

    assert len(input_files) == len(mask_files), (
        "Input and mask directories must have the same number of files."
    )

    if args["holdout_lon_min"] is not None and args["holdout_lon_max"] is not None:
        # If holdout longitudes are specified, create a custom test set
        holdout_lon_min = args["holdout_lon_min"]
        holdout_lon_max = args["holdout_lon_max"]

        def is_in_holdout(file_path):
            try:
                lon_str = file_path.stem.split(".tif")[0].split("_")[4]
                lon = float(lon_str)
                return holdout_lon_min <= lon <= holdout_lon_max
            except (IndexError, ValueError):
                return False

        test_input = [f for f in input_files if is_in_holdout(f)]
        test_mask = [f for f in mask_files if is_in_holdout(f)]
        train_input = [f for f in input_files if f not in test_input]
        train_mask = [f for f in mask_files if f not in test_mask]

        print(f"Custom holdout longitudes: {holdout_lon_min} to {holdout_lon_max}")
        print(f"Number of training samples: {len(train_input)}")
        print(f"Number of testing samples: {len(test_input)}")
        print(f"Train ratio: {len(train_input) / len(input_files):.2f}")
        print(f"Test ratio: {len(test_input) / len(input_files):.2f}")

    else:
        # Split the files into train and test sets
        train_input, test_input, train_mask, test_mask = train_test_split(
            input_files, mask_files, test_size=test_size, random_state=random_state
        )

    # store them in train test split folders
    train_input = [str(file) for file in train_input]
    test_input = [str(file) for file in test_input]
    train_mask = [str(file) for file in train_mask]
    test_mask = [str(file) for file in test_mask]

    # Create directories if they do not exist
    (input_path / "train").mkdir(parents=True, exist_ok=True)
    (input_path / "test").mkdir(parents=True, exist_ok=True)
    (mask_path / "train").mkdir(parents=True, exist_ok=True)
    (mask_path / "test").mkdir(parents=True, exist_ok=True)

    # Move files to the respective directories
    for file in train_input:
        Path(file).rename(input_path / "train" / Path(file).name)
    for file in test_input:
        Path(file).rename(input_path / "test" / Path(file).name)
    for file in train_mask:
        Path(file).rename(mask_path / "train" / Path(file).name)
    for file in test_mask:
        Path(file).rename(mask_path / "test" / Path(file).name)

    return train_input, test_input, train_mask, test_mask


if __name__ == "__main__":
    train_input, test_input, train_mask, test_mask = create_train_test_split(input_path, mask_path)
    print(f"Train input files: {len(train_input)}")
    print(f"Test input files: {len(test_input)}")
    print(f"Train mask files: {len(train_mask)}")
    print(f"Test mask files: {len(test_mask)}")
