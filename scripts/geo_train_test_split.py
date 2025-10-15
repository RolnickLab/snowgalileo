from src.data.config import DATA_FOLDER
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import DEFAULT_SEED
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="landsat_eval_tifs/patches_UTM_5_95_cropped")
argparser.add_argument("--mask_folder", type=str, default="landsat_eval_masks/all/patches_UTM_5_95_subset")
argparser.add_argument("--train_region", type=str, default="random", choices=["random", "alps", "rockies", "himalayas", "northern_polar", "northern_hemisphere", "southern_hemisphere"])
argparser.add_argument("--test_region", type=str, default="random", choices=["random", "alps", "rockies", "himalayas", "northern_polar", "northern_hemisphere", "southern_hemisphere"])

args = argparser.parse_args().__dict__

input_path = Path(DATA_FOLDER / args["input_folder"])
mask_path = Path(DATA_FOLDER / args["mask_folder"])

train_lon_min, train_lon_max = None, None
test_lon_min, test_lon_max = None, None
train_lat_min, train_lat_max = None, None
test_lat_min, test_lat_max = None, None

if args["train_region"] == "northern_hemisphere" or args["test_region"] == "northern_hemisphere" or args["train_region"] == "southern_hemisphere" or args["test_region"] == "southern_hemisphere":
    raise NotImplementedError("Current data doesn't support northern / southern hemisphere splits.")

# simple latitude / longitude-based splits
# NOTE: these assumptions will need to adjust when we use data beyond landsat_eval
if args["train_region"] != "random":
    train_lat_min, train_lat_max = {
        "northern_hemisphere": (0, 90),
        "southern_hemisphere": (-90, 0),
        "northern_polar": (60, 90),
        "himalayas": (25, 40), # to avoid overlap with other ranges in the dataset
    }.get(args["train_region"], (None, None))
    train_lon_min, train_lon_max = {
        "alps": (5, 20),
        "rockies": (-135, -105),
        "himalayas": (70, 100),
    }.get(args["train_region"], (None, None))

if args["test_region"] != "random":
    test_lat_min, test_lat_max = {
        "northern_hemisphere": (0, 90),
        "southern_hemisphere": (-90, 0),
        "northern_polar": (60, 90),
        "himalayas": (25, 40),
    }.get(args["test_region"], (None, None))
    test_lon_min, test_lon_max = {
        "alps": (5, 20),
        "rockies": (-135, -105),
        "himalayas": (70, 100),
    }.get(args["test_region"], (None, None))

# assert that input and mask path contain the same number of files
assert len(list(input_path.glob("*.tif"))) == len(list(mask_path.glob("*.tif")))

print(f"Train region: {args['train_region']} (lat: {train_lat_min} to {train_lat_max}, lon: {train_lon_min} to {train_lon_max})")
print(f"Test region: {args['test_region']} (lat: {test_lat_min} to {test_lat_max}, lon: {test_lon_min} to {test_lon_max})")

def is_in_region(file_path, lat_min, lat_max, lon_min, lon_max):
    try:
        lat_str = file_path.stem.split(".tif")[0].split("_")[3]
        lon_str = file_path.stem.split(".tif")[0].split("_")[4]
        lat = float(lat_str)
        lon = float(lon_str)
        lat_check = (lat_min is None or lat >= lat_min) and (lat_max is None or lat <= lat_max)
        lon_check = (lon_min is None or lon >= lon_min) and (lon_max is None or lon <= lon_max)
        return lat_check and lon_check
    except (IndexError, ValueError):
        return False

def create_train_test_split(input_path, mask_path, test_size=0.2, random_state=DEFAULT_SEED):
    input_files = sorted(Path(input_path).glob("*.tif"))
    mask_files = sorted(Path(mask_path).glob("*.tif"))

    assert all(f.stem == m.stem for f, m in zip(input_files, mask_files)), "Input and mask files not aligned!"
    assert len(input_files) == len(mask_files), (
        "Input and mask directories must have the same number of files."
    )

    # If specific regions are defined, filter files accordingly
    if args["train_region"] != "random":
        train_input_files = [f for f in input_files if is_in_region(f, train_lat_min, train_lat_max, train_lon_min, train_lon_max)]
        train_mask_files = [f for f in mask_files if is_in_region(f, train_lat_min, train_lat_max, train_lon_min, train_lon_max)]
        assert all(f.stem == m.stem for f, m in zip(train_input_files, train_mask_files)), "Input and mask files not aligned!"

    if args["test_region"] != "random":
        test_input_files = [f for f in input_files if is_in_region(f, test_lat_min, test_lat_max, test_lon_min, test_lon_max)]
        test_mask_files = [f for f in mask_files if is_in_region(f, test_lat_min, test_lat_max, test_lon_min, test_lon_max)]
        assert all(f.stem == m.stem for f, m in zip(test_input_files, test_mask_files)), "Input and mask files not aligned!"

    if args["test_region"] != "random":
        # Ensure no overlap between train and test sets
        assert set(train_input_files).isdisjoint(set(test_input_files)), "Train and test regions overlap."

        test_input = [f for f in input_files if is_in_region(f, test_lat_min, test_lat_max, test_lon_min, test_lon_max)]
        test_mask = [f for f in mask_files if is_in_region(f, test_lat_min, test_lat_max, test_lon_min, test_lon_max)]
        test_id = args["test_region"]

    if args["train_region"] != "random":
        train_input = [f for f in input_files if is_in_region(f, train_lat_min, train_lat_max, train_lon_min, train_lon_max)]
        train_mask = [f for f in mask_files if is_in_region(f, train_lat_min, train_lat_max, train_lon_min, train_lon_max)]
        train_id = args["train_region"]

    elif args["train_region"] == "random" and args["test_region"] != "random":
        train_input = [f for f in input_files if f not in test_input]
        train_mask = [f for f in mask_files if f not in test_mask]
        train_id = f"holdout_{test_id}"

    elif args["train_region"] != "random" and args["test_region"] == "random":
        test_input = [f for f in input_files if f not in train_input]
        test_mask = [f for f in mask_files if f not in train_mask]
        test_id = f"holdout_{train_id}"

    else:
        raise ValueError("At least one of train_region or test_region must be specified as non-random.")

    print(f"Number of training samples: {len(train_input)}")
    print(f"Number of testing samples: {len(test_input)}")
    print(f"Train ratio: {len(train_input) / len(input_files):.2f}")
    print(f"Test ratio: {len(test_input) / len(input_files):.2f}")

    # store them in train test split folders
    train_input = [str(file) for file in train_input]
    test_input = [str(file) for file in test_input]
    train_mask = [str(file) for file in train_mask]
    test_mask = [str(file) for file in test_mask]

    # Create directories if they do not exist
    (input_path / f"train_{train_id}").mkdir(parents=True, exist_ok=True)
    (input_path / f"test_{test_id}").mkdir(parents=True, exist_ok=True)
    (mask_path / f"train_{train_id}").mkdir(parents=True, exist_ok=True)
    (mask_path / f"test_{test_id}").mkdir(parents=True, exist_ok=True)

    # Copy files to respective directories
    for file in train_input:
        Path(file).copy(input_path / f"train_{train_id}" / Path(file).name)
    for file in test_input:
        Path(file).copy(input_path / f"test_{test_id}" / Path(file).name)
    for file in train_mask:
        Path(file).copy(mask_path / f"train_{train_id}" / Path(file).name)
    for file in test_mask:
        Path(file).copy(mask_path / f"test_{test_id}" / Path(file).name)

    return train_input, test_input, train_mask, test_mask

if __name__ == "__main__":
    train_input, test_input, train_mask, test_mask = create_train_test_split(input_path, mask_path)
    print(f"Train input files: {len(train_input)}")
    print(f"Test input files: {len(test_input)}")
    print(f"Train mask files: {len(train_mask)}")
    print(f"Test mask files: {len(test_mask)}")