from src.data.config import DATA_FOLDER
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import DEFAULT_SEED
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_folder", type=str, default="landsat_eval_tifs/patches_UTM_5_95_cropped")
argparser.add_argument("--mask_folder", type=str, default="landsat_eval_masks/all/patches_UTM_5_95_subset")
args = argparser.parse_args().__dict__

input_path = Path(DATA_FOLDER / args["input_folder"])
mask_path = Path(DATA_FOLDER / args["mask_folder"])

assert len(list(input_path.glob("*.tif"))) == len(list(mask_path.glob("*.tif")))

def create_train_test_split(input_path, mask_path, test_size=0.2, random_state=DEFAULT_SEED):

    # Make sure input_files and mask_files are properly matched
    # both should contain the same filenames in corresponding order
    input_files = sorted(Path(input_path).glob("*.tif"))
    mask_files = sorted(Path(mask_path).glob("*.tif"))

    assert all(f.stem == m.stem for f, m in zip(input_files, mask_files)), "Input and mask files not aligned!"
    assert len(input_files) == len(mask_files), (
        "Input and mask directories must have the same number of files."
    )

    # Pair them together before splitting
    pairs = list(zip(input_files, mask_files))

    train_pairs, test_pairs = train_test_split(
        pairs, test_size=test_size, random_state=random_state
    )

    # Create directories
    (input_path / "train").mkdir(parents=True, exist_ok=True)
    (input_path / "test").mkdir(parents=True, exist_ok=True)
    (mask_path / "train").mkdir(parents=True, exist_ok=True)
    (mask_path / "test").mkdir(parents=True, exist_ok=True)

    # Move files
    for input_file, mask_file in train_pairs:
        input_file.rename(input_path / "train" / input_file.name)
        mask_file.rename(mask_path / "train" / mask_file.name)

    for input_file, mask_file in test_pairs:
        input_file.rename(input_path / "test" / input_file.name)
        mask_file.rename(mask_path / "test" / mask_file.name)

    return train_pairs, test_pairs

if __name__ == "__main__":
    train_pairs, test_pairs = create_train_test_split(input_path, mask_path)
    print(f"Train input files: {len(train_pairs)}")
    print(f"Test input files: {len(test_pairs)}")
