import hashlib
import json
from pathlib import Path

from tqdm import tqdm

SPLIT_TO_HEX = {
    0: ["0", "1", "2", "3"],
    1: ["4", "5", "6", "7"],
    2: ["8", "9", "a", "b"],
    3: ["c", "d", "e", "f"],
}


def assign_splits_rslearn(dataset_path: Path, split_id: int):
    """
    This function is used to make the split.json used by rslearn
    when training the satlas models. It is not used by the presto codebase
    """
    assert dataset_path.exists()
    example_id_to_label = {}
    fnames = list(dataset_path.glob("windows/*/*/layers/label/data.geojson"))
    for fname in tqdm(fnames):
        with open(fname) as f:
            category = json.load(f)["features"][0]["properties"]["new_label"]
            if category in ["unknown", "unlabeled", "human", "natural"]:
                continue
            group = fname.parents[3]
            example_id = fname.parents[2]
            example_id_to_label[(group, example_id)] = category

    split_data = {}
    for group, example_id in example_id_to_label.keys():
        if group in ["peru3", "peru3_flagged_in_peru", "peru_interesting"]:
            is_val = hashlib.sha256(example_id.encode()).hexdigest()[0] in SPLIT_TO_HEX[split_id]
            print(is_val, example_id, hashlib.sha256(example_id.encode()).hexdigest()[0])
            if is_val:
                split_data[example_id] = "val"
            else:
                split_data[example_id] = "train"
        elif group in ["nadia2", "nadia3", "brazil_interesting"]:
            split_data[example_id] = "train"

    # now apply these splits to the metadata tags
    window_metadatas = list(dataset_path.glob("windows/*/*/metadata.json"))

    for metadata_fname in tqdm(window_metadatas):
        example_id = example_id.parents[0]
        if example_id not in split_data:
            continue
        with open(metadata_fname) as f:
            metadata = json.load(f)
        if "options" not in metadata or metadata["options"] is None:
            metadata["options"] = {}
        metadata["options"]["split"] = split_data[example_id]
        with open(metadata_fname, "w") as f:
            json.dump(metadata, f)


if __name__ == "__main__":
    assign_splits_rslearn(Path("../../../rslearn_amazon_conservation_closetime"), 0)
