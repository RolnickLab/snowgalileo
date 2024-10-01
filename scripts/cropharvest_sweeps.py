import csv
from itertools import combinations, product
from pathlib import Path
from typing import List

import torch

from src.data import Dataset, Normalizer
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.eval import BinaryCropHarvestEval
from src.flexipresto import Encoder
from src.masking import MASKING_MODES, STR2DICT
from src.utils import config_dir, device

SHAPES = list(STR2DICT.keys())


def append_to_csv(file_path, input_list):
    with open(file_path, "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(input_list)


def generate_combinations():
    all_combinations = []
    for r in range(1, 5):
        shape_combos = combinations(SHAPES, r)

        for shape_combo in shape_combos:
            mode_lists = [STR2DICT[shape] for shape in shape_combo]
            mode_combos = product(*mode_lists)
            for mode_combo in mode_combos:
                all_combinations.append(list(mode_combo))

    return all_combinations


def update_output_channels(task: BinaryCropHarvestEval, new_output_channels: List[str]):
    if isinstance(new_output_channels, str):
        new_output_channels = [new_output_channels]
    output_channels = [0] * len(MASKING_MODES)
    for i, val in enumerate(MASKING_MODES):
        if val[1] in new_output_channels:
            output_channels[i] = 1
    device = task.condition["output_channels"].device  # type: ignore
    task.condition["output_channels"] = torch.Tensor(output_channels).to(device)


if __name__ == "__main__":
    model_path = "data/outputs/ezoy5r08"
    savefile_path = "ezoy5r08_cropharvest_sweep.csv"
    model = Encoder.load_from_folder(Path(model_path)).to(device)
    encoder_depth = len(model.blocks)
    normalizing_dict = Dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)

    output_channel_combinations = generate_combinations()

    append_to_csv(
        file_path=savefile_path,
        input_list=["country", "output_channels", "KNN@5", "KNN@5_c", "LR", "LR_c"],
    )

    task = BinaryCropHarvestEval(
        country="Togo", normalizer=normalizer, do_condition=True, eval_mode="val"
    )
    for channel_combo in output_channel_combinations:
        print(f"Running for {channel_combo}")
        update_output_channels(task, channel_combo)
        output = task.evaluate_model_on_task(
            model, model_modes=["Logistic Regression", "KNNat5 Classifier"]
        )

        # retrieve the appropriate keys
        output_keys = list(output.keys())
        lr_keys = [
            k for k in output_keys if "Regression" in k and "f1" in k and not k.endswith("_c")
        ]
        assert len(lr_keys) == 1
        lr_key = lr_keys[0]
        lr_c_key = [
            k for k in output_keys if "Regression" in k and "f1" in k and k.endswith("_c")
        ][0]
        k_key = [k for k in output_keys if "KNNat5" in k and "f1" in k and not k.endswith("_c")][0]
        k_c_key = [k for k in output_keys if "KNNat5" in k and "f1" in k and k.endswith("_c")][0]
        # save and print
        full_row = [
            "Togo",
            channel_combo,
            output[k_key],
            output[k_c_key],
            output[lr_key],
            output[lr_c_key],
        ]
        print(full_row)
        append_to_csv(file_path=savefile_path, input_list=full_row)
