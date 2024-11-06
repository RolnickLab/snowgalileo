# type: ignore

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    config_filepath = Path("../config/mae")
    assert config_filepath.exists()

    template = {
        "training": {
            "patch_sizes": [1, 2, 3, 4, 5, 6, 7, 8],
            "conditioner_mode": "no_cond",
            "max_lr": 0.002,
            "num_epochs": 200,
            "batch_size": 128,
            "effective_batch_size": 512,
            "warmup_epochs": 30,
            "final_lr": 1e-06,
            "conditioner_multiplier": 0.1,
            "weight_decay": 0.02,
            "conditioner_weight_decay": 0.02,
            "grad_clip": True,
            "betas": [0.9, 0.999],
            "ema": [0.996, 1.0],
            "shape_time_combinations": [
                {"size": 4, "timesteps": 12},
                {"size": 5, "timesteps": 6},
                {"size": 6, "timesteps": 4},
                {"size": 7, "timesteps": 3},
                {"size": 9, "timesteps": 3},
                {"size": 12, "timesteps": 3},
            ],
            "masking_probabilities": [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "augmentation": {"flip+rotate": True},
            "encode_ratio": 0.1,
            "decode_ratio": 0.5,
            "max_unmasking_channels": 17,
            "target_condition": False,
            "loss_type": "MAE",
            "tau": 0.1,
            "pred2unit": False,
            "loss_mask_other_samples": False,
            "normalization": "std",
            "eval_eurosat_every_n_epochs": 10,
            "random_masking": "full",
            "unmasking_channels_combo": "all",
        },
        "model": {
            "encoder": {
                "embedding_size": 192,
                "depth": 12,
                "num_heads": 3,
                "mlp_ratio": 4,
                "max_sequence_length": 24,
                "freeze_projections": False,
                "drop_path": 0.1,
                "max_patch_size": 8,
            },
            "decoder": {
                "depth": 4,
                "num_heads": 3,
                "mlp_ratio": 4,
                "max_sequence_length": 24,
                "learnable_channel_embeddings": True,
                "max_patch_size": 8,
                "embedding_size": 192,
            },
        },
    }

    token_exit_cfg = {
        "S1": 12,
        "S2_RGB": 12,
        "S2_Red_Edge": 12,
        "S2_NIR_10m": 12,
        "S2_NIR_20m": 12,
        "S2_SWIR": 12,
        "NDVI": 6,
        "ERA5": 6,
        "TC": 6,
        "VIIRS": 12,
        "SRTM": 6,
        "DW": 0,
        "WC": 0,
        "LS": 0,
        "location": 12,
        "DW_static": 0,
        "WC_static": 0,
    }

    existing_configs = [int(x.stem) for x in config_filepath.glob("*.json")]
    cur_config = max(existing_configs) + 1
    print(f"Starting from config number {cur_config}")

    config_values = {
        "id": [],
        "target_depth": [],
        "loss_function": [],
        "masking": [],
        "decoder_context": [],
    }
    for depth in ["half", "full", "varied"]:
        for loss in ["mse", "LatentMIM", "ours"]:
            for masking in [
                "random",
                "space+time",
                "random+space+time",
            ]:  #  "space+time, channel shapes", "random+space+time, channel shapes"]:
                for decoder_context in ["all", "decoder_and_encoder"]:
                    working_config = deepcopy(template)

                    if depth == "half":
                        working_config["training"]["target_exit_after"] = 6
                    elif depth == "full":
                        working_config["training"]["target_exit_after"] = 12
                    elif depth == "varied":
                        working_config["training"]["token_exit_cfg"] = deepcopy(token_exit_cfg)

                    if loss == "mse":
                        working_config["training"]["loss_type"] = "mse"
                    elif loss == "LatentMIM":
                        working_config["training"]["loss_type"] = "patch_disc"
                        working_config["training"]["loss_mask_other_samples"] = True
                    elif loss == "ours":
                        working_config["training"]["loss_type"] = "patch_disc"
                        working_config["training"]["loss_mask_other_samples"] = False

                    if masking == "random":
                        working_config["training"]["random_masking"] = "full"
                        working_config["training"]["unmasking_channels_combo"] = "all"
                    elif masking == "space+time":
                        working_config["training"]["random_masking"] = "none"
                        working_config["training"]["unmasking_channels_combo"] = "all"
                    elif masking == "random+space+time":
                        working_config["training"]["random_masking"] = "half"
                        working_config["training"]["unmasking_channels_combo"] = "all"
                    elif masking == "space+time, channel shapes":
                        working_config["training"]["random_masking"] = "none"
                        working_config["training"]["unmasking_channels_combo"] = "shapes"
                    elif masking == "random+space+time, channel shapes":
                        working_config["training"]["random_masking"] = "half"
                        working_config["training"]["unmasking_channels_combo"] = "shapes"

                    working_config["training"]["target_masking"] = decoder_context

                    config_values["id"].append(cur_config)
                    config_values["target_depth"].append(depth)
                    config_values["loss_function"].append(loss)
                    config_values["masking"].append(masking)
                    config_values["decoder_context"].append(decoder_context)

                    with (config_filepath / f"{cur_config}.json").open("w") as f:
                        json.dump(working_config, f, indent=2)
                    print(f"saved {cur_config}.json")
                    cur_config += 1

    config_df = pd.DataFrame(data=config_values)
    config_df.to_csv("without_shapes.csv")
