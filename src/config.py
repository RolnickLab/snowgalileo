import random
from typing import Dict, List, Union

BASE_GSD_HIGH_RES = 10
BASE_GSD_MED_RES = 300
BASE_GSD_LOW_RES = 500
DEFAULT_SEED = 42


def get_random_config(
    model_size: str = "tiny",
):
    config: Dict[str, Dict] = {"training": {}, "model": {}}

    ### MODELS ###
    models = {
        "tiny": {
            "embedding_size": 128,
            "depth": 4,
            "num_heads": 8,
        },
        "vitb-tiny": {
            "embedding_size": 192,
            "depth": 12,
            "num_heads": 3,
        },
        "base": {
            "embedding_size": 768,
            "depth": 12,
            "num_heads": 12,
        },
    }
    config["model"]["encoder"] = models[model_size]
    config["model"]["encoder"]["mlp_ratio"] = 4
    config["model"]["encoder"]["max_sequence_length"] = 24
    config["model"]["encoder"]["freeze_projections"] = False
    config["model"]["encoder"]["drop_path"] = 0.1
    config["model"]["decoder"] = {}

    config["model"]["decoder"]["depth"] = random.choice([3, 4, 5])
    if config["model"]["encoder"]["embedding_size"] == 128:
        config["model"]["decoder"]["embedding_size"] = 128
        config["training"]["patch_sizes_high_res"] = ([8],)
        config["training"]["patch_sizes_med_res"] = ([1],)
        config["training"]["patch_sizes_low_res"] = ([1],)
    elif config["model"]["encoder"]["embedding_size"] == 192:
        config["model"]["decoder"]["embedding_size"] = 192
        config["training"]["patch_sizes_high_res"] = ([8],)
        config["training"]["patch_sizes_med_res"] = ([1],)
        config["training"]["patch_sizes_low_res"] = ([1],)
    elif config["model"]["encoder"]["embedding_size"] == 768:
        config["model"]["decoder"]["embedding_size"] = random.choice([128, 256, 512])
        config["training"]["patch_sizes_high_res"] = ([8],)
        config["training"]["patch_sizes_med_res"] = ([1],)
        config["training"]["patch_sizes_low_res"] = ([1],)
    else:
        raise ValueError(
            f"encoder embedding size didn't match options: {config['model']['encoder']['embedding_size']}"
        )

    config["model"]["decoder"]["num_heads"] = 4
    config["model"]["decoder"]["mlp_ratio"] = 4
    config["model"]["decoder"]["max_sequence_length"] = 24
    config["model"]["decoder"]["learnable_channel_embeddings"] = random.choice([True, False])

    config["training"]["max_lr"] = random.choice([1e-3, 2e-3, 3e-3])

    ### OPTIMIZATION ###
    config["training"]["num_epochs"] = 300
    config["training"]["batch_size"] = 16
    config["training"]["effective_batch_size"] = 512
    config["training"]["warmup_epochs"] = 0.1
    config["training"]["final_lr"] = 1e-6
    config["training"]["random_masking"] = random.choice(["full", "half", "none"])
    config["training"]["unmasking_channels_combo"] = random.choice(["shapes", "all"])
    weight_decay = random.choice([0.01, 0.02])
    config["training"]["weight_decay"] = weight_decay
    config["training"]["grad_clip"] = True
    config["training"]["betas"] = [0.9, 0.999]
    config["training"]["ema"] = [0.996, 1.0]

    ### DATA and MASKING ###
    config["training"]["shape_time_combinations"] = [
        {"size": 4, "timesteps": 12},
        {"size": 5, "timesteps": 6},
        {"size": 6, "timesteps": 4},
        {"size": 7, "timesteps": 3},
        {"size": 9, "timesteps": 3},
        {"size": 12, "timesteps": 3},
    ]
    config["training"]["masking_probabilities"] = [
        0.3,
        0.3,
        0.3,
        0.3,
        0.8,
        0.8,
        0.6,
        0.6,
        0.6,
        0.6,
        0.3,
        0.3,
        0.3,
        0.3,
        0.5,
        0.2,
        0.2,
    ]
    config["training"]["augmentation"] = {"flip+rotate": True}
    config["training"]["encode_ratio"] = 0.1
    config["training"]["decode_ratio"] = 0.8
    config["training"]["max_unmasking_channels"] = 2
    encoder_depth = config["model"]["encoder"]["depth"]
    possible_exit_depths: List[Union[str, int]] = [0, encoder_depth // 2, encoder_depth]
    config["training"]["target_exit_after"] = random.choice(possible_exit_depths)

    ### LOSS ###
    config["training"]["loss_type"] = "patch_disc"
    loss_name = "PD"  # for the run_name
    config["training"]["tau"] = 0.1
    config["training"]["pred2unit"] = False
    config["training"]["loss_mask_other_samples"] = False

    # norm
    config["training"]["normalization"] = "std"

    ### LOGGING ###
    config["training"]["eval_eurosat_every_n_epochs"] = 10

    run_name = f"{model_size}_{config['training']}_DecEmb:{config['model']['decoder']['learnable_channel_embeddings']}_Loss:{loss_name}_LRs:{config['training']['max_lr']}:{config['training']['weight_decay']}"
    return config, run_name
