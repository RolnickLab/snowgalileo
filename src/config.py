import random
from typing import Dict

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
        config["training"]["patch_sizes_high_res"] = 10
        config["training"]["patch_sizes_med_res"] = 1
        config["training"]["patch_sizes_low_res"] = 1
    elif config["model"]["encoder"]["embedding_size"] == 192:
        config["model"]["decoder"]["embedding_size"] = 192
        config["training"]["patch_sizes_high_res"] = 10
        config["training"]["patch_sizes_med_res"] = 1
        config["training"]["patch_sizes_low_res"] = 1
    elif config["model"]["encoder"]["embedding_size"] == 768:
        config["model"]["decoder"]["embedding_size"] = random.choice([128, 256, 512])
        config["training"]["patch_sizes_high_res"] = 10
        config["training"]["patch_sizes_med_res"] = 1
        config["training"]["patch_sizes_low_res"] = 1
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
    weight_decay = random.choice([0.01, 0.02])
    config["training"]["weight_decay"] = weight_decay
    config["training"]["grad_clip"] = True
    config["training"]["betas"] = [0.9, 0.999]

    ### DATA and MASKING ###
    config["training"]["augmentation"] = {"flip+rotate": True}
    config["training"]["encode_ratio"] = 0.1
    config["training"]["decode_ratio"] = 0.8

    ### LOSS ###
    config["training"]["loss_type"] = "patch_disc"
    loss_name = "PD"  # for the run_name

    # norm
    config["training"]["normalization"] = "std"

    run_name = f"{model_size}_{config['training']}_DecEmb:{config['model']['decoder']['learnable_channel_embeddings']}_Loss:{loss_name}_LRs:{config['training']['max_lr']}:{config['training']['weight_decay']}"
    return config, run_name
