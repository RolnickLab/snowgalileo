import random
from typing import Dict

BASE_GSD = 10
DEFAULT_SEED = 42


def get_random_config(model_size: str = "tiny"):
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
    config["model"]["encoder"]["drop_path"] = random.choice([0, 0.1])
    config["model"]["decoder"] = {}

    if config["model"]["encoder"]["embedding_size"] == 128:
        config["model"]["decoder"]["embedding_size"] = 128
        config["model"]["decoder"]["depth"] = random.choice([1, 2, 3])
        config["training"]["patch_sizes"] = [1, 2, 3, 4, 5, 6, 7, 8]
    elif config["model"]["encoder"]["embedding_size"] == 192:
        config["model"]["decoder"]["embedding_size"] = random.choice([128, 192])
        config["model"]["decoder"]["depth"] = random.choice([1, 2, 3, 4])
        config["training"]["patch_sizes"] = [1, 2, 3, 4, 5, 6, 7, 8]
    elif config["model"]["encoder"]["embedding_size"] == 768:
        config["model"]["decoder"]["embedding_size"] = random.choice([128, 256, 512])
        config["model"]["decoder"]["depth"] = random.choice([1, 2, 3, 4])
        config["training"]["patch_sizes"] = [6, 8, 10, 12, 14, 16]
    else:
        raise ValueError(
            f"encoder embedding size didn't match options: {config['model']['encoder']['embedding_size']}"
        )

    if config["model"]["decoder"]["embedding_size"] == 192:
        config["model"]["decoder"]["num_heads"] = random.choice([2, 3, 8])
    else:
        config["model"]["decoder"]["num_heads"] = random.choice([2, 8])
    config["model"]["decoder"]["mlp_ratio"] = 4
    config["model"]["decoder"]["max_sequence_length"] = 24
    config["model"]["decoder"]["learnable_channel_embeddings"] = random.choice([True, False])

    config["training"]["conditioner_mode"] = "lora"
    config["model"]["lora_generator"] = {}
    config["model"]["lora_generator"]["dim"] = random.choice([128, 256])
    config["model"]["lora_generator"]["rank"] = random.choice([12, 32, 64])
    config["model"]["lora_generator"]["do_input_condition"] = random.choice([True, False])
    config["training"]["max_lr"] = random.choice([5e-4, 8e-4, 1e-3])

    ### OPTIMIZATION ###
    config["training"]["num_epochs"] = 300
    config["training"]["batch_size"] = 16
    config["training"]["effective_batch_size"] = 512
    config["training"]["warmup_epochs"] = 0.1
    config["training"]["final_lr"] = 1e-6
    config["training"]["conditioner_multiplier"] = random.choice([0.1, 0.05])

    config["training"]["weight_decay"] = random.choice([0.01, 0.02, 0.05])
    config["training"]["conditioner_weight_decay"] = random.choice([0.01, 0.02, 0.05])
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
    config["training"]["unmasking_probabilities"] = [
        0.5,
        0.8,
        0.8,
        0.6,
        0.3,
        0.3,
        0.4,
        0.4,
        0.4,
        0.4,
        0.6,
        0.6,
        0.6,
        0.6,
        0.5,
        0.8,
        0.8,
    ]
    config["training"]["augmentation"] = {"flip+rotate": True}
    config["training"]["encode_ratio"] = random.choice([0.1, 0.2])
    config["training"]["decode_ratio"] = random.choice([0.5, 0.7, 0.8])
    config["training"]["target_exit_after"] = random.choice(
        range(config["model"]["encoder"]["depth"] + 1)
    )
    config["training"]["target_condition"] = random.choice([True, False])

    ### LOSS ###
    config["training"]["loss_type"] = "patch_disc"
    loss_name = "PD"  # for the run_name
    config["training"]["tau"] = random.choice([0.1, 0.2])
    config["training"]["pred2unit"] = random.choice([True, False])
    config["training"]["loss_mask_other_samples"] = False

    # norm
    config["training"]["normalization"] = "std"

    ### LOGGING ###
    config["training"]["eval_eurosat_every_n_epochs"] = 10

    run_name = f"{model_size}_{config['training']['conditioner_mode']}_DecEmb:{config['model']['decoder']['learnable_channel_embeddings']}_Loss:{loss_name}_LRs:{config['training']['max_lr']}:{config['training']['conditioner_multiplier']}_WDs:{config['training']['weight_decay']}:{config['training']['conditioner_weight_decay']}"
    return config, run_name
