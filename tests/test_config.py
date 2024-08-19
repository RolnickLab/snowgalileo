import unittest

from src.conditioner import LearnedMixture, LoRAGenerator
from src.config import get_random_config
from src.flexipresto import Encoder
from src.utils import check_config, config_dir, load_check_config


class TestConfigs(unittest.TestCase):
    def test_configs_mae(self):
        configs = list((config_dir / "mae").glob("*.json"))

        for config_path in configs:
            loaded_config = load_check_config(config_path.name)

            if loaded_config["training"]["conditioner_mode"] == "lora":
                encoder_conditioner = LoRAGenerator(**loaded_config["model"]["conditioner"])
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=encoder_conditioner)
            elif loaded_config["training"]["conditioner_mode"] == "moe":
                encoder_conditioner = LearnedMixture(**loaded_config["model"]["conditioner"])
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=encoder_conditioner)
            else:
                assert "conditioner" not in loaded_config["model"].keys()
                _ = Encoder(**loaded_config["model"]["encoder"])

    def test_random_configs(self):
        for i in range(10):
            loaded_config = check_config(get_random_config())

            # check we can load the models
            if loaded_config["training"]["conditioner_mode"] == "lora":
                encoder_conditioner = LoRAGenerator(**loaded_config["model"]["conditioner"])
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=encoder_conditioner)
            elif loaded_config["training"]["conditioner_mode"] == "moe":
                encoder_conditioner = LearnedMixture(**loaded_config["model"]["conditioner"])
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=encoder_conditioner)
            else:
                assert "conditioner" not in loaded_config["model"].keys()
                _ = Encoder(**loaded_config["model"]["encoder"])
