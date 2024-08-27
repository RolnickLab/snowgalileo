import json
import unittest

from src.conditioner import LearnedMixture, LoRAGenerator
from src.config import get_random_config
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Normalizer
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

    def test_random_configs_tiny(self):
        for _ in range(3):
            config, _ = get_random_config(model_size="tiny")
            loaded_config = check_config(config)

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

    def test_random_configs_vitb_tiny(self):
        for _ in range(3):
            config, _ = get_random_config(model_size="vitb-tiny")
            loaded_config = check_config(config)

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

    def test_random_configs_base(self):
        for _ in range(3):
            config, _ = get_random_config(model_size="base")
            loaded_config = check_config(config)

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

    def test_normalization_dict(self):
        if (config_dir / NORMALIZATION_DICT_FILENAME).exists():
            with (config_dir / NORMALIZATION_DICT_FILENAME).open("r") as f:
                norm_dict = json.load(f)
            normalizer = Normalizer(std_clip=True, normalizing_dicts=norm_dict)
            for key, val in normalizer.shift_div_dict.items():
                divs = val["div"]
                for d in divs:
                    self.assertNotEqual(d, 0, f"0 in {key}")
