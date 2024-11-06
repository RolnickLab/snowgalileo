import json
import unittest

from src.conditioner import LearnedMixture, LoRAGenerator, LoRATemplates, TokenConditioner
from src.config import get_random_config
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Normalizer
from src.flexipresto import Encoder
from src.utils import check_config, config_dir, load_check_config


class TestConfigs(unittest.TestCase):
    @staticmethod
    def check_models_can_be_loaded(config):
        # check we can load the models
        if config["training"]["conditioner_mode"] == "lora-g":
            encoder_conditioner = LoRAGenerator(**config["model"]["conditioner"])
            _ = Encoder(**config["model"]["encoder"], conditioner=encoder_conditioner)
        elif config["training"]["conditioner_mode"] == "moe":
            encoder_conditioner = LearnedMixture(**config["model"]["conditioner"])
            _ = Encoder(**config["model"]["encoder"], conditioner=encoder_conditioner)
        elif config["training"]["conditioner_mode"] == "lora-t":
            encoder_conditioner = LoRATemplates(**config["model"]["conditioner"])
            _ = Encoder(**config["model"]["encoder"], conditioner=encoder_conditioner)
        elif config["training"]["conditioner_mode"] == "token":
            encoder_conditioner = TokenConditioner(**config["model"]["conditioner"])
            _ = Encoder(**config["model"]["encoder"], conditioner=encoder_conditioner)
        else:
            assert "conditioner" not in config["model"].keys()
            _ = Encoder(**config["model"]["encoder"])

    def test_configs_mae(self):
        configs = list((config_dir / "mae").glob("*.json"))

        for config_path in configs:
            try:
                loaded_config = load_check_config(config_path.name)
                self.check_models_can_be_loaded(loaded_config)
            except Exception as e:
                print(f"Failed for {config_path} with {e}")
                raise e

    def test_random_configs_tiny(self):
        for c in ["lora-g", "lora-t", "moe"]:
            config, _ = get_random_config(model_size="tiny", conditioner_mode=c)
            loaded_config = check_config(config)
            self.check_models_can_be_loaded(loaded_config)

    def test_random_configs_vitb_tiny(self):
        for c in ["lora-g", "lora-t", "moe"]:
            config, _ = get_random_config(model_size="vitb-tiny", conditioner_mode=c)
            loaded_config = check_config(config)
            self.check_models_can_be_loaded(loaded_config)

    def test_random_configs_base(self):
        for c in ["lora-g", "lora-t", "moe"]:
            config, _ = get_random_config(model_size="base", conditioner_mode=c)
            loaded_config = check_config(config)
            self.check_models_can_be_loaded(loaded_config)

    def test_normalization_dict(self):
        if (config_dir / NORMALIZATION_DICT_FILENAME).exists():
            with (config_dir / NORMALIZATION_DICT_FILENAME).open("r") as f:
                norm_dict = json.load(f)
        output_dict = {}
        for key, val in norm_dict.items():
            if "n" not in key:
                output_dict[int(key)] = val
            else:
                output_dict[key] = val
        normalizer = Normalizer(std=True, normalizing_dicts=output_dict)
        for key, val in normalizer.shift_div_dict.items():
            divs = val["div"]
            for d in divs:
                self.assertNotEqual(d, 0, f"0 in {key}")
