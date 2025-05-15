import json
import unittest

from src.config import get_random_config
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Normalizer
from src.flexipresto import Encoder
from src.utils import check_config, config_dir, load_check_config


class TestConfigs(unittest.TestCase):
    @staticmethod
    def check_models_can_be_loaded(config):
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
        config, _ = get_random_config(model_size="tiny")
        loaded_config = check_config(config)
        self.check_models_can_be_loaded(loaded_config)

    def test_random_configs_vitb_tiny(self):
        config, _ = get_random_config(model_size="vitb-tiny")
        loaded_config = check_config(config)
        self.check_models_can_be_loaded(loaded_config)

    def test_random_configs_base(self):
        config, _ = get_random_config(model_size="base")
        loaded_config = check_config(config)
        self.check_models_can_be_loaded(loaded_config)

    def test_normalization_dict(self):
        if NORMALIZATION_DICT_FILENAME.exists():
            with NORMALIZATION_DICT_FILENAME.open("r") as f:
                norm_dict = json.load(f)
        output_dict = {}
        for key, val in norm_dict.items():
            output_dict[key] = val
        normalizer = Normalizer(std=True, normalizing_dicts=output_dict)
        for key, val in normalizer.shift_div_dict.items():
            divs = val["div"]
            for d in divs:
                self.assertNotEqual(d, 0, f"0 in {key}")
