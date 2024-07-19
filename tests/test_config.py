import unittest

from src.conditioner import TokenConditioner
from src.flexipresto import Encoder, PrestoPixelDecoder
from src.utils import config_dir, load_check_config


class TestConfigs(unittest.TestCase):
    def test_configs_mae(self):
        configs = list((config_dir / "mae").glob("*.json"))

        for config_path in configs:
            loaded_config = load_check_config(config_path.name, "mae")

            # check we can load the models
            if loaded_config["training"]["use_conditions"]:
                conditioner = TokenConditioner(**loaded_config["model"]["conditioner"])
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=conditioner)
                _ = PrestoPixelDecoder(**loaded_config["model"]["decoder"])
            else:
                assert "conditioner" not in loaded_config["model"].keys()
