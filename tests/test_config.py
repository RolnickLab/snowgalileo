import unittest

from src.conditioner import LearnedMixture
from src.flexipresto import Encoder, PrestoPixelDecoder
from src.utils import config_dir, load_check_config


class TestConfigs(unittest.TestCase):
    def test_configs_mae(self):
        configs = list((config_dir / "mae").glob("*.json"))

        for config_path in configs:
            loaded_config = load_check_config(config_path.name, "mae")

            # check we can load the models
            if loaded_config["training"]["encoder_conditioner"]:
                encoder_conditioner = LearnedMixture(
                    **loaded_config["model"]["encoder_conditioner"]
                )
                _ = Encoder(**loaded_config["model"]["encoder"], conditioner=encoder_conditioner)
            else:
                assert "encoder_conditioner" not in loaded_config["model"].keys()
                _ = Encoder(**loaded_config["model"]["encoder"])
            if loaded_config["training"]["decoder_conditioner"]:
                decoder_conditioner = LearnedMixture(
                    **loaded_config["model"]["decoder_conditioner"]
                )
                _ = PrestoPixelDecoder(
                    **loaded_config["model"]["decoder"], conditioner=decoder_conditioner
                )
            else:
                assert "decoder_conditioner" not in loaded_config["model"].keys()
                _ = PrestoPixelDecoder(**loaded_config["model"]["decoder"])
