"""Contract tests for :func:`snow_galileo.inference.model.build_model`.

``build_model`` is shared by both Stage-2 entry points (``infer_aoi_cubes.py`` and
``04_infer_bow_valley_daily_fsc.py``). Its contract is "returns a model that is ready to run":
weights loaded from the checkpoint, on ``infer.device``, in eval mode. Both callers depend
on that — the pre-built-cube runner calls the model directly and would silently run a
dropout-active, randomly-initialised model if any part of it regressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from snow_galileo.data.local_sources.settings import InferenceSettings
from snow_galileo.fsc.patch_predict import EncoderWithHead
from snow_galileo.inference.model import build_model
from snow_galileo.snowgalileo import Encoder
from snow_galileo.utils import config_dir, load_check_config

EVAL_CONFIG_NAME = "fsc_inference_bow_river_tiny.json"


def _reference_model() -> EncoderWithHead:
    """Build the tiny model the way ``build_model`` does, so its state dict strict-loads."""
    with (config_dir / "eval" / EVAL_CONFIG_NAME).open() as fh:
        eval_config = json.load(fh)
    enc_cfg = load_check_config("ai4snow_tiny.json")["model"]["encoder"]
    return EncoderWithHead(
        Encoder(**enc_cfg),
        eval_config=eval_config["finetune"],
        sigmoid_slope=eval_config["hyperparameters_snowgalileo"]["sigmoid_slope"],
    )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    """Write a real (untrained but deterministic) tiny checkpoint to disk."""
    torch.manual_seed(0)
    path = tmp_path / "tiny_finetuned.pth"
    torch.save(_reference_model().state_dict(), path)
    return path


def _settings(checkpoint: Path) -> InferenceSettings:
    return InferenceSettings(
        checkpoint=checkpoint,
        eval_config_name=EVAL_CONFIG_NAME,
        decoder_mode="finetune",
        device="cpu",
    )


def test_build_model_raises_when_checkpoint_absent(tmp_path: Path) -> None:
    """A missing checkpoint must fail loudly, never fall back to random weights."""
    settings = _settings(tmp_path / "does_not_exist.pth")

    with pytest.raises(FileNotFoundError, match="does_not_exist.pth"):
        build_model(settings)


def test_build_model_returns_model_in_eval_mode(checkpoint: Path) -> None:
    """Eval mode is part of the contract: callers forward without setting it themselves."""
    model = build_model(_settings(checkpoint))

    assert not model.training


def test_build_model_returns_model_on_configured_device(checkpoint: Path) -> None:
    """Every parameter lands on ``infer.device`` — ``map_location`` alone would not do it."""
    settings = _settings(checkpoint)

    model = build_model(settings)

    expected = torch.device(settings.device).type
    assert all(param.device.type == expected for param in model.parameters())


def test_build_model_loads_the_checkpoint_weights(checkpoint: Path) -> None:
    """The weights must come from the checkpoint, not from a fresh random init."""
    saved = torch.load(checkpoint, map_location="cpu")

    model = build_model(_settings(checkpoint))

    loaded = model.state_dict()
    assert loaded.keys() == saved.keys()
    assert all(torch.equal(loaded[name].cpu(), tensor.cpu()) for name, tensor in saved.items())
