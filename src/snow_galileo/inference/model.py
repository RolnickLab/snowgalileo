"""Construct the finetuned ``EncoderWithHead`` from an :class:`InferenceSettings`.

Shared by both Stage-2 operator entry points — the direct-source sweep
(``04_infer_bow_valley_daily_fsc.py``) and the pre-built-cube runner (``infer_aoi_cubes.py``).
Each carried its own private copy of this loader, and the copies had already drifted on
their return contract (one returned a CPU model, the other a device-resident one), so the
two scripts agreed on how to build the model only by coincidence. One function, one
contract.

The construction mirrors ``scripts/eval_only.py`` / ``predict_and_generate_output.py``
(``Encoder(**enc_cfg)`` -> ``EncoderWithHead`` -> ``load_state_dict``). That legacy GEE
path builds the same model from argparse arguments rather than from settings and is
deliberately left untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
import torch

from snow_galileo.data.local_sources.settings import InferenceSettings
from snow_galileo.fsc.patch_predict import EncoderWithHead
from snow_galileo.snowgalileo import Encoder
from snow_galileo.utils import config_dir, load_check_config

logger = structlog.get_logger(__name__)


def build_model(infer: InferenceSettings) -> EncoderWithHead:
    """Build the pretrained ``EncoderWithHead`` from the configured checkpoint.

    The eval-config filename's size token selects the ``ai4snow_<size>.json`` encoder
    config, the head ``eval_config`` and ``sigmoid_slope`` come from the eval JSON, then the
    finetuned state is strict-loaded.

    Args:
        infer: Inference settings (checkpoint, eval config, decoder mode, device).

    Returns:
        The loaded model, moved to ``infer.device`` and in eval mode — ready to run.
        The move is explicit rather than left to the caller: ``load_state_dict`` copies
        into the model's existing parameters in place, so ``map_location`` on its own
        would place the *state dict* on the device and leave the model on the CPU.

    Raises:
        FileNotFoundError: If the checkpoint does not exist — fail loudly rather than
            silently initialise random weights, which would yield a meaningless COG.
    """
    if not infer.checkpoint.exists():
        raise FileNotFoundError(
            f"Inference checkpoint not found: {infer.checkpoint}. Point `checkpoint` in the "
            "inference config (or INFER_CHECKPOINT) at a finetuned EncoderWithHead .pth."
        )

    with (config_dir / "eval" / infer.eval_config_name).open() as fh:
        eval_config = json.load(fh)
    sigmoid_slope = eval_config["hyperparameters_snowgalileo"]["sigmoid_slope"]

    # Encoder size token is the trailing word of the eval-config filename (e.g. "tiny").
    size_token = Path(infer.eval_config_name).stem.split("_")[-1]
    enc_cfg = load_check_config(f"ai4snow_{size_token}.json")["model"]["encoder"]

    model = EncoderWithHead(
        Encoder(**enc_cfg),
        eval_config=eval_config[infer.decoder_mode],
        sigmoid_slope=sigmoid_slope,
    )
    state = torch.load(infer.checkpoint, map_location=infer.device)
    model.load_state_dict(state)
    logger.info(
        "model_loaded",
        checkpoint=str(infer.checkpoint),
        size=size_token,
        decoder_mode=infer.decoder_mode,
        device=infer.device,
    )
    return model.to(infer.device).eval()
