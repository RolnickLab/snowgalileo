r"""Operator entry point — daily Bow Valley FSC inference (TASK-016).

Reads ``cube.yaml`` (the sweep definition: window, mode, roots) and ``inference.yaml``
(how to run the model: checkpoint, eval config, batch, device), builds the in-AOI grid,
constructs the pretrained ``EncoderWithHead`` from the checkpoint via the **existing**
load path, and runs :class:`~src.inference.driver.InferenceGridDriver` to write one daily
FSC COG per inference day into ``processing_root/daily_fsc/``.

**Downstream is sacred.** The model is built exactly as ``scripts/eval_only.py`` /
``predict_and_generate_output.py`` do (``Encoder(**enc_cfg)`` → ``EncoderWithHead`` →
``load_state_dict``); no downstream code is touched, and the GEE runner keeps working in
parallel. The checkpoint is **required** — if it is absent the script fails loudly rather
than silently initializing random weights (an all-random sweep would yield a plausible but
meaningless COG).

Example:
    uv run python scripts/developer_scripts/bow_valley_inference_local/infer_bow_valley_daily_fsc.py \\
        --cube-config configs/bow_valley/cube.yaml \\
        --config configs/bow_valley/inference.yaml --limit 4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import structlog
import torch
import typer

from src.data.local_sources.exporter import LocalSourceExporter
from src.data.local_sources.grid import build_grid
from src.data.local_sources.settings import CubeSettings, InferenceSettings
from src.fsc.patch_predict import EncoderWithHead
from src.inference.driver import InferenceGridDriver
from src.snowgalileo import Encoder
from src.utils import config_dir, load_check_config

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Run daily Bow Valley FSC inference.")


def _build_model(infer: InferenceSettings) -> EncoderWithHead:
    """Build the pretrained ``EncoderWithHead`` from the configured checkpoint.

    Mirrors ``scripts/eval_only.py`` exactly: the eval-config filename's size token
    selects the ``ai4snow_<size>.json`` encoder config, the head ``eval_config`` and
    ``sigmoid_slope`` come from the eval JSON, then the finetuned state is strict-loaded.

    Args:
        infer: The inference settings (checkpoint, eval config, decoder mode, device).

    Returns:
        The loaded model on ``infer.device``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist (fail loudly — never
            silently initialize random weights).
    """
    if not infer.checkpoint.exists():
        raise FileNotFoundError(
            f"Inference checkpoint not found: {infer.checkpoint}. Set `checkpoint` in "
            "inference.yaml (or INFER_CHECKPOINT) to a finetuned EncoderWithHead .pth. "
            "Q6 must be resolved before an inference run."
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
    )
    return model


@app.command()
def main(
    cube_config: Annotated[
        Path, typer.Option(help="Path to cube.yaml (sweep window/mode/roots).")
    ] = Path("configs/bow_valley/cube.yaml"),
    config: Annotated[
        Path, typer.Option(help="Path to inference.yaml (model run config).")
    ] = Path("configs/bow_valley/inference.yaml"),
    limit: Annotated[
        Optional[int],
        typer.Option(help="Cap the number of cells (smoke run); None = all in-AOI cells."),
    ] = None,
) -> None:
    """Run the daily FSC inference sweep and write one COG per day."""
    cube = CubeSettings.from_yaml(cube_config)
    infer = InferenceSettings.from_yaml(config)

    grid = build_grid(mode=cube.mode, mode_b_inset_m=cube.mode_b_inset_m)
    if limit is not None:
        grid = grid[:limit]

    exporter = LocalSourceExporter(
        out_dir=cube.cubes_dir,
        placeholder=False,
        archive_root=cube.archive_root,
        cube_cache_dir=cube.cube_cache_dir,
        cache_max_entries=cube.cache_max_entries,
    )
    model = _build_model(infer)
    out_dir = infer.out_dir if infer.out_dir is not None else cube.daily_fsc_dir

    driver = InferenceGridDriver(
        exporter=exporter,
        model=model,
        grid=grid,
        window_start=cube.window_start,
        window_end=cube.window_end,
        out_dir=out_dir,
        device=infer.device,
        batch_size=infer.batch_size,
        export_workers=infer.export_workers,
    )
    logger.info(
        "inference_start",
        cells=len(grid),
        window=f"{cube.window_start.isoformat()}..{cube.window_end.isoformat()}",
        out_dir=str(out_dir),
    )
    cogs = driver.run()
    logger.info("inference_complete", days=len(cogs), out_dir=str(out_dir))


if __name__ == "__main__":
    app()
