"""Read-only shim onto the unchanged ``LandsatEvalDataset`` loader (TASK-015).

**Why this module exists.** Driving the loader for a single in-memory
``(cell, day)`` cube tif requires bypassing its folder-glob ``__init__`` and
setting only the attributes the inference ``__getitem__`` path reads — exactly
what ``test_tracer_end_to_end.py`` does. That ``__new__``-then-set-attrs trick is
the **one** place this pipeline is coupled to the loader's private surface.
Confining it here keeps that coupling out of the driver: if the loader ever
changes its private attributes, **only this file** changes — never the driver,
and never ``src/fsc/`` (downstream is sacred; this module edits nothing there, it
only constructs and reads the public objects).

The loader and ``EncoderWithHead`` are used **as-is**; this is pure orchestration.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from snow_galileo.data.config import (
    DATASET_OUTPUT_HW_HIGH_RES,
    NORMALIZATION_DICT_FILENAME,
    NUM_TIMESTEPS,
)
from snow_galileo.data.dataset import Normalizer
from snow_galileo.fsc.downstream_augmentation import DownstreamAugmentation
from snow_galileo.fsc.landsat_eval import LandsatEvalDataset
from snow_galileo.utils import config_dir

if TYPE_CHECKING:
    from collections.abc import Sequence


@functools.lru_cache(maxsize=1)
def _inference_normalizer() -> Normalizer:
    """Build the same ``Normalizer`` the GEE/eval path uses, once per process.

    ``LandsatEval`` defaults to ``normalization="std"`` and every caller
    (``finetune.py``, ``eval_only.py``, ``run_inference.py``) takes that default, so the
    checkpoint was trained on std-normalized inputs and inference must match. Cached
    because this is called once per cube — hundreds of thousands of times in a full sweep
    — and it reads a JSON off disk.

    Returns:
        A ``std=True`` normalizer built from ``configs/normalizing_dict.json``.
    """
    normalizing_dict = LandsatEvalDataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    return Normalizer(std=True, normalizing_dicts=normalizing_dict)


def masked_output_for_tif(tif_path: Path) -> Sequence[torch.Tensor]:
    """Return the loader's inference ``MaskedOutput`` for one cube tif.

    Builds a ``LandsatEvalDataset`` in ``inference`` split with ``__init__``
    bypassed (its real ``__init__`` requires a configured data-folder tree),
    setting only the attributes the inference ``__getitem__`` path reads, then
    returns ``ds[0]``'s masked-output tuple (the 13 model-input tensors).

    This mirrors the ``split="inference"`` path the GEE ``_predict_and_store_output``
    runner uses (unchanged), **including its normalizer** — that runner reaches it via
    ``_get_dataset``, which always assigns one. Leaving ``normalizer`` unset feeds the
    encoder raw physical units (DN, dB, metres, Kelvin) against a checkpoint trained on
    std-normalized inputs, which yields wrong-but-plausible predictions.

    Args:
        tif_path: A 308-band ``PR_*.tif`` cube on the EPSG:32611 cell grid.

    Returns:
        The masked-output tuple of input tensors (no batch dim), ready to be
        stacked into a batch and passed to ``EncoderWithHead.forward``.
    """
    ds = LandsatEvalDataset.__new__(LandsatEvalDataset)
    ds.split = "inference"
    ds.h5pys_only = False
    ds.h5py_folder = None
    ds.normalizer = _inference_normalizer()
    ds.augmentation = DownstreamAugmentation(False)
    ds.output_hw_high_res = DATASET_OUTPUT_HW_HIGH_RES
    ds.output_timesteps = NUM_TIMESTEPS
    ds.exclude_prediction_date = False
    ds.exclude_prediction_high_res = False
    ds.exclude_prediction_sensors = False
    ds.exclude_prediction_era5 = True
    ds.pairs = [(tif_path, None)]

    masked_output, _path = ds[0]
    return masked_output
