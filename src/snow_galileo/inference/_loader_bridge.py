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

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from snow_galileo.data.config import DATASET_OUTPUT_HW_HIGH_RES, NUM_TIMESTEPS
from snow_galileo.fsc.downstream_augmentation import DownstreamAugmentation
from snow_galileo.fsc.landsat_eval import LandsatEvalDataset

if TYPE_CHECKING:
    from collections.abc import Sequence


def masked_output_for_tif(tif_path: Path) -> Sequence[torch.Tensor]:
    """Return the loader's inference ``MaskedOutput`` for one cube tif.

    Builds a ``LandsatEvalDataset`` in ``inference`` split with ``__init__``
    bypassed (its real ``__init__`` requires a configured data-folder tree),
    setting only the attributes the inference ``__getitem__`` path reads, then
    returns ``ds[0]``'s masked-output tuple (the 13 model-input tensors).

    This mirrors the ``inference_dataset`` fixture in
    ``test_tracer_end_to_end.py`` and the ``split="inference"`` path the GEE
    ``_predict_and_store_output`` runner uses — both unchanged.

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
    ds.normalizer = None
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
