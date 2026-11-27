"""The loader bridge must normalize its inputs exactly like the GEE/eval path.

Regression test for a silent inference bug: ``_loader_bridge`` set
``ds.normalizer = None``, so the local direct-source pipeline fed the encoder **raw
physical units** (reflectance DN, S1 dB, DEM metres, ERA5 Kelvin) while the checkpoint
was finetuned on std-normalized inputs. Every prediction was wrong-but-plausible, and
nothing failed loudly.

The GEE runner (``LandsatEval._predict_and_store_output``) reaches its inference dataset
through ``_get_dataset``, which *always* assigns a normalizer, defaulting to
``normalization="std"``. Every caller that built the checkpoint (``finetune.py``) or
consumes it (``eval_only.py``, ``run_inference.py``) takes that default. So the bridge's
normalizer must equal ``Normalizer(std=True, normalizing_dicts=<normalizing_dict.json>)``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from snow_galileo.data.config import NORMALIZATION_DICT_FILENAME
from snow_galileo.data.dataset import Normalizer
from snow_galileo.data.local_sources.base import GridCell
from snow_galileo.fsc.landsat_eval import LandsatEvalDataset
from snow_galileo.inference._loader_bridge import _inference_normalizer, masked_output_for_tif
from snow_galileo.utils import config_dir

_MIN_X = 450_000.0
_MIN_Y = 5_620_000.0
_CELL_SIZE_M = 1_000.0
_WINDOW_END = date(2025, 5, 28)


@pytest.fixture()
def valid_cube(tmp_path: Path) -> Path:
    """A 308-band cube whose pixels are all **valid** (no ``-9999``).

    The placeholder exporter emits an all-``-9999`` cube, and ``Normalizer._normalize`` is
    ``np.where(valid_mask, (x - shift) / div, NO_DATA_VALUE)`` — so on a fully-invalid cube
    normalization is a genuine no-op and the missing-normalizer bug is *undetectable*. That
    is exactly how it survived the tracer test. Overwrite the fill with a valid constant so
    the normalizer has something to act on.
    """
    import rasterio

    from snow_galileo.data.earthengine.eo import EE_SPACE_BANDS, ESA_WORLDCOVER_BAND_INDEX
    from snow_galileo.data.local_sources.exporter import LocalSourceExporter

    cell = GridCell.from_utm_bounds(
        cell_id=0,
        min_x=_MIN_X,
        min_y=_MIN_Y,
        max_x=_MIN_X + _CELL_SIZE_M,
        max_y=_MIN_Y + _CELL_SIZE_M,
    )
    exporter = LocalSourceExporter(out_dir=tmp_path, placeholder=True)
    cube = exporter.export(cell=cell, window_end=_WINDOW_END)

    with rasterio.open(cube, "r+") as src:
        for band in range(1, src.count + 1):
            src.write(np.ones((src.height, src.width), dtype=np.float32), band)
        # The ESA WorldCover band is categorical, not continuous: a blanket 1.0 is not a valid
        # class code, so ``one_hot_encode_esa_worldcover`` warns and blanks all 11 one-hot
        # channels to NO_DATA_VALUE. Write a real class code (70 = snow and ice) so the
        # WorldCover block carries valid data the normalizer can act on.
        wc_band = src.count - len(EE_SPACE_BANDS) + ESA_WORLDCOVER_BAND_INDEX + 1
        src.write(np.full((src.height, src.width), 70.0, dtype=np.float32), wc_band)
    return cube


def test_bridge_normalizer_matches_the_eval_path() -> None:
    """The bridge's normalizer is the one ``LandsatEval._get_dataset`` would build."""
    expected = Normalizer(
        std=True,
        normalizing_dicts=LandsatEvalDataset.load_normalization_values(
            path=config_dir / NORMALIZATION_DICT_FILENAME
        ),
    )
    actual = _inference_normalizer()

    assert actual.shift_div_dict.keys() == expected.shift_div_dict.keys()
    for group, expected_values in expected.shift_div_dict.items():
        for key in ("shift", "div"):
            np.testing.assert_array_equal(
                actual.shift_div_dict[group][key],
                expected_values[key],
                err_msg=f"{group}/{key} diverges from the eval path's normalizer",
            )


def test_bridge_actually_applies_normalization(valid_cube: Path) -> None:
    """Model inputs are normalized, not raw — the bug this file exists for.

    Drives the same loader with ``normalizer=None`` (the old behaviour) and asserts the
    bridge's tensors differ. If someone reverts the bridge to ``None``, this fails.
    """
    normalized = masked_output_for_tif(valid_cube)

    unnormalized = _bridge_dataset_attrs(valid_cube)
    unnormalized.normalizer = None
    raw, _path = unnormalized[0]

    assert any(
        not torch.allclose(torch.as_tensor(normalized[i]).float(), torch.as_tensor(raw[i]).float())
        for i in range(len(raw))
    ), "Bridge output is identical to the un-normalized loader output — normalization is a no-op."


def _bridge_dataset_attrs(tif_path: Path) -> LandsatEvalDataset:
    """A dataset configured exactly as the bridge does, so only ``normalizer`` differs."""
    from snow_galileo.data.config import DATASET_OUTPUT_HW_HIGH_RES, NUM_TIMESTEPS
    from snow_galileo.fsc.downstream_augmentation import DownstreamAugmentation

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
    return ds
