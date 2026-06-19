"""Tracer-bullet end-to-end test for the local-source pipeline (TASK-004).

This test plumbs the *whole* pipeline with all-``-9999`` placeholder adapters:

    LocalSourceExporter.export(cell, window_end)            # new code
        -> a 308-band PR_*.tif on the EPSG:32611 cell grid
    LandsatEvalDataset._tif_to_array(tif)                   # UNCHANGED downstream
        -> the six model tensors + valid-data masks
    EncoderWithHead.forward(...)                            # UNCHANGED downstream
        -> a (10, 10) FSC map in [0, 1]

FSC is degenerate (every input is fill, so everything is masked), but **every
shape, band name, mask path, and filename parse is proven correct before any
real adapter exists** (TASK-006…TASK-014 only swap placeholder bytes for real
bytes — the contract proven here does not move).

The downstream loader and encoder are exercised as the *real* objects, not
re-implementations:

- ``LandsatEvalDataset`` is built via ``__new__`` (its data-folder-dependent
  ``__init__`` is bypassed); ``_tif_to_array`` touches no instance state beyond
  the methods it calls, so this drives the genuine downstream array-assembly
  path against our exported tif.
- The ``Encoder`` is a real (untrained) ``snow_galileo.snowgalileo.Encoder`` wrapped in a
  real ``EncoderWithHead``; we assert only shape and the sigmoid range, not
  values (weights are random — values are meaningless, the *plumbing* is the point).

PLAN §6 nine tracer assertions (all asserted below):
    1. space_time_high_res_x == (100, 100, 8, 15)
    2. space_time_med_res_x  == (5, 5, 8, 2)
    3. space_time_low_res_x  == (2, 2, 8, 11)
    4. time_x                == (8, 9)
    5. space_x               == (100, 100, 14)
    6. static_x              == (3,)
    7. FSC (10, 10) in [0, 1]
    8. valid_data_mask_* set on -9999 / below-threshold inputs (here: everywhere)
    9. filename parses to window_end.month
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from snow_galileo.data.config import DATASET_OUTPUT_HW_HIGH_RES, NO_DATA_VALUE, NUM_TIMESTEPS
from snow_galileo.data.earthengine import eo as _eo
from snow_galileo.data.local_sources.base import CELL_TARGET_CRS, GridCell
from snow_galileo.data.local_sources.layout import (
    DYNAMIC_BANDS,
    STATIC_BANDS,
    TOTAL_BANDS,
    full_band_order,
)
from snow_galileo.fsc.downstream_augmentation import DownstreamAugmentation
from snow_galileo.fsc.landsat_eval import LandsatEvalDataset
from snow_galileo.fsc.patch_predict import EncoderWithHead
from snow_galileo.snowgalileo import Encoder
from snow_galileo.utils import config_dir

# A Bow Valley UTM 11N cell (EPSG:32611) — 1 km, 100x100 px @ 10 m. The UTM
# easting/northing here sit under the AOI's ~50.7 N / -116.5 E centre.
_MIN_X = 450_000.0
_MIN_Y = 5_620_000.0
_CELL_SIZE_M = 1_000.0
_WINDOW_END = date(2025, 5, 28)


@pytest.fixture()
def cell() -> GridCell:
    """One EPSG:32611 1 km cell on the canonical 100x100 @ 10 m target grid."""
    return GridCell.from_utm_bounds(
        cell_id=0,
        min_x=_MIN_X,
        min_y=_MIN_Y,
        max_x=_MIN_X + _CELL_SIZE_M,
        max_y=_MIN_Y + _CELL_SIZE_M,
    )


@pytest.fixture()
def exported_cube(tmp_path: Path, cell: GridCell) -> Path:
    """Export one placeholder cube and return its path.

    Imported lazily so collection of the *other* tracer assertions does not hard
    depend on the exporter module existing at import time.
    """
    from snow_galileo.data.local_sources.exporter import LocalSourceExporter

    exporter = LocalSourceExporter(out_dir=tmp_path, placeholder=True)
    return exporter.export(cell=cell, window_end=_WINDOW_END)


@pytest.fixture(scope="module")
def loader() -> LandsatEvalDataset:
    """Real ``LandsatEvalDataset`` with ``__init__`` bypassed (filename + array path)."""
    return LandsatEvalDataset.__new__(LandsatEvalDataset)


# --- The exported cube itself (AC-1 shape upstream, AC-6 filename) ---------- #


def test_exported_cube_is_308_band_utm_grid(exported_cube: Path, cell: GridCell) -> None:
    """The cube is a 308-band EPSG:32611 100x100 GeoTIFF with -9999 nodata (AC-1/AC-4)."""
    with rasterio.open(exported_cube) as src:
        assert src.count == TOTAL_BANDS == 308
        assert src.width == 100 and src.height == 100
        assert src.crs.to_string() == CELL_TARGET_CRS
        assert src.nodata == NO_DATA_VALUE
        # Placeholder: every band is entirely fill.
        assert np.all(src.read() == NO_DATA_VALUE)


def test_filename_matches_contract_and_parses_to_month(
    exported_cube: Path, loader: LandsatEvalDataset
) -> None:
    """Emitted filename parses to ``window_end.month`` via the real loader (AC-6, assertion 9)."""
    assert exported_cube.name.startswith("PR_")
    assert exported_cube.name.endswith("_SC00.tif")
    assert loader.prediction_month_from_file(exported_cube) == _WINDOW_END.month


# --- AC-5 / AC-26: band-name equality vs create_ee_image's source lists ----- #


def test_band_order_equals_ee_source_lists() -> None:
    """Exporter band order derives from the same lists ``create_ee_image`` uses.

    AC-26 is a band-*order* regression guard "via ``layout.py`` re-export from
    ``eo.py``" (SPEC AC-26). ``create_ee_image`` builds a live ``ee.Image`` (needs
    Earth Engine init) so we cannot diff against a materialised GEE band list in a
    unit test; instead we assert the exporter's full band sequence is exactly the
    ``eo`` source lists interleaved (dynamic x T, then static) — the same lists
    ``create_ee_image`` consumes (``eo.py:450,455``).
    """
    expected: list[str] = []
    for t in range(8):
        expected.extend(f"{name}_t{t}" for name in _eo.EO_ALL_DYNAMIC_IN_TIME_BANDS)
    expected.extend(_eo.EE_SPACE_BANDS)

    assert full_band_order() == expected
    assert len(full_band_order()) == TOTAL_BANDS
    assert DYNAMIC_BANDS == list(_eo.EO_ALL_DYNAMIC_IN_TIME_BANDS)
    assert STATIC_BANDS == list(_eo.EE_SPACE_BANDS)


# --- AC-1 / AC-3 / AC-8: loader array assembly + masks (assertions 1-6, 8) --- #


@pytest.fixture()
def loaded(exported_cube: Path, loader: LandsatEvalDataset):
    """The full ``DatasetOutput`` from driving the real loader on the cube."""
    return loader._tif_to_array(exported_cube)


@pytest.fixture()
def inference_dataset(exported_cube: Path) -> LandsatEvalDataset:
    """A real ``LandsatEvalDataset`` in ``inference`` mode over the one cube.

    ``__init__`` is bypassed (it requires a configured data-folder tree); we set
    only the attributes ``__getitem__``/``load_tif`` read so the **real** masking
    convention (0=valid → invert → channel-group aggregate → 1=masked) is applied,
    rather than hand-rolling masks in the test.
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
    ds.pairs = [(exported_cube, None)]
    return ds


def test_tensor_shapes_match_plan_six(loaded) -> None:
    """The six assembled tensors have the exact PLAN §6 shapes (assertions 1-6)."""
    assert loaded.space_time_high_res_x.shape == (100, 100, 8, 15)
    assert loaded.space_time_med_res_x.shape == (5, 5, 8, 2)
    assert loaded.space_time_low_res_x.shape == (2, 2, 8, 11)
    assert loaded.time_x.shape == (8, 9)
    assert loaded.space_x.shape == (100, 100, 14)
    assert loaded.static_x.shape == (3,)


def test_valid_masks_flag_all_placeholder_inputs_invalid(loaded) -> None:
    """Every all-``-9999`` placeholder input is masked invalid (AC-3, assertion 8).

    ``create_valid_mask`` uses 0=invalid, 1=valid; with every dynamic input at the
    universal nodata value, the high/med/low/time valid-masks must be all-zero.
    """
    assert not np.any(loaded.valid_data_mask_space_time_high_res)
    assert not np.any(loaded.valid_data_mask_space_time_med_res)
    assert not np.any(loaded.valid_data_mask_space_time_low_res)
    assert not np.any(loaded.valid_data_mask_time)


# --- AC-2 / assertion 7: encoder returns a (10, 10) FSC map in [0, 1] -------- #


def test_encoder_returns_fsc_map_in_unit_range(
    inference_dataset: LandsatEvalDataset,
) -> None:
    """A real ``EncoderWithHead`` returns a (10, 10) FSC map in [0, 1] (AC-2, assertion 7).

    Drives the genuine ``__getitem__`` inference path so the encoder receives the
    real ``MaskedOutput`` (correct mask convention + channel-group shapes), then a
    real (untrained) ``Encoder``: values are random, so we assert only shape and
    the sigmoid's [0, 1] range — the plumbing, not the numbers, is the contract.
    """
    with (config_dir / "eval" / "fsc_inference_bow_river_tiny.json").open() as fh:
        eval_config = json.load(fh)["finetune"]  # {"token_mapping": "spatial_mean", ...}

    masked_output, _path = inference_dataset[0]

    torch.manual_seed(0)
    encoder = Encoder()  # untrained: values meaningless, plumbing/shape is the point
    model = EncoderWithHead(
        encoder=encoder,
        patch_size_high_res=10,
        inputs_per_target=10,
        num_patches_per_dim=10,
        sigmoid_slope=1.0,
        eval_config=eval_config,
    ).eval()

    # Add the batch dim the encoder's forward expects (DataLoader-style).
    batched = [torch.as_tensor(t).unsqueeze(0) for t in masked_output]

    with torch.no_grad():
        logits = model(
            *batched, patch_size_high_res=10, patch_size_med_res=1, patch_size_low_res=1
        )

    fsc = logits.reshape(10, 10)
    assert fsc.shape == (10, 10)
    assert torch.all((fsc >= 0.0) & (fsc <= 1.0))
