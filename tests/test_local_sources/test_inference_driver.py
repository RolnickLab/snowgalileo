"""Tests for the Stage-2 inference orchestration (TASK-015).

Covers ``src/inference/{windows,mosaic,driver}.py``:

- **windows** — sliding 8-day window + inference-day enumeration (pure date math).
- **DailyMosaicWriter** (AC-28, AC-29) — per-day FSC COG in EPSG:32611, disjoint
  seams, NN-only placement, all-masked → nodata, coverage-fraction tag.
- **InferenceGridDriver** (AC-31) — iterates configured window × all in-AOI cells,
  never the legacy CSV ``date`` column.

Everything is synthetic: a tiny untrained ``EncoderWithHead`` (or a stub model)
and a placeholder/monkeypatched exporter, so the suite needs **no** GPU and **no**
checkpoint — mirroring ``test_tracer_end_to_end.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch

from src.data.config import NO_DATA_VALUE, NUM_TIMESTEPS
from src.data.local_sources.base import GridCell
from src.inference.mosaic import DailyMosaicWriter
from src.inference.windows import eight_day_window, inference_days

# A 1 km cell at 100 m FSC px → 10×10 block per cell.
_CELL_M = 1_000.0
_FSC_PX = 10
_BASE_X = 450_000.0
_BASE_Y = 5_621_000.0


def _cell(cell_id: int, col: int, row: int) -> GridCell:
    """Build a 1 km UTM cell at grid position ``(col, row)`` (row grows south)."""
    min_x = _BASE_X + col * _CELL_M
    max_y = _BASE_Y - row * _CELL_M
    return GridCell.from_utm_bounds(
        cell_id=cell_id,
        min_x=min_x,
        min_y=max_y - _CELL_M,
        max_x=min_x + _CELL_M,
        max_y=max_y,
    )

# --------------------------------------------------------------------------- #
# windows.py
# --------------------------------------------------------------------------- #


def test_eight_day_window_is_ascending_and_ends_at_window_end() -> None:
    """``eight_day_window`` yields ``NUM_TIMESTEPS`` ascending days ending at d."""
    window_end = date(2025, 4, 6)
    days = eight_day_window(window_end)

    assert len(days) == NUM_TIMESTEPS == 8
    assert days == sorted(days)
    assert days[-1] == window_end
    assert days[0] == date(2025, 3, 30)  # window_end - 7
    # Contiguous, 1-day stride.
    assert all((days[i + 1] - days[i]).days == 1 for i in range(len(days) - 1))


def test_inference_days_inclusive_span() -> None:
    """``inference_days`` enumerates every day in [start, end] inclusive."""
    start, end = date(2025, 4, 6), date(2025, 5, 28)
    days = inference_days(start, end)

    assert days[0] == start
    assert days[-1] == end
    assert len(days) == (end - start).days + 1 == 53
    assert days == sorted(days)


def test_inference_days_single_day() -> None:
    """A one-day window yields exactly that day."""
    d = date(2025, 4, 6)
    assert inference_days(d, d) == [d]


def test_inference_days_rejects_reversed_range() -> None:
    """``window_end`` before ``window_start`` is a hard error."""
    with pytest.raises(ValueError, match="precedes"):
        inference_days(date(2025, 5, 28), date(2025, 4, 6))


# --------------------------------------------------------------------------- #
# mosaic.py — DailyMosaicWriter (AC-28, AC-29)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def grid_2x2() -> list[GridCell]:
    """Four adjacent 1 km cells in a 2×2 block."""
    return [
        _cell(0, col=0, row=0),
        _cell(1, col=1, row=0),
        _cell(2, col=0, row=1),
        _cell(3, col=1, row=1),
    ]


def test_mosaic_2x2_disjoint_seams_no_double_write(
    grid_2x2: list[GridCell], tmp_path: Path
) -> None:
    """Four adjacent cells place four 10×10 blocks at disjoint offsets (AC-29).

    Each cell gets a distinct constant FSC so we can prove placement is exact and
    no pixel is shared. The writer's internal seam guard asserts no double-write.
    """
    writer = DailyMosaicWriter(grid=grid_2x2, out_dir=tmp_path, fsc_px_per_cell=_FSC_PX)
    # 20×20 mosaic (2 cells × 10 px each side).
    assert (writer.height, writer.width) == (20, 20)

    fsc_by_cell = {cid: np.full((10, 10), 0.1 * (cid + 1), dtype=np.float32) for cid in range(4)}
    out = writer.write_day(date(2025, 4, 6), fsc_by_cell)

    with rasterio.open(out) as src:
        data = src.read(1)
    # Expected quadrant offsets: cell0=TL, cell1=TR, cell2=BL, cell3=BR.
    assert np.allclose(data[0:10, 0:10], 0.1)
    assert np.allclose(data[0:10, 10:20], 0.2)
    assert np.allclose(data[10:20, 0:10], 0.3)
    assert np.allclose(data[10:20, 10:20], 0.4)
    # Every pixel written exactly once → no nodata, full coverage.
    assert not np.any(data == NO_DATA_VALUE)


def test_mosaic_placement_is_nn_only_no_invented_values(
    grid_2x2: list[GridCell], tmp_path: Path
) -> None:
    """Placed FSC values are bit-identical to the input patches — no interpolation.

    A patch with sharp internal structure (a 0/1 checkerboard) must survive
    placement unchanged: a bilinear/averaging mosaic would create intermediate
    values; nearest/direct placement cannot.
    """
    writer = DailyMosaicWriter(grid=[grid_2x2[0]], out_dir=tmp_path, fsc_px_per_cell=_FSC_PX)
    checker = np.indices((10, 10)).sum(axis=0) % 2  # 0/1 checkerboard
    patch = checker.astype(np.float32)
    out = writer.write_day(date(2025, 4, 6), {0: patch})

    with rasterio.open(out) as src:
        data = src.read(1)
    # Only 0.0 and 1.0 survive — no invented intermediate FSC.
    assert set(np.unique(data)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(data[0:10, 0:10], patch)


def test_mosaic_all_masked_cell_is_nodata_and_coverage_recorded(
    grid_2x2: list[GridCell], tmp_path: Path
) -> None:
    """A ``None`` (all-masked) cell stays nodata; coverage fraction is tagged (AC-28)."""
    writer = DailyMosaicWriter(grid=grid_2x2, out_dir=tmp_path, fsc_px_per_cell=_FSC_PX)
    # Only two of four cells predict; the other two are None → nodata.
    fsc_by_cell: dict[int, np.ndarray | None] = {
        0: np.full((10, 10), 0.5, dtype=np.float32),
        1: None,
        2: np.full((10, 10), 0.5, dtype=np.float32),
        3: None,
    }
    out = writer.write_day(date(2025, 4, 6), fsc_by_cell)

    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 32611
        assert src.nodata == NO_DATA_VALUE
        data = src.read(1)
        coverage = float(src.tags()["aoi_coverage_fraction"])

    # Cell1 (TR) and cell3 (BR) blocks are nodata.
    assert np.all(data[0:10, 10:20] == NO_DATA_VALUE)
    assert np.all(data[10:20, 10:20] == NO_DATA_VALUE)
    # 2 of 4 cells valid → 0.5 coverage.
    assert coverage == pytest.approx(0.5)


def test_mosaic_rejects_wrong_patch_shape(grid_2x2: list[GridCell], tmp_path: Path) -> None:
    """A patch that is not ``fsc_px_per_cell`` square is rejected."""
    writer = DailyMosaicWriter(grid=grid_2x2, out_dir=tmp_path, fsc_px_per_cell=_FSC_PX)
    with pytest.raises(ValueError, match="patch shape"):
        writer.write_day(date(2025, 4, 6), {0: np.zeros((5, 5), dtype=np.float32)})


def test_mosaic_requires_non_empty_grid(tmp_path: Path) -> None:
    """An empty grid is a construction error."""
    with pytest.raises(ValueError, match="non-empty grid"):
        DailyMosaicWriter(grid=[], out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# driver.py — InferenceGridDriver (AC-31)
# --------------------------------------------------------------------------- #


class _StubExporter:
    """Records every ``(cell_id, window_end)`` it is asked to export.

    Returns a sentinel path; the loader is stubbed in these tests so the path is
    never opened — we are asserting the *loop semantics*, not the cube bytes.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, date]] = []

    def export(self, *, cell: GridCell, window_end: date) -> Path:
        self.calls.append((cell.cell_id, window_end))
        return Path(f"/tmp/cube_{cell.cell_id}_{window_end:%Y%m%d}.tif")


class _StubModel:
    """A stand-in ``EncoderWithHead``: returns a fixed ``(B, 100, 1)`` logits map."""

    def __init__(self) -> None:
        self.seen_batch_sizes: list[int] = []

    def to(self, _device: object) -> _StubModel:
        return self

    def eval(self) -> _StubModel:
        return self

    def __call__(self, *batched: object, **_kw: object):
        batch = batched[0]
        b = batch.shape[0]  # type: ignore[union-attr]
        self.seen_batch_sizes.append(b)
        return torch.full((b, 100, 1), 0.5)


@pytest.fixture()
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    """Stub the loader bridge so the driver never opens a real cube tif.

    Returns a thirteen-tensor MaskedOutput whose six masks are all-ones (a valid
    cell). The driver imports the symbol into its own namespace, so we patch it
    there.
    """
    def _fake(_tif: Path):
        x = [torch.zeros(1) for _ in range(6)]  # inputs (unused by the stub model)
        masks = [torch.ones(1) for _ in range(6)]  # all-valid
        month = torch.zeros(1, dtype=torch.long)
        return (*x, *masks, month)

    monkeypatch.setattr("src.inference.driver.masked_output_for_tif", _fake)


def test_driver_iterates_window_x_cells_ignoring_csv_date(
    grid_2x2: list[GridCell], patched_loader: None, tmp_path: Path
) -> None:
    """Driver predicts window-days × all cells; CSV ``date`` never enters the loop (AC-31).

    The grid cells carry only geometry — no per-cell date. Two cells (which in the
    legacy CSV would have *different* sampled dates) are both exported for every
    configured day. We assert the export call set is exactly the cross-product of
    the configured window × cells.
    """
    from src.inference.driver import InferenceGridDriver

    exporter = _StubExporter()
    model = _StubModel()
    start, end = date(2025, 4, 6), date(2025, 4, 8)  # 3-day window

    driver = InferenceGridDriver(
        exporter=exporter,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        grid=grid_2x2,
        window_start=start,
        window_end=end,
        out_dir=tmp_path,
        batch_size=2,
    )
    outputs = driver.run()

    # One COG per configured day.
    assert len(outputs) == 3
    # Exactly the cross-product window-days × cells was exported.
    expected = {
        (cid, day)
        for day in (date(2025, 4, 6), date(2025, 4, 7), date(2025, 4, 8))
        for cid in range(4)
    }
    assert set(exporter.calls) == expected
    assert len(exporter.calls) == 12  # 3 days × 4 cells, no CSV-date filtering
    # Each cell is predicted on every day (not gated by any per-cell date).
    for cid in range(4):
        days_for_cell = {d for (c, d) in exporter.calls if c == cid}
        assert days_for_cell == {date(2025, 4, 6), date(2025, 4, 7), date(2025, 4, 8)}


def test_driver_drops_fully_masked_cell_to_nodata(
    grid_2x2: list[GridCell], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cell whose every valid-mask is all-zero yields no prediction → nodata (AC-28)."""
    from src.inference.driver import InferenceGridDriver

    def _fake(tif: Path):
        # cell_0 fully masked (masks all-zero), the rest valid (masks all-one).
        cid = int(tif.stem.split("_")[1])
        x = [torch.zeros(1) for _ in range(6)]
        mask_val = 0.0 if cid == 0 else 1.0
        masks = [torch.full((1,), mask_val) for _ in range(6)]
        return (*x, *masks, torch.zeros(1, dtype=torch.long))

    monkeypatch.setattr("src.inference.driver.masked_output_for_tif", _fake)

    driver = InferenceGridDriver(
        exporter=_StubExporter(),  # type: ignore[arg-type]
        model=_StubModel(),  # type: ignore[arg-type]
        grid=grid_2x2,
        window_start=date(2025, 4, 6),
        window_end=date(2025, 4, 6),
        out_dir=tmp_path,
        batch_size=4,
    )
    (out,) = driver.run()

    with rasterio.open(out) as src:
        data = src.read(1)
    # cell_0 is the top-left 10×10 block → all nodata; the rest are 0.5.
    assert np.all(data[0:10, 0:10] == NO_DATA_VALUE)
    assert np.all(data[0:10, 10:20] == 0.5)


def test_driver_end_to_end_with_real_loader_and_encoder(tmp_path: Path) -> None:
    """Full plumbing: real exporter + real loader bridge + untrained encoder.

    Proves the driver drives the *unchanged* downstream loader and
    ``EncoderWithHead`` end to end on a placeholder cube — values are meaningless
    (random weights, all-masked input), so we assert only that a valid COG is
    produced. Mirrors ``test_tracer_end_to_end.py``.
    """
    import json

    from src.data.local_sources.exporter import LocalSourceExporter
    from src.fsc.patch_predict import EncoderWithHead
    from src.inference.driver import InferenceGridDriver
    from src.snowgalileo import Encoder
    from src.utils import config_dir

    cube_dir = tmp_path / "cubes"
    fsc_dir = tmp_path / "daily_fsc"
    exporter = LocalSourceExporter(out_dir=cube_dir, placeholder=True)

    with (config_dir / "eval" / "fsc_inference_bow_river_tiny.json").open() as fh:
        eval_config = json.load(fh)["finetune"]

    torch.manual_seed(0)
    model = EncoderWithHead(
        encoder=Encoder(),
        patch_size_high_res=10,
        inputs_per_target=10,
        num_patches_per_dim=10,
        sigmoid_slope=1.0,
        eval_config=eval_config,
    )

    grid = [_cell(0, col=0, row=0)]
    driver = InferenceGridDriver(
        exporter=exporter,
        model=model,
        grid=grid,
        window_start=date(2025, 5, 28),
        window_end=date(2025, 5, 28),
        out_dir=fsc_dir,
        batch_size=1,
    )
    (out,) = driver.run()

    assert out.exists()
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 32611
        assert src.nodata == NO_DATA_VALUE
        assert "aoi_coverage_fraction" in src.tags()
