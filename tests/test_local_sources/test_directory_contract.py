"""Directory-contract test (TASK-016, SPEC AC-32 / FR-20b).

Asserts the **write-boundary contract** of a Stage-2 cube+inference run:

1. **No write into either archive.** A cube+inference run creates/modifies **zero**
   files under ``data/clipped_bow_valley_selection_raw`` or
   ``data/bow_valley_selection_raw`` — Stage 2 is read-only against both archives.
2. **Outputs in the correct ``processing_root`` subdir.** Assembled cubes land in
   ``cubes/``; daily FSC COGs in ``daily_fsc/``.
3. **Intermediate/deliverable separation.** Deleting ``cube_cache/`` + ``scratch/``
   leaves every file in ``cubes/`` + ``daily_fsc/`` intact.

The run is **synthetic and self-contained**: a ``placeholder=True`` exporter (all-``-9999``
cube — no archive reads) and a tiny **untrained** ``EncoderWithHead`` (no checkpoint, no
GPU), all writing under ``tmp_path``. AC-32 is about *where bytes land*, not parity, so the
placeholder cube is sufficient and keeps the test fast and archive-free. The two real
archive roots are snapshotted before/after to prove they are untouched even though the
exporter is pointed at ``tmp_path`` — a regression that wired a write back into an archive
would surface here.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import rasterio
import torch

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell
from src.data.local_sources.paths import LocalPaths

# A 1 km UTM cell (matches the inference-driver test helpers).
_CELL_M = 1_000.0
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


def _snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """Map every file under ``root`` to ``(size, mtime)`` (empty if root absent)."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime)
        for p in root.rglob("*")
        if p.is_file()
    }


def _tiny_model() -> object:
    """Build a tiny untrained ``EncoderWithHead`` (no checkpoint) for the FSC head."""
    from src.fsc.patch_predict import EncoderWithHead
    from src.snowgalileo import Encoder
    from src.utils import config_dir

    with (config_dir / "eval" / "fsc_inference_bow_river_tiny.json").open() as fh:
        eval_config = json.load(fh)["finetune"]

    torch.manual_seed(0)
    return EncoderWithHead(
        encoder=Encoder(),
        patch_size_high_res=10,
        inputs_per_target=10,
        num_patches_per_dim=10,
        sigmoid_slope=1.0,
        eval_config=eval_config,
    )


def test_stage2_run_respects_directory_contract(tmp_path: Path) -> None:
    """AC-32: cube+inference writes only under ``processing_root``, never the archives."""
    from src.data.local_sources.exporter import LocalSourceExporter
    from src.inference.driver import InferenceGridDriver

    paths = LocalPaths()
    before_raw = _snapshot(paths.raw_root)
    before_clipped = _snapshot(paths.clipped_root)

    processing_root = tmp_path / "bow_valley_processing"
    cubes_dir = processing_root / "cubes"
    cube_cache_dir = processing_root / "cube_cache"
    scratch_dir = processing_root / "scratch"
    daily_fsc_dir = processing_root / "daily_fsc"

    exporter = LocalSourceExporter(out_dir=cubes_dir, placeholder=True)
    model = _tiny_model()
    grid = [_cell(0, col=0, row=0), _cell(1, col=1, row=0)]

    driver = InferenceGridDriver(
        exporter=exporter,
        model=model,  # type: ignore[arg-type]
        grid=grid,
        window_start=date(2025, 5, 28),
        window_end=date(2025, 5, 28),
        out_dir=daily_fsc_dir,
        batch_size=2,
    )
    cogs = driver.run()

    # 1. Both archive roots are byte-for-byte untouched (zero writes/mods).
    assert _snapshot(paths.raw_root) == before_raw, "Stage 2 wrote into the raw archive."
    assert _snapshot(paths.clipped_root) == before_clipped, (
        "Stage 2 wrote into the clipped archive."
    )

    # 2. Outputs in the correct processing_root subdirs.
    cube_tifs = list(cubes_dir.glob("PR_*.tif"))
    assert len(cube_tifs) == len(grid), "Each cell's cube should land in cubes/."
    assert len(cogs) == 1, "One daily-FSC COG for the one configured day."
    (cog,) = cogs
    assert cog.parent == daily_fsc_dir
    with rasterio.open(cog) as src:
        assert src.crs.to_epsg() == 32611
        assert src.nodata == NO_DATA_VALUE

    # 3. Intermediate/deliverable separation: deleting cache + scratch leaves
    #    the deliverables (cubes/, daily_fsc/) intact.
    cube_cache_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (cube_cache_dir / "0").mkdir(exist_ok=True)
    (cube_cache_dir / "0" / "20250528_s2.npz").write_bytes(b"intermediate")
    (scratch_dir / "worker.tmp").write_bytes(b"transient")

    deliverables_before = _snapshot(cubes_dir) | _snapshot(daily_fsc_dir)
    import shutil

    shutil.rmtree(cube_cache_dir)
    shutil.rmtree(scratch_dir)

    assert not cube_cache_dir.exists()
    assert not scratch_dir.exists()
    assert _snapshot(cubes_dir) | _snapshot(daily_fsc_dir) == deliverables_before, (
        "Deleting cube_cache/ + scratch/ must not affect cubes/ or daily_fsc/."
    )


def test_processing_subdirs_derive_from_root(tmp_path: Path) -> None:
    """``CubeSettings`` derives every Stage-2 subdir from ``processing_root`` (FR-20b)."""
    from src.data.local_sources.settings import CubeSettings

    settings = CubeSettings(processing_root=tmp_path / "proc")  # type: ignore[call-arg]
    root = tmp_path / "proc"
    assert settings.cube_cache_dir == root / "cube_cache"
    assert settings.cubes_dir == root / "cubes"
    assert settings.daily_fsc_dir == root / "daily_fsc"
    assert settings.manifests_dir == root / "manifests"
    assert settings.scratch_dir == root / "scratch"
    # No subdir escapes the processing root.
    for sub in (
        settings.cube_cache_dir,
        settings.cubes_dir,
        settings.daily_fsc_dir,
        settings.manifests_dir,
        settings.scratch_dir,
    ):
        assert root in sub.parents
