"""Tests for the parallel per-cell cube export helper (perf, SPEC NFR).

Uses the **placeholder** exporter (all-``-9999``, no archive reads) so the test is fast and
archive-free while still exercising the real process-pool path: worker initializer builds an
exporter per process, work items are the picklable ``(GridCell, date)`` pairs, and every
cell's ``PR_*.tif`` lands in ``out_dir``. Also covers the serial fallback (``workers=1``)
and the empty-grid no-op.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from src.data.local_sources import parallel_export
from src.data.local_sources.base import GridCell
from src.data.local_sources.parallel_export import (
    _init_worker,
    _resolve_workers,
    export_cells_parallel,
)

_CELL_M = 1_000.0
_BASE_X = 450_000.0
_BASE_Y = 5_621_000.0


def _cell(cell_id: int, col: int) -> GridCell:
    min_x = _BASE_X + col * _CELL_M
    return GridCell.from_utm_bounds(
        cell_id=cell_id, min_x=min_x, min_y=_BASE_Y - _CELL_M,
        max_x=min_x + _CELL_M, max_y=_BASE_Y,
    )


def test_resolve_workers_clamps_to_cores_and_items() -> None:
    """Worker count is clamped to ``[1, min(cpu_count, n_items)]``."""
    assert _resolve_workers(8, n_items=2) == 2  # never more workers than items
    assert _resolve_workers(1, n_items=100) == 1
    assert _resolve_workers(None, n_items=1) == 1
    assert _resolve_workers(1000, n_items=1000) <= (1000)  # capped by cpu_count internally


def test_empty_grid_is_noop(tmp_path: Path) -> None:
    """No cells → no work, empty result, no pool spawned."""
    out = export_cells_parallel(
        cells=[], window_end=date(2025, 5, 19),
        out_dir=tmp_path / "cubes", archive_root=tmp_path / "arch", workers=4,
    )
    assert out == []


def test_serial_path_writes_all_cubes(tmp_path: Path) -> None:
    """``workers=1`` exports every cell serially (placeholder exporter, no archive)."""
    cubes = tmp_path / "cubes"
    grid = [_cell(0, 0), _cell(1, 1), _cell(2, 2)]
    out = export_cells_parallel(
        cells=grid, window_end=date(2025, 5, 19),
        out_dir=cubes, archive_root=tmp_path / "arch", workers=1, placeholder=True,
    )
    assert len(out) == len(grid)
    assert all(p.exists() and p.name.startswith("PR_") for p in out)


def test_parallel_path_writes_all_cubes(tmp_path: Path) -> None:
    """``workers>1`` spawns the pool and writes one cube per cell."""
    cubes = tmp_path / "cubes"
    grid = [_cell(i, i) for i in range(4)]
    out = export_cells_parallel(
        cells=grid, window_end=date(2025, 5, 19),
        out_dir=cubes, archive_root=tmp_path / "arch", workers=4, placeholder=True,
    )
    assert len(out) == len(grid)
    assert len({p.name for p in out}) == len(grid)  # distinct files
    assert all(p.exists() for p in out)


def test_init_worker_threads_cache_into_exporter(tmp_path: Path) -> None:
    """``cube_cache_dir``/``cache_max_entries`` reach the worker's exporter (step 3).

    Calls the worker initializer directly (no pool) and inspects the exporter it builds.
    Guards the param plumbing — the actual Step-3 risk — without an archive: real mode +
    a cache dir → the exporter owns a ``CubeCache`` with the requested cap; ``None`` →
    no cache.
    """
    from src.data.local_sources.cube_cache import CubeCache

    # Real mode + cache dir → exporter owns a CubeCache with the given cap.
    _init_worker(
        tmp_path / "cubes", tmp_path / "arch", False, False,
        tmp_path / "cube_cache", 1234,
    )
    exp = parallel_export._WORKER_EXPORTER
    assert exp is not None
    assert isinstance(exp._cache, CubeCache)
    assert exp._cache.max_entries == 1234
    assert exp._cache.root == tmp_path / "cube_cache"

    # No cache dir → no cache (behaviour-identical to pre-step-3).
    _init_worker(tmp_path / "cubes", tmp_path / "arch", False, False, None, 1234)
    exp2 = parallel_export._WORKER_EXPORTER
    assert exp2 is not None and exp2._cache is None
