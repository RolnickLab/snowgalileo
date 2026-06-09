"""Parallel per-cell cube export (SPEC NFR — ``multiprocessing.Pool`` per-cell export).

Per-cell cube assembly is embarrassingly parallel: each cell produces its own
``PR_*.tif`` from read-only archives, sharing no state. With the windowed-read fix each
worker holds only ~600 MB, so a 16-core / 62 GB host can run many in parallel.

**Why a process pool with a worker-built exporter (not a pickled one).** The adapters hold
rasterio/h5py state and are cheap to rebuild but awkward to pickle, and JP2/GDAL decode is
CPU-bound (a thread pool would serialize on the GIL). So each worker process builds **one**
:class:`~src.data.local_sources.exporter.LocalSourceExporter` in an initializer and reuses
it for every cell it draws — only the tiny ``(GridCell, date)`` work items cross the process
boundary. ``GridCell`` (a frozen dataclass of affine/polygon/ints) and ``datetime.date`` are
picklable; the exporter is not sent.

This module is **additive orchestration** — it constructs and drives the unchanged exporter,
touching no adapter or downstream code.
"""

from __future__ import annotations

import datetime
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import structlog

from src.data.local_sources.base import GridCell
from src.data.local_sources.exporter import LocalSourceExporter

logger = structlog.get_logger(__name__)

#: Default worker count: leave a couple of cores for I/O / the OS. Capped by cell count
#: and ``os.cpu_count()`` at call time.
_DEFAULT_WORKERS = 8

# --- per-worker process state (built once by the initializer) ----------------- #
_WORKER_EXPORTER: LocalSourceExporter | None = None


def _init_worker(out_dir: Path, archive_root: Path, placeholder: bool) -> None:
    """Build this worker process's single exporter (runs once per process)."""
    global _WORKER_EXPORTER
    _WORKER_EXPORTER = LocalSourceExporter(
        out_dir=out_dir, placeholder=placeholder, archive_root=archive_root
    )


def _export_one(item: tuple[GridCell, datetime.date]) -> tuple[int, str]:
    """Export one ``(cell, window_end)`` in the worker; return ``(cell_id, path)``."""
    cell, window_end = item
    assert _WORKER_EXPORTER is not None, "worker exporter not initialized"
    path = _WORKER_EXPORTER.export(cell=cell, window_end=window_end)
    return cell.cell_id, str(path)


def _resolve_workers(requested: int | None, n_items: int) -> int:
    """Clamp the worker count to ``[1, min(cpu_count, n_items)]``."""
    cores = os.cpu_count() or 1
    base = requested if requested is not None else _DEFAULT_WORKERS
    return max(1, min(base, cores, n_items))


def export_cells_parallel(
    *,
    cells: list[GridCell],
    window_end: datetime.date,
    out_dir: Path,
    archive_root: Path,
    workers: int | None = None,
    placeholder: bool = False,
) -> list[Path]:
    """Export one cube per cell across a process pool; return the written paths.

    Each worker process builds its own exporter once (see :func:`_init_worker`) and
    processes cells as they are scheduled. Falls back to a **serial** loop when only one
    worker is resolved (single cell, or ``workers=1``) so a small ``--limit`` run pays no
    pool-spawn cost.

    Args:
        cells: The grid cells to export (each yields one ``PR_*.tif``).
        window_end: The 8-day window's end (prediction) day for every cell.
        out_dir: Cube output directory (the workers write here).
        archive_root: The clipped archive every worker's adapters read.
        workers: Requested worker count; ``None`` → :data:`_DEFAULT_WORKERS`, clamped to
            ``[1, min(cpu_count, len(cells))]``.
        placeholder: Build placeholder (all-``-9999``) exporters in the workers — the
            archive-free tracer mode (default ``False`` = real adapters).

    Returns:
        The written cube paths (order not guaranteed — sort if needed).
    """
    if not cells:
        return []

    n_workers = _resolve_workers(workers, len(cells))
    items = [(cell, window_end) for cell in cells]

    if n_workers == 1:
        _init_worker(out_dir, archive_root, placeholder)
        paths = [Path(_export_one(item)[1]) for item in items]
        logger.info("exported_cells_serial", cells=len(paths), out_dir=str(out_dir))
        return paths

    out_dir.mkdir(parents=True, exist_ok=True)
    pool_paths: list[Path] = []
    logger.info(
        "export_pool_start",
        cells=len(cells),
        workers=n_workers,
        window_end=window_end.isoformat(),
    )
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(out_dir, archive_root, placeholder),
    ) as pool:
        futures = [pool.submit(_export_one, item) for item in items]
        done = 0
        for fut in as_completed(futures):
            _cell_id, path = fut.result()
            pool_paths.append(Path(path))
            done += 1
            if done % 25 == 0 or done == len(items):
                logger.info("export_progress", done=done, total=len(items))
    logger.info("export_pool_complete", cells=len(pool_paths), workers=n_workers)
    return pool_paths
