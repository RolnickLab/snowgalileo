r"""Operator entry point — assemble Bow Valley direct-source cubes (TASK-016).

Reads ``cube.yaml`` (:class:`~src.data.local_sources.settings.CubeSettings`), builds
the in-AOI grid, and writes one canonical 308-band cube tif per ``(cell, window_end)``
into ``processing_root/cubes/`` using the **real-adapter** exporter (not placeholder).

This is the cube half of Stage 2. It is **additive** — it composes existing components
(``build_grid``, ``LocalSourceExporter``) and edits no downstream code; the GEE export
path is untouched.

Example:
    uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \\
        --config configs/bow_valley/cube.yaml --limit 4
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from src.data.local_sources.grid import build_grid
from src.data.local_sources.parallel_export import export_cells_parallel
from src.data.local_sources.settings import CubeSettings

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Assemble Bow Valley direct-source cubes.")


@app.command()
def main(
    config: Annotated[
        Path, typer.Option(help="Path to cube.yaml.")
    ] = Path("configs/bow_valley/cube.yaml"),
    limit: Annotated[
        Optional[int],
        typer.Option(help="Cap the number of cells (smoke run); None = all in-AOI cells."),
    ] = None,
    window_end: Annotated[
        Optional[str],
        typer.Option(help="Window-end day YYYY-MM-DD; default = cube.yaml window_end."),
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option(help="Parallel export workers; None = default (~8), clamped to cores/cells."),
    ] = None,
    verify_s1_cache: Annotated[
        bool,
        typer.Option(
            help="Verify the offline per-granule S1 SNAP cache covers each cell's window "
            "(fail loud if missing). On by default; build the cache first with "
            "scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py. "
            "Pass --no-verify-s1-cache to deliberately allow S1-free cubes.",
        ),
    ] = True,
) -> None:
    """Export one cube per ``(cell, window_end)`` from the clipped archive (parallel)."""
    settings = CubeSettings.from_yaml(config)
    end = (
        datetime.date.fromisoformat(window_end)
        if window_end is not None
        else settings.window_end
    )

    grid = build_grid(mode=settings.mode, mode_b_inset_m=settings.mode_b_inset_m)
    if limit is not None:
        grid = grid[:limit]

    logger.info(
        "cube_export_start",
        cells=len(grid),
        window_end=end.isoformat(),
        archive_root=str(settings.archive_root),
        out_dir=str(settings.cubes_dir),
        workers=workers,
    )
    paths = export_cells_parallel(
        cells=grid,
        window_end=end,
        out_dir=settings.cubes_dir,
        archive_root=settings.archive_root,
        workers=workers,
        verify_s1_cache=verify_s1_cache,
    )
    logger.info("cube_export_complete", cubes=len(paths), out_dir=str(settings.cubes_dir))


if __name__ == "__main__":
    app()
