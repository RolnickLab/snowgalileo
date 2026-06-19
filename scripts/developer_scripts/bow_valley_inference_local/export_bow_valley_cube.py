r"""Operator entry point — assemble Bow Valley direct-source cubes (TASK-016).

Reads ``cube.yaml`` (:class:`~src.data.local_sources.settings.CubeSettings`), builds
the in-AOI grid, and writes one canonical 308-band cube tif per ``(cell, window_end)``
into ``processing_root/cubes/`` using the **real-adapter** exporter (not placeholder).

This is the cube half of Stage 2. It is **additive** — it composes existing components
(``build_grid``, ``LocalSourceExporter``) and edits no downstream code; the GEE export
path is untouched.

Example:
    uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \\
        export --config configs/bow_valley/cube.yaml --limit 4

    # Wipe the cube cache on demand (reports entries removed):
    uv run python scripts/developer_scripts/bow_valley_inference_local/export_bow_valley_cube.py \\
        clean-cache --config configs/bow_valley/cube.yaml
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from src.data.local_sources.cube_cache import CubeCache
from src.data.local_sources.cube_cache_cli import (
    CachePolicy,
    CachePolicyError,
    resolve_cache_policy,
)
from src.data.local_sources.grid import build_grid
from src.data.local_sources.parallel_export import export_cells_parallel
from src.data.local_sources.settings import CubeSettings

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Assemble Bow Valley direct-source cubes.")


@app.command()
def export(
    config: Annotated[Path, typer.Option(help="Path to cube.yaml.")] = Path(
        "configs/bow_valley/cube.yaml"
    ),
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
    cache_policy: Annotated[
        CachePolicy,
        typer.Option(
            help="How to treat an existing cube cache: 'prompt' (ask if non-empty; errors "
            "on a non-TTY), 'reuse' (keep it), or 'overwrite' (clear once up front). Use "
            "'overwrite' after an adapter or clip change that the version stamp can't catch.",
        ),
    ] = CachePolicy.PROMPT,
) -> None:
    """Export one cube per ``(cell, window_end)`` from the clipped archive (parallel)."""
    settings = CubeSettings.from_yaml(config)

    # Resolve reuse/overwrite ONCE in this parent process before any worker spawns; the
    # pool then reuses the resulting (clean or kept) dir. Fail loud on a non-TTY prompt
    # rather than silently reusing a possibly-stale cache.
    try:
        resolve_cache_policy(
            root=settings.cube_cache_dir,
            policy=cache_policy,
            max_entries=settings.cache_max_entries,
        )
    except CachePolicyError as exc:
        raise typer.BadParameter(str(exc)) from exc

    end = (
        datetime.date.fromisoformat(window_end) if window_end is not None else settings.window_end
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
        cube_cache_dir=settings.cube_cache_dir,
        cache_max_entries=settings.cache_max_entries,
    )
    logger.info("cube_export_complete", cubes=len(paths), out_dir=str(settings.cubes_dir))


@app.command("clean-cache")
def clean_cache(
    config: Annotated[Path, typer.Option(help="Path to cube.yaml.")] = Path(
        "configs/bow_valley/cube.yaml"
    ),
) -> None:
    """Wipe the cube cache (``CubeSettings.cube_cache_dir``), reporting entries removed.

    A manual reset for the "did my clips/adapters change?" case the version stamp can't
    catch. Constructs the cache once with ``overwrite=True`` — the only path that clears —
    in this single parent process, so there is no cross-worker clear race.
    """
    settings = CubeSettings.from_yaml(config)
    root = settings.cube_cache_dir
    # Count first (read-only scan), then clear via the overwrite path.
    before = len(CubeCache(root, settings.cache_max_entries))
    CubeCache(root, settings.cache_max_entries, overwrite=True)
    logger.info("cube_cache_cleaned", root=str(root), entries_removed=before)
    typer.echo(f"Cleared {before} cube cache entr{'y' if before == 1 else 'ies'} from {root}")


if __name__ == "__main__":
    app()
