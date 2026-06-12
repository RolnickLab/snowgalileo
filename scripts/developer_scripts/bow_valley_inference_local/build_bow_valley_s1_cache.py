"""Build the Bow Valley Sentinel-1 per-cell SNAP dB+angle cache (offline prestep).

The cube exporter's :class:`~src.data.local_sources.s1.S1Adapter` reads a **pre-built**
per-``(granule, cell)`` cache of terrain-corrected σ⁰ (dB) + local-incidence-angle tifs;
it does **not** run SNAP itself (a thin, hermetic read port). This script is that build
step — the S1 analogue of the clip stage — running ESA SNAP ``gpt`` once per covered
``(granule, cell)`` over the production grid.

Run it whenever the clipped S1 archive changes, before exporting cubes::

    uv run python scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py
    # then (re-)export cubes — S1 bands will be populated, not -9999.

It is idempotent: already-cached ``(granule, cell)`` tifs are skipped unless
``--overwrite``. SNAP terrain correction is bounded **per cell** (a full-AOI/full-scene
run NPE-corrupts on the clip's range-geometry empty regions), so the cache is keyed by
``cache_tif_name(stem, cell.cell_id)`` — the same key the adapter resolves by.

Fails loudly up front if ESA SNAP ``gpt``, the SNAP graph, or the clipped S1 archive is
missing, rather than producing a partial/empty cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from src.data.local_sources.grid import build_grid
from src.data.local_sources.paths import LocalPaths
from src.data.local_sources.s1_snap import (
    _DEFAULT_GPT,
    _DEFAULT_GRAPH,
    build_s1_cache,
)

logger = structlog.get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    paths = LocalPaths()
    parser = argparse.ArgumentParser(
        description="Build the Bow Valley Sentinel-1 per-cell SNAP cache.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=paths.clipped_root / "sentinel1",
        help="Clipped S1 archive holding S1*_IW_GRDH_*.zip (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=paths.clipped_root / "sentinel1_snap",
        help="Output directory for the s1_grd_*_cell*.tif cache (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=("A", "B"),
        default="A",
        help="Grid sweep mode (A=legacy-CSV cells in AOI; B=AOI tiling). Default: %(default)s.",
    )
    parser.add_argument(
        "--gpt", type=Path, default=_DEFAULT_GPT, help="ESA SNAP gpt executable."
    )
    parser.add_argument(
        "--graph", type=Path, default=_DEFAULT_GRAPH, help="Production SNAP graph XML."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run SNAP even where a cache tif already exists.",
    )
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> None:
    """Fail loudly before any SNAP run if a hard prerequisite is missing."""
    problems: list[str] = []
    if not args.gpt.exists():
        problems.append(f"ESA SNAP gpt not found at {args.gpt} (install SNAP or pass --gpt).")
    if not args.graph.exists():
        problems.append(f"SNAP graph not found at {args.graph}.")
    if not args.archive_root.exists():
        problems.append(f"Clipped S1 archive not found at {args.archive_root}.")
    else:
        n_granules = len(list(args.archive_root.glob("S1*_IW_GRDH_*.zip")))
        if n_granules == 0:
            problems.append(
                f"No S1*_IW_GRDH_*.zip granules under {args.archive_root} — nothing to build."
            )
    if problems:
        for problem in problems:
            logger.error("s1_cache_preflight_failed", problem=problem)
        sys.exit(1)


def main() -> None:
    args = _parse_args()
    _preflight(args)

    cells = build_grid(args.mode)
    n_granules = len(list(args.archive_root.glob("S1*_IW_GRDH_*.zip")))
    logger.info(
        "s1_cache_build_begin",
        n_granules=n_granules,
        n_cells=len(cells),
        archive_root=str(args.archive_root),
        cache_dir=str(args.cache_dir),
        overwrite=args.overwrite,
    )

    cached = build_s1_cache(
        archive_root=args.archive_root,
        cells=cells,
        cache_dir=args.cache_dir,
        gpt=args.gpt,
        graph=args.graph,
        overwrite=args.overwrite,
    )

    logger.info(
        "s1_cache_build_complete",
        n_cache_tifs=len(cached),
        cache_dir=str(args.cache_dir),
    )
    print(f"Built/verified {len(cached)} S1 cache tif(s) in {args.cache_dir}")


if __name__ == "__main__":
    main()
