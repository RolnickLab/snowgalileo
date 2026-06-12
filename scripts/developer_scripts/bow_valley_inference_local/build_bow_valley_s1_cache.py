"""Build the Bow Valley Sentinel-1 per-granule SNAP dB+angle cache (offline prestep).

The cube exporter's :class:`~src.data.local_sources.s1.S1Adapter` reads a **pre-built**
per-granule, AOI-wide cache of terrain-corrected σ⁰ + ellipsoid-incidence-angle tifs; it
does **not** run SNAP itself (a thin, hermetic read port). This script is that build
step — running ESA SNAP ``gpt`` **once per raw granule** over the AOI, with the
``geoRegion`` Subset applied after Terrain-Correction (map geometry — no "Empty region!"
NPE; S1 GRD best practice). One AOI-wide tif per granule; the adapter windows it to each
cell, so it serves both grid modes (no per-cell or per-mode build).

Run it whenever the **raw** S1 archive changes, before exporting cubes::

    uv run python scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py
    # then (re-)export cubes — S1 bands will be populated, not -9999.

It is idempotent: already-cached per-granule tifs are skipped unless ``--overwrite``.

Fails loudly up front if ESA SNAP ``gpt``, the SNAP graph, the raw S1 archive, or the
AOI is missing, rather than producing a partial/empty cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

from src.data.local_sources.clip.settings import load_aoi_polygon
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
        description="Build the Bow Valley Sentinel-1 per-granule SNAP cache.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=paths.raw_root / "sentinel1",
        help="Raw S1 archive holding full-swath S1*_IW_GRDH_*.zip (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=paths.clipped_root / "sentinel1_snap",
        help="Output directory for the s1_grd_*.tif cache (default: %(default)s).",
    )
    parser.add_argument(
        "--aoi",
        type=Path,
        default=paths.aoi_path,
        help="AOI polygon (EPSG:4326 GeoJSON) the post-TC Subset crops to (default: %(default)s).",
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
    if not args.aoi.exists():
        problems.append(f"AOI polygon not found at {args.aoi}.")
    if not args.archive_root.exists():
        problems.append(f"Raw S1 archive not found at {args.archive_root}.")
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

    aoi = load_aoi_polygon(args.aoi)
    n_granules = len(list(args.archive_root.glob("S1*_IW_GRDH_*.zip")))
    logger.info(
        "s1_cache_build_begin",
        n_granules=n_granules,
        archive_root=str(args.archive_root),
        cache_dir=str(args.cache_dir),
        overwrite=args.overwrite,
    )

    cached = build_s1_cache(
        archive_root=args.archive_root,
        aoi_4326=aoi,
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
