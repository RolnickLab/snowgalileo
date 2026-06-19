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

from pathlib import Path

import structlog
import typer

from src.data.local_sources.clip.settings import load_aoi_polygon
from src.data.local_sources.paths import LocalPaths
from src.data.local_sources.s1_snap import (
    _DEFAULT_GPT,
    _DEFAULT_GRAPH,
    build_s1_cache,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Build the Bow Valley Sentinel-1 per-granule SNAP cache.",
)

# Path defaults resolve from LocalPaths (env-overridable, LOCAL_ prefix) so the driver
# can be repointed at another region without editing this CLI; the flags win per-run.
_PATHS = LocalPaths()
DEFAULT_ARCHIVE_ROOT = _PATHS.raw_root / "sentinel1"
DEFAULT_CACHE_DIR = _PATHS.clipped_root / "sentinel1_snap"
DEFAULT_AOI = _PATHS.aoi_path


def _preflight(*, archive_root: Path, cache_dir: Path, aoi: Path, gpt: Path, graph: Path) -> None:
    """Fail loudly (``typer.Exit(1)``) before any SNAP run if a prerequisite is missing.

    Args:
        archive_root: Raw S1 archive holding full-swath ``S1*_IW_GRDH_*.zip``.
        cache_dir: Output directory for the ``s1_grd_*.tif`` cache (unchecked — created).
        aoi: AOI polygon GeoJSON.
        gpt: ESA SNAP ``gpt`` executable.
        graph: Production SNAP graph XML.

    Raises:
        typer.Exit: With code ``1`` if any hard prerequisite is missing.
    """
    problems: list[str] = []
    if not gpt.exists():
        problems.append(f"ESA SNAP gpt not found at {gpt} (install SNAP or pass --gpt).")
    if not graph.exists():
        problems.append(f"SNAP graph not found at {graph}.")
    if not aoi.exists():
        problems.append(f"AOI polygon not found at {aoi}.")
    if not archive_root.exists():
        problems.append(f"Raw S1 archive not found at {archive_root}.")
    elif not list(archive_root.glob("S1*_IW_GRDH_*.zip")):
        problems.append(f"No S1*_IW_GRDH_*.zip granules under {archive_root} — nothing to build.")
    if problems:
        for problem in problems:
            logger.error("s1_cache_preflight_failed", problem=problem)
        raise typer.Exit(code=1)


@app.command()
def build(
    archive_root: Path = typer.Option(
        DEFAULT_ARCHIVE_ROOT,
        "--archive-root",
        help="Raw S1 archive holding full-swath S1*_IW_GRDH_*.zip.",
    ),
    cache_dir: Path = typer.Option(
        DEFAULT_CACHE_DIR, "--cache-dir", help="Output directory for the s1_grd_*.tif cache."
    ),
    aoi: Path = typer.Option(
        DEFAULT_AOI,
        "--aoi",
        help="AOI polygon (EPSG:4326 GeoJSON) the post-TC Subset crops to.",
    ),
    gpt: Path = typer.Option(_DEFAULT_GPT, "--gpt", help="ESA SNAP gpt executable."),
    graph: Path = typer.Option(_DEFAULT_GRAPH, "--graph", help="Production SNAP graph XML."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-run SNAP even where a cache tif already exists."
    ),
) -> None:
    """Build the per-granule S1 SNAP dB+angle cache from raw (offline, heavy)."""
    _preflight(archive_root=archive_root, cache_dir=cache_dir, aoi=aoi, gpt=gpt, graph=graph)

    aoi_polygon = load_aoi_polygon(aoi)
    n_granules = len(list(archive_root.glob("S1*_IW_GRDH_*.zip")))
    logger.info(
        "s1_cache_build_begin",
        n_granules=n_granules,
        archive_root=str(archive_root),
        cache_dir=str(cache_dir),
        overwrite=overwrite,
    )

    cached = build_s1_cache(
        archive_root=archive_root,
        aoi_4326=aoi_polygon,
        cache_dir=cache_dir,
        gpt=gpt,
        graph=graph,
        overwrite=overwrite,
    )

    logger.info(
        "s1_cache_build_complete",
        n_cache_tifs=len(cached),
        cache_dir=str(cache_dir),
    )
    typer.echo(f"Built/verified {len(cached)} S1 cache tif(s) in {cache_dir}")


if __name__ == "__main__":
    app()
