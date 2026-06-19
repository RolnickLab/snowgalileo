"""Raw-archive processing stage CLI — clip + Sentinel-1 SNAP, into the read roots.

Turns the raw download archive in ``data/bow_valley_selection_raw`` into the inputs
the cube adapters read. Most modalities are a **non-destructive AOI clip** (crop to
``data/bow_valley_inference_aoi.geojson`` → ``data/clipped_bow_valley_selection_raw``,
the single archive every ``LocalSource*`` adapter reads); each product passes a
two-stage intersect gate (``clip.gate``), failing products produce **no output file**,
and a per-source manifest records every decision.

**Sentinel-1 is processed, never clipped.** The cube's S1 value domain (GEE
``COPERNICUS/S1_GRD``) needs the full ESA SNAP chain — calibration + terrain
correction. So S1 is *processed* from the **raw** granules into a per-granule,
AOI-wide dB+angle cache (``process-s1``) that is the **single** S1 product everything
downstream reads — both the ``S1Adapter`` (cube) and the viewer's S1 quicklook. There
is no raw-DN clipped-S1 product; ``sentinel1`` is **not** a clip source. See
``docs/agents/planning/bow_valley/020-data-ingestion/022-s1-pergranule-snap.md``.

Commands:
    clip-source   Clip one modality (e.g. ``worldcover``). S1 is not a valid source.
    clip-all      Clip every clip-modality (S1 excluded — it is processed, not clipped).
    process-s1    Build the per-granule S1 SNAP dB+angle cache from raw (offline, heavy).
    process-all   process-s1 (raw S1 → SNAP), THEN clip every other modality — the full
                  raw → read-roots pipeline, in the process-then-clip order.

Run ``--dry-run`` (clip commands) to evaluate only the metadata gate (no writes).

The clip routines/gate live in ``src.data.local_sources.clip``; the S1 SNAP chain in
``src.data.local_sources.s1_snap``. This module is the CLI boundary. Run via
``uv run python scripts/developer_scripts/bow_valley_inference_local/process_raw_dataset.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import structlog
import typer

from src.data.local_sources.clip.manifest import ManifestRow, write_manifest
from src.data.local_sources.clip.orchestrator import SOURCES, clip_one_source
from src.data.local_sources.clip.settings import ClipSettings, load_aoi_polygon
from src.data.local_sources.paths import LocalPaths
from src.data.local_sources.s1_snap import (
    _DEFAULT_GPT,
    _DEFAULT_GRAPH,
    build_s1_cache,
)

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
        if os.environ.get("LOG_JSON")
        else structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()

app = typer.Typer(help="Process the Bow Valley raw archive: clip + Sentinel-1 SNAP.")

# Path defaults resolve from LocalPaths (env-overridable, LOCAL_ prefix) so the
# clip stage can be repointed at another region without editing this CLI. The
# --input-dir / --output-dir / --aoi flags still win for one-off runs.
_PATHS = LocalPaths()
DEFAULT_INPUT = _PATHS.raw_root
DEFAULT_OUTPUT = _PATHS.clipped_root
DEFAULT_AOI = _PATHS.aoi_path
MANIFEST_NAME = "clip_manifest.csv"

# Sentinel-1 SNAP processing: read RAW granules, write the per-granule AOI-wide cache
# under the clipped root (where the S1Adapter resolves it as archive_root/sentinel1_snap).
DEFAULT_S1_RAW = _PATHS.raw_root / "sentinel1"
DEFAULT_S1_CACHE = _PATHS.clipped_root / "sentinel1_snap"


def _summarize(rows: list[ManifestRow]) -> dict[str, int]:
    """Count rows by action for the run summary."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.action.value] = counts.get(row.action.value, 0) + 1
    return counts


@app.command("clip-source")
def clip_source(
    source: str = typer.Argument(..., help=f"One of: {', '.join(SOURCES)}"),
    input_dir: Path = typer.Option(DEFAULT_INPUT, "--input-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output-dir"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Gate only; write nothing."),
) -> None:
    """Clip one modality through the intersect gate and write its manifest."""
    if source not in SOURCES:
        raise typer.BadParameter(f"Unknown source '{source}'. Choose from {SOURCES}.")

    settings = ClipSettings()
    aoi = load_aoi_polygon(aoi_path)
    rows = clip_one_source(
        source=source,
        input_dir=input_dir,
        output_dir=output_dir,
        aoi_4326=aoi,
        settings=settings,
        dry_run=dry_run,
    )

    if not dry_run:
        write_manifest(rows, output_dir / source / MANIFEST_NAME)
    logger.info("clip-source done", source=source, dry_run=dry_run, **_summarize(rows))


def _run_clip_all(
    *, input_dir: Path, output_dir: Path, aoi_path: Path, dry_run: bool, only: Optional[str]
) -> None:
    """Clip every modality (or an ``--only`` subset) and write a combined manifest."""
    settings = ClipSettings()
    aoi = load_aoi_polygon(aoi_path)
    selected = only.split(",") if only else SOURCES

    all_rows: list[ManifestRow] = []
    for source in selected:
        if not (input_dir / source).exists():
            logger.warning("source missing; skipping", source=source)
            continue
        rows = clip_one_source(
            source=source,
            input_dir=input_dir,
            output_dir=output_dir,
            aoi_4326=aoi,
            settings=settings,
            dry_run=dry_run,
        )
        if not dry_run:
            write_manifest(rows, output_dir / source / MANIFEST_NAME)
        all_rows.extend(rows)

    if not dry_run:
        write_manifest(all_rows, output_dir / MANIFEST_NAME)
    logger.info("clip-all done", dry_run=dry_run, total=len(all_rows), **_summarize(all_rows))


@app.command("clip-all")
def clip_all(
    input_dir: Path = typer.Option(DEFAULT_INPUT, "--input-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output-dir"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Gate only; write nothing."),
    only: Optional[str] = typer.Option(
        None, "--only", help="Comma-separated subset of sources to run."
    ),
) -> None:
    """Clip every modality (or an ``--only`` subset) and write a combined manifest."""
    _run_clip_all(
        input_dir=input_dir, output_dir=output_dir, aoi_path=aoi_path,
        dry_run=dry_run, only=only,
    )


def _run_process_s1(
    *, raw_s1_dir: Path, cache_dir: Path, aoi_path: Path, gpt: Path, graph: Path, overwrite: bool
) -> int:
    """Build the per-granule S1 SNAP cache from raw; return the number of cache tifs.

    Fails loud (``typer.Exit(1)``) if a hard prerequisite (gpt, graph, AOI, raw archive,
    or any granule) is missing, rather than producing a partial/empty cache.
    """
    problems: list[str] = []
    if not gpt.exists():
        problems.append(f"ESA SNAP gpt not found at {gpt} (install SNAP or pass --gpt).")
    if not graph.exists():
        problems.append(f"SNAP graph not found at {graph}.")
    if not aoi_path.exists():
        problems.append(f"AOI polygon not found at {aoi_path}.")
    if not raw_s1_dir.exists():
        problems.append(f"Raw S1 archive not found at {raw_s1_dir}.")
    elif not list(raw_s1_dir.glob("S1*_IW_GRDH_*.zip")):
        problems.append(f"No S1*_IW_GRDH_*.zip granules under {raw_s1_dir} — nothing to build.")
    if problems:
        for problem in problems:
            logger.error("process-s1 preflight failed", problem=problem)
        raise typer.Exit(code=1)

    logger.info(
        "process-s1 begin",
        raw_s1_dir=str(raw_s1_dir), cache_dir=str(cache_dir), overwrite=overwrite,
    )
    cached = build_s1_cache(
        archive_root=raw_s1_dir,
        aoi_4326=load_aoi_polygon(aoi_path),
        cache_dir=cache_dir,
        gpt=gpt,
        graph=graph,
        overwrite=overwrite,
    )
    logger.info("process-s1 done", n_cache_tifs=len(cached), cache_dir=str(cache_dir))
    return len(cached)


@app.command("process-s1")
def process_s1(
    raw_s1_dir: Path = typer.Option(DEFAULT_S1_RAW, "--raw-s1-dir"),
    cache_dir: Path = typer.Option(DEFAULT_S1_CACHE, "--cache-dir"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
    gpt: Path = typer.Option(_DEFAULT_GPT, "--gpt", help="ESA SNAP gpt executable."),
    graph: Path = typer.Option(_DEFAULT_GRAPH, "--graph", help="Production SNAP graph XML."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-run SNAP even where a cache tif already exists."
    ),
) -> None:
    """Process raw Sentinel-1 → per-granule AOI-wide SNAP dB+angle cache (offline, heavy).

    SNAP runs once per raw granule over the AOI (geoRegion Subset applied after Terrain-
    Correction). Idempotent — already-cached granules are skipped unless ``--overwrite``;
    raw granules are read-only. The ``S1Adapter`` windows each AOI-wide tif per cell.
    """
    n = _run_process_s1(
        raw_s1_dir=raw_s1_dir, cache_dir=cache_dir, aoi_path=aoi_path,
        gpt=gpt, graph=graph, overwrite=overwrite,
    )
    typer.echo(f"Built/verified {n} S1 cache tif(s) in {cache_dir}")


@app.command("process-all")
def process_all(
    input_dir: Path = typer.Option(DEFAULT_INPUT, "--input-dir"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT, "--output-dir"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
    only: Optional[str] = typer.Option(
        None, "--only", help="Comma-separated subset of clip sources to run."
    ),
    raw_s1_dir: Path = typer.Option(DEFAULT_S1_RAW, "--raw-s1-dir"),
    cache_dir: Path = typer.Option(DEFAULT_S1_CACHE, "--cache-dir"),
    gpt: Path = typer.Option(_DEFAULT_GPT, "--gpt"),
    graph: Path = typer.Option(_DEFAULT_GRAPH, "--graph"),
    overwrite_s1: bool = typer.Option(
        False, "--overwrite-s1", help="Re-run SNAP even where an S1 cache tif exists."
    ),
) -> None:
    """Full raw → read-roots pipeline: process raw S1 through SNAP, then clip the rest.

    Order is **process-then-clip**: ``process-s1`` runs first (raw S1 → the per-granule
    SNAP dB+angle cache the cube and viewer both read — the heavy step, hours), then
    ``clip-all`` crops every other modality (S1 is not a clip source). Clip is not dry-run
    here (it writes the archive). Both steps are idempotent, so re-running resumes where it
    left off.
    """
    _run_process_s1(
        raw_s1_dir=raw_s1_dir, cache_dir=cache_dir, aoi_path=aoi_path,
        gpt=gpt, graph=graph, overwrite=overwrite_s1,
    )
    _run_clip_all(
        input_dir=input_dir, output_dir=output_dir, aoi_path=aoi_path, dry_run=False, only=only,
    )
    logger.info("process-all done")


if __name__ == "__main__":
    app()
