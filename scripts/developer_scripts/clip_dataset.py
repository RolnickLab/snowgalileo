"""AOI clip stage CLI (Phase 0.5) — non-destructive crop to ``data/bow_valley_inference_aoi.geojson``.

Crops every raw dataset in ``data/bow_valley_selection_raw`` to the authoritative
AOI, into ``data/clipped_bow_valley_selection_raw`` — the single archive root every
``LocalSource*`` adapter reads. Each product passes a two-stage intersect gate
(``clip.gate``); failing products produce **no output file**. A per-source manifest
records every decision.

Commands:
    clip-source   Clip one modality (e.g. ``worldcover``).
    clip-all      Clip every modality.

Run ``--dry-run`` to evaluate only the metadata gate (no pixels decoded, no writes).

The clip routines and gate live in the ``src.data.local_sources.clip`` package; this
module is just the CLI boundary. Run via ``uv run python scripts/developer_scripts/clip_dataset.py``.
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

app = typer.Typer(help="Non-destructive AOI clip stage for the Bow Valley raw archive.")

# Path defaults resolve from LocalPaths (env-overridable, LOCAL_ prefix) so the
# clip stage can be repointed at another region without editing this CLI. The
# --input-dir / --output-dir / --aoi flags still win for one-off runs.
_PATHS = LocalPaths()
DEFAULT_INPUT = _PATHS.raw_root
DEFAULT_OUTPUT = _PATHS.clipped_root
DEFAULT_AOI = _PATHS.aoi_path
MANIFEST_NAME = "clip_manifest.csv"


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


if __name__ == "__main__":
    app()
