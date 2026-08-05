r"""Operator entry point — daily Bow Valley FSC inference (TASK-016).

Reads ``cube.yaml`` (the sweep definition: window, mode, roots) and ``inference.yaml``
(how to run the model: checkpoint, eval config, batch, device), builds the in-AOI grid,
constructs the pretrained ``EncoderWithHead`` from the checkpoint via the **existing**
load path, and runs :class:`~snow_galileo.inference.driver.InferenceGridDriver` to write one daily
FSC COG per inference day into ``processing_root/daily_fsc/``.

**Downstream is sacred.** The model is built exactly as ``scripts/eval_only.py`` /
``predict_and_generate_output.py`` do (``Encoder(**enc_cfg)`` → ``EncoderWithHead`` →
``load_state_dict``); no downstream code is touched, and the GEE runner keeps working in
parallel. The checkpoint is **required** — if it is absent the script fails loudly rather
than silently initializing random weights (an all-random sweep would yield a plausible but
meaningless COG).

**Two cube sources.** By default cubes are assembled from the archive on the fly
(``LocalSourceExporter``) — the fused build+infer sweep. With ``--cubes-only`` the cubes
already in ``cubes_dir`` are reused instead (:class:`PrebuiltCubeSource`), which is the
mode to use when re-running inference after a model/mosaic change: the export, not the
forward pass, dominates the sweep. Both inject into the same
:class:`~snow_galileo.inference.driver.InferenceGridDriver`, so the inference path is one
code path, not two.

**Selecting days.** With no flags the days come from ``cube.yaml``'s
``[window_start, window_end]`` range. ``--dates`` / ``--dates-file`` instead run an
explicit, possibly non-contiguous list — the cubes carry their prediction date in the
filename, so any subset of already-built days can be re-inferred on its own.

**Incomplete days are skipped, never patched.** Under ``--cubes-only`` each requested day is
pre-flighted against the cube dir; a day missing any cell's cube is dropped whole and its
absent cubes are listed in ``<out_dir>/missing_cubes.txt`` plus an end-of-run summary. A
mosaic with a nodata hole in it renders as a plausible COG and nothing downstream can tell
it is incomplete — so it is never written.

Example:
    uv run python scripts/developer_scripts/bow_valley_inference_local/04_infer_bow_valley_daily_fsc.py \\
        --cube-config configs/bow_valley/cube.yaml \\
        --config configs/bow_valley/inference.yaml --limit 4

    # Re-infer three already-built days over the cubes on disk:
    uv run python scripts/developer_scripts/bow_valley_inference_local/04_infer_bow_valley_daily_fsc.py \\
        --cubes-only --dates 2025-04-06,2025-04-10,2025-04-26
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer

from snow_galileo.data.local_sources.base import GridCell
from snow_galileo.data.local_sources.cube_cache_cli import (
    CachePolicy,
    CachePolicyError,
    resolve_cache_policy,
)
from snow_galileo.data.local_sources.exporter import LocalSourceExporter
from snow_galileo.data.local_sources.grid import build_grid
from snow_galileo.data.local_sources.settings import CubeSettings, InferenceSettings
from snow_galileo.inference.driver import InferenceGridDriver
from snow_galileo.inference.model import build_model
from snow_galileo.inference.prebuilt import PrebuiltCubeSource
from snow_galileo.inference.windows import inference_days

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Run daily Bow Valley FSC inference.")

#: Name of the artifact listing every cube a ``--cubes-only`` run could not find.
MISSING_CUBES_FILENAME = "missing_cubes.txt"


def _resolve_days(
    *,
    dates: str | None,
    dates_file: Path | None,
    cube: CubeSettings,
) -> list[datetime.date]:
    """Resolve which days to infer: an explicit list, or the configured window.

    ``--dates`` and ``--dates-file`` are mutually exclusive. Either way the result is
    deduplicated and sorted; an explicit list need not be contiguous (each cube carries its
    own 8-day input window, baked in at build time).

    Args:
        dates: Comma-separated ISO days (``2025-04-06,2025-04-10``), or ``None``.
        dates_file: File of ISO days, one per line; ``#`` comments and blanks ignored.
        cube: The sweep settings, whose window is the fallback.

    Returns:
        The days to infer, ascending.

    Raises:
        typer.BadParameter: If both date options are given, if a day is not ISO
            ``YYYY-MM-DD``, or if the resolved list is empty.
    """
    if dates is not None and dates_file is not None:
        raise typer.BadParameter("Pass --dates or --dates-file, not both.")

    if dates is not None:
        tokens = [token.strip() for token in dates.split(",")]
    elif dates_file is not None:
        if not dates_file.exists():
            raise typer.BadParameter(f"--dates-file {dates_file} does not exist.")
        tokens = [
            stripped
            for line in dates_file.read_text().splitlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        ]
    else:
        return inference_days(cube.window_start, cube.window_end)

    days: set[datetime.date] = set()
    for token in tokens:
        if not token:
            continue
        try:
            days.add(datetime.date.fromisoformat(token))
        except ValueError as exc:
            raise typer.BadParameter(f"{token!r} is not an ISO date (YYYY-MM-DD).") from exc

    if not days:
        raise typer.BadParameter("No dates resolved — the list is empty.")
    return sorted(days)


def _complete_days(
    *,
    source: PrebuiltCubeSource,
    grid: list[GridCell],
    days: list[datetime.date],
    out_dir: Path,
) -> tuple[list[datetime.date], dict[datetime.date, list[str]]]:
    """Split the requested days into complete ones and ones missing cubes.

    Runs before any inference so an incomplete day is dropped whole. A day whose cubes are
    partly absent would otherwise mosaic into a COG with a nodata hole — a file that renders
    plausibly and that nothing downstream can distinguish from a complete one.

    Args:
        source: The prebuilt-cube source to resolve filenames against.
        grid: Every cell the mosaic needs.
        days: The requested days.
        out_dir: Where :data:`MISSING_CUBES_FILENAME` is written when anything is missing.

    Returns:
        ``(complete_days, missing_by_day)`` — the days safe to infer, and the absent cube
        filenames keyed by the day they were skipped for.
    """
    complete: list[datetime.date] = []
    missing_by_day: dict[datetime.date, list[str]] = {}

    for day in days:
        missing = source.missing(cells=grid, day=day)
        if missing:
            missing_by_day[day] = missing
            logger.warning(
                "date_skipped_missing_cubes",
                day=day.isoformat(),
                missing=len(missing),
                of=len(grid),
                example=missing[0],
            )
        else:
            complete.append(day)

    if missing_by_day:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / MISSING_CUBES_FILENAME
        lines = [
            f"# Cubes absent from {source.cube_dir} — these days were SKIPPED (no COG written).",
            f"# {sum(len(v) for v in missing_by_day.values())} cube(s) across "
            f"{len(missing_by_day)} day(s).",
        ]
        for day, missing in missing_by_day.items():
            lines.append(f"\n[{day.isoformat()}]  {len(missing)} of {len(grid)} missing")
            lines.extend(f"  {name}" for name in missing)
        # The trailing "" is what emits the file's final newline. It lives inside the
        # join rather than as `+ "\n"` because `[tool.flynt] transform-concats` rewrites
        # that concat into an f-string with a backslash in the expression part — PEP 701
        # syntax that needs 3.12+, while pyproject pins requires-python >= 3.11.
        report.write_text("\n".join([*lines, ""]))

    return complete, missing_by_day


def _summarize(
    *,
    cogs: list[Path],
    missing_by_day: dict[datetime.date, list[str]],
    grid: list[GridCell],
    out_dir: Path,
) -> None:
    """Echo the end-of-run summary, repeating any skip so it is not lost in a long log."""
    typer.echo("\n==== RUN SUMMARY ====")
    typer.echo(f"Wrote {len(cogs)} COG(s) to {out_dir}")

    if not missing_by_day:
        return

    typer.echo(f"\nSKIPPED {len(missing_by_day)} date(s) for missing cubes:")
    for day, missing in missing_by_day.items():
        typer.echo(f"  {day.isoformat()}  ({len(missing)} of {len(grid)} missing)")
    typer.echo(f"\nMissing cubes listed in: {out_dir / MISSING_CUBES_FILENAME}")


@app.command()
def main(
    cube_config: Annotated[
        Path, typer.Option(help="Path to cube.yaml (sweep window/mode/roots).")
    ] = Path("configs/bow_valley/cube.yaml"),
    config: Annotated[
        Path, typer.Option(help="Path to inference.yaml (model run config).")
    ] = Path("configs/bow_valley/inference.yaml"),
    limit: Annotated[
        Optional[int],
        typer.Option(help="Cap the number of cells (smoke run); None = all in-AOI cells."),
    ] = None,
    cache_policy: Annotated[
        CachePolicy,
        typer.Option(
            help="How to treat an existing cube cache: 'prompt' (ask if non-empty; errors "
            "on a non-TTY), 'reuse' (keep it), or 'overwrite' (clear once up front). Use "
            "'overwrite' after an adapter or clip change that the version stamp can't catch.",
        ),
    ] = CachePolicy.PROMPT,
    cubes_only: Annotated[
        bool,
        typer.Option(
            "--cubes-only",
            help="Reuse the cubes already in cubes_dir instead of assembling them from the "
            "archive. Days whose cubes are incomplete are skipped (no partial COG).",
        ),
    ] = False,
    dates: Annotated[
        Optional[str],
        typer.Option(
            help="Comma-separated ISO days to infer, e.g. '2025-04-06,2025-04-10'. Need not "
            "be contiguous. Omit to use cube.yaml's window. Exclusive with --dates-file.",
        ),
    ] = None,
    dates_file: Annotated[
        Optional[Path],
        typer.Option(
            help="File of ISO days, one per line ('#' comments ignored). Exclusive with --dates.",
        ),
    ] = None,
    cube_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="--cubes-only: read cubes from here instead of the config's cubes_dir. For a "
            "run whose cubes live outside processing_root (e.g. full_run/cubes).",
        ),
    ] = None,
    read_workers: Annotated[
        int,
        typer.Option(
            help="Threads reading each batch's cubes (~180 ms/cube, the --cubes-only "
            "bottleneck). Throughput plateaus at ~4 and degrades past 8; 1 = serial.",
        ),
    ] = 4,
) -> None:
    """Run the daily FSC inference sweep and write one COG per day."""
    cube = CubeSettings.from_yaml(cube_config)
    infer = InferenceSettings.from_yaml(config)
    days = _resolve_days(dates=dates, dates_file=dates_file, cube=cube)

    grid = build_grid(mode=cube.mode, mode_b_inset_m=cube.mode_b_inset_m)
    if limit is not None:
        grid = grid[:limit]

    model = build_model(infer)
    out_dir = infer.out_dir if infer.out_dir is not None else cube.daily_fsc_dir

    if cube_dir is not None and not cubes_only:
        raise typer.BadParameter("--cube-dir only applies with --cubes-only.")

    exporter: LocalSourceExporter | PrebuiltCubeSource
    missing_by_day: dict[datetime.date, list[str]] = {}
    if cubes_only:
        # No cube cache is touched: nothing is assembled, so there is nothing to memoize
        # and nothing to prune. The driver finds no `_cache` on this source and skips it.
        source = PrebuiltCubeSource(cube_dir=cube_dir if cube_dir is not None else cube.cubes_dir)
        days, missing_by_day = _complete_days(source=source, grid=grid, days=days, out_dir=out_dir)
        if not days:
            _summarize(cogs=[], missing_by_day=missing_by_day, grid=grid, out_dir=out_dir)
            raise typer.BadParameter(
                f"No requested day has a complete set of {len(grid)} cubes in "
                f"{source.cube_dir}. Nothing to infer."
            )
        exporter = source
    else:
        # Resolve reuse/overwrite ONCE here in the parent before the exporter (and the
        # driver's worker pool) are built; they then reuse the resulting dir. The exporter
        # below is constructed with overwrite_cache=False (its default) — the clear, if any,
        # already happened.
        try:
            resolve_cache_policy(
                root=cube.cube_cache_dir,
                policy=cache_policy,
                max_entries=cube.cache_max_entries,
            )
        except CachePolicyError as exc:
            raise typer.BadParameter(str(exc)) from exc

        exporter = LocalSourceExporter(
            out_dir=cube.cubes_dir,
            placeholder=False,
            archive_root=cube.archive_root,
            cube_cache_dir=cube.cube_cache_dir,
            cache_max_entries=cube.cache_max_entries,
        )

    driver = InferenceGridDriver(
        exporter=exporter,  # type: ignore[arg-type]  # duck-typed .export (see PrebuiltCubeSource)
        model=model,
        grid=grid,
        days=days,
        out_dir=out_dir,
        device=infer.device,
        batch_size=infer.batch_size,
        export_workers=infer.export_workers,
        read_workers=read_workers,
    )
    logger.info(
        "inference_start",
        cells=len(grid),
        days=len(days),
        window=f"{days[0].isoformat()}..{days[-1].isoformat()}",
        cubes_only=cubes_only,
        out_dir=str(out_dir),
    )
    cogs = driver.run()
    logger.info("inference_complete", days=len(cogs), out_dir=str(out_dir))
    _summarize(cogs=cogs, missing_by_day=missing_by_day, grid=grid, out_dir=out_dir)


if __name__ == "__main__":
    app()
