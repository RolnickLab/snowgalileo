from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from snow_galileo.data.local_sources.grid import KeepRule, generate
from snow_galileo.data.local_sources.paths import LocalPaths

#: Default inference window (PLAN §3 Temporal window), inclusive of both ends.
DEFAULT_WINDOW_START: date = date(2025, 4, 6)
DEFAULT_WINDOW_END: date = date(2025, 5, 28)

#: Repo-root-relative default paths (resolved against the package's repo root).
DEFAULT_LEGACY_CSV: Path = Path("tests/fixtures/sampled_cells_bow_river_with_dates.csv")
#: AOI default resolves from LocalPaths (env-overridable, LOCAL_ prefix) so the
#: grid generator and the clip stage share one source of truth — see
#: data/BOW_VALLEY_DATA_LAYOUT.md.
DEFAULT_AOI_PATH: Path = LocalPaths().aoi_path
DEFAULT_OUTPUT_CSV: Path = Path("configs/bow_valley/cube_cells.csv")
DEFAULT_MANIFEST_PATH: Path = Path("configs/bow_valley/cell_filter_manifest.csv")


def generate_grid_csv(
    legacy_csv: Annotated[
        Path, typer.Option(help="Legacy cell-sampling CSV.")
    ] = DEFAULT_LEGACY_CSV,
    aoi_path: Annotated[Path, typer.Option("--aoi", help="AOI GeoJSON.")] = DEFAULT_AOI_PATH,
    output_csv: Annotated[
        Path, typer.Option(help="Generated cube CSV output.")
    ] = DEFAULT_OUTPUT_CSV,
    manifest_path: Annotated[
        Path, typer.Option(help="Kept/dropped manifest output.")
    ] = DEFAULT_MANIFEST_PATH,
    require_fully_inside: Annotated[
        bool,
        typer.Option("--require-fully-inside", help="Keep only fully-contained cells (→ 338)."),
    ] = False,
    window_start: Annotated[
        str, typer.Option(help="First inference day, YYYY-MM-DD.")
    ] = DEFAULT_WINDOW_START.isoformat(),
    window_end: Annotated[
        str, typer.Option(help="Last inference day, YYYY-MM-DD.")
    ] = DEFAULT_WINDOW_END.isoformat(),
) -> None:
    """Emit the generated cube CSV and the kept/dropped cell manifest."""
    keep_rule: KeepRule = "fully_inside" if require_fully_inside else "centre_in"
    cube_csv = generate(
        cube_cells_csv=legacy_csv,
        aoi_path=aoi_path,
        output_csv=output_csv,
        manifest_path=manifest_path,
        keep_rule=keep_rule,
        window_start=date.fromisoformat(window_start),
        window_end=date.fromisoformat(window_end),
    )
    typer.echo(f"Wrote {len(cube_csv)} rows to {output_csv}")


def main() -> None:
    """CLI entry point (single command — emits the cube CSV + manifest)."""
    typer.run(generate_grid_csv)


if __name__ == "__main__":
    main()
