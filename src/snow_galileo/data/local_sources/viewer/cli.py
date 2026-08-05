"""Launch the Bow Valley data viewer with per-tab default folders set from the CLI.

A thin Typer wrapper around ``solara run <app module>``. ``solara run`` does not forward
arbitrary flags to the app, so the viewer reads its folder defaults from the ``VIEWER_*``
environment (pydantic-settings, see ``ViewerSettings``). This wrapper turns
``--clipped-root`` / ``--cubes-dir`` / ``--fsc-dir`` flags into those env vars, then execs
``solara run`` so the process *becomes* the server (signals / Ctrl-C behave normally).

Run via the launcher entrypoint::

    uv run python scripts/developer_scripts/bow_valley_inference_local/data_viewer.py \
        --cubes-dir /data/cubes \
        --fsc-dir /data/fsc \
        --clipped-root /data/clipped

Any trailing args after ``--`` are forwarded verbatim to ``solara run`` (e.g. host/port)::

    uv run python .../data_viewer.py --cubes-dir /data/cubes -- --port 8900
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

#: Dotted module path of the Solara app this wrapper launches. ``solara run`` imports it by
#: module path, so the app lives in the installed package (not a loose script file).
_APP = "snow_galileo.data.local_sources.viewer.app"

app = typer.Typer(add_completion=False, help=__doc__)


def _resolve(label: str, path: Path) -> str:
    """Resolve a folder flag to an absolute path string, warning if it is missing.

    A missing folder is not fatal — the viewer renders an empty tab with its in-app folder
    picker, so the user can still navigate from the launch default. We only warn.

    Args:
        label: Human label for the flag (for the warning).
        path: The folder the user passed.

    Returns:
        The absolute path as a string, suitable for an environment variable.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        typer.secho(
            f"warning: {label} does not exist: {resolved} (the tab will open empty)",
            fg=typer.colors.YELLOW,
            err=True,
        )
    return str(resolved)


@app.command()
def main(
    clipped_root: Annotated[
        Optional[Path],
        typer.Option("--clipped-root", help="Default folder for the Clip tab (clipped archive)."),
    ] = None,
    cubes_dir: Annotated[
        Optional[Path],
        typer.Option("--cubes-dir", help="Default folder for the Cube tab (PR_*.tif cubes)."),
    ] = None,
    fsc_dir: Annotated[
        Optional[Path],
        typer.Option("--fsc-dir", help="Default folder for the Daily FSC tab (fsc_*.tif COGs)."),
    ] = None,
    solara_args: Annotated[
        Optional[list[str]],
        typer.Argument(help="Extra args forwarded verbatim to `solara run` (e.g. --port 8900)."),
    ] = None,
) -> None:
    """Set the viewer's folder defaults from flags, then exec ``solara run``."""
    env = os.environ.copy()
    if clipped_root is not None:
        env["VIEWER_CLIPPED_ROOT"] = _resolve("--clipped-root", clipped_root)
    if cubes_dir is not None:
        env["VIEWER_CUBES_DIR"] = _resolve("--cubes-dir", cubes_dir)
    if fsc_dir is not None:
        env["VIEWER_DAILY_FSC_DIR"] = _resolve("--fsc-dir", fsc_dir)

    argv = ["solara", "run", _APP, *(solara_args or [])]
    typer.secho(f"launching: {' '.join(argv)}", fg=typer.colors.GREEN, err=True)
    # exec so this process *becomes* the Solara server — no orphaned wrapper, and Ctrl-C /
    # signals reach Solara directly.
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    app()
