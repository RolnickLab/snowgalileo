"""Launcher entrypoint for the Bow Valley data viewer.

The viewer (Solara app + its launch CLI) now lives in the installed package under
``snow_galileo.data.local_sources.viewer`` (``app.py`` / ``cli.py``). This script is a thin
shim so the long-standing developer path keeps working::

    uv run python scripts/developer_scripts/bow_valley_inference_local/data_viewer.py \
        --cubes-dir /data/cubes --fsc-dir /data/fsc --clipped-root /data/clipped

It delegates to the Typer CLI, which sets the ``VIEWER_*`` folder defaults from the flags
and execs ``solara run`` on the app module. See ``viewer/cli.py`` for the full flag list.
"""

from __future__ import annotations

from snow_galileo.data.local_sources.viewer.cli import app

if __name__ == "__main__":
    app()
