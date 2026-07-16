"""Cube source that *finds* an already-exported cube instead of building one.

:class:`PrebuiltCubeSource` satisfies the same informal contract
:class:`~snow_galileo.inference.driver.InferenceGridDriver` already requires of its
injected exporter — ``export(*, cell, window_end) -> Path`` — but resolves the path
against a directory of cubes written by an earlier run rather than assembling one
from the archive. Injecting it lets the driver re-run inference (new checkpoint, new
threshold, new mosaic fix) over the ~84 GB of cubes already on disk without paying the
export cost again.

The driver needs no special case for it: ``run()`` finds no ``_cache`` attribute (so it
skips the day-frontier prune) and ``_pre_export_day`` sees a non-:class:`LocalSourceExporter`
(so it takes the serial ``.export()`` fallback) — both already documented degradations.

**The filename is reconstructed, never globbed.** The exporter names each cube
``build_cube_filename(window_end, lat, lon)`` with ``lat``/``lon`` from
``layout.cell_centre_lat_lon(cell)``; calling that same shared derivation here is the
join key, so a cube can only be matched to the cell it was actually exported for.
Globbing by date would return files, not cells, and could silently pair a cube with the
wrong cell.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import structlog

from snow_galileo.data.local_sources.base import GridCell
from snow_galileo.data.local_sources.layout import build_cube_filename, cell_centre_lat_lon

logger = structlog.get_logger(__name__)


class CubeNotFoundError(FileNotFoundError):
    """A cube for a requested ``(cell, day)`` is absent from the cube directory."""


class PrebuiltCubeSource:
    """Resolves ``(cell, day)`` to a cube already on disk. Never writes.

    Args:
        cube_dir: Directory holding the cubes an earlier export wrote.
    """

    def __init__(self, *, cube_dir: Path) -> None:
        self.cube_dir = cube_dir

    def cube_name(self, *, cell: GridCell, window_end: datetime.date) -> str:
        """Return the filename the exporter wrote for this ``(cell, day)``.

        Composes the exporter's own two steps — :func:`~snow_galileo.data.local_sources.
        layout.cell_centre_lat_lon` then :func:`~snow_galileo.data.local_sources.layout.
        build_cube_filename` — so the name can only drift from what is on disk if the
        exporter's own naming drifts with it.

        Args:
            cell: The grid cell.
            window_end: The 8-day window's end (prediction) day.

        Returns:
            The cube's filename, e.g. ``PR_20250406_50.7306_-116.3218_SC00.tif``.
        """
        lat, lon = cell_centre_lat_lon(cell=cell)
        return build_cube_filename(window_end=window_end, lat=lat, lon=lon)

    def missing(self, *, cells: list[GridCell], day: datetime.date) -> list[str]:
        """Return the filenames absent from :pyattr:`cube_dir` for one day.

        The pre-flight the caller runs *before* inference, so an incomplete day is skipped
        whole rather than mosaicked with an invisible hole in it.

        Args:
            cells: Every cell the day's mosaic needs.
            day: The window-end (prediction) day.

        Returns:
            The missing filenames (empty when the day is complete).
        """
        return [
            name
            for cell in cells
            if not (self.cube_dir / (name := self.cube_name(cell=cell, window_end=day))).exists()
        ]

    def export(self, *, cell: GridCell, window_end: datetime.date) -> Path:
        """Return the existing cube for one ``(cell, day)``.

        Named ``export`` to satisfy the driver's injected-exporter contract; it builds
        nothing.

        Args:
            cell: The grid cell.
            window_end: The 8-day window's end (prediction) day.

        Returns:
            Path to the cube on disk.

        Raises:
            CubeNotFoundError: If the cube is absent. Callers pre-flight with
                :meth:`missing`, so reaching this means the directory changed mid-run.
        """
        cube = self.cube_dir / self.cube_name(cell=cell, window_end=window_end)
        if not cube.exists():
            raise CubeNotFoundError(
                f"No prebuilt cube {cube.name} in {self.cube_dir} "
                f"(cell {cell.cell_id}, day {window_end.isoformat()})."
            )
        return cube
