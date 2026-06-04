"""Filename-contract tests for the local-source exporter (TASK-003, AC-9).

The ``LocalSourceExporter`` (TASK-004) writes one multiband GeoTIFF per
``(cell, window-end-day)`` whose **filename** must be parsed correctly by the
**unchanged** downstream loader ``LandsatEvalDataset.prediction_month_from_file``
(``src/fsc/landsat_eval.py:171-181``). The filename is therefore a hard contract
between the new exporter and sacred downstream code.

Contract (PLAN §3 "Filename convention", SPEC FR-18 / AC-9):

    PR_{YYYYMMDD_window_end}_{LAT}_{LON}_SC00.tif

- regex ``^PR_\\d{8}_-?\\d+\\.\\d+_-?\\d+\\.\\d+_SC\\d+\\.tif$`` matches every name;
- ``prediction_month_from_file`` returns ``window_end.month`` (it reads the month
  from ``name.split("_")[1][4:6]`` on the ``PR`` branch).

The builder under test lives in :mod:`src.data.local_sources.layout` so the
exporter and these tests share one definition (single source of truth — the
filename format is a layout concern, not exporter-internal).
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from src.data.local_sources.layout import CUBE_FILENAME_REGEX, build_cube_filename
from src.fsc.landsat_eval import LandsatEvalDataset

#: Synthetic (window_end, lat, lon) triples spanning the contract's edge cases:
#: every month digit pair, both hemispheres of longitude (signed), and the
#: archive's real Bow Valley latitude band (~50-52 N).
CASES: list[tuple[date, float, float]] = [
    (date(2025, 1, 5), 51.1234, -115.6789),
    (date(2025, 4, 6), 50.7298, -116.5619),  # default window start
    (date(2025, 5, 28), 52.3067, -114.5277),  # default window end
    (date(2025, 9, 30), 51.0001, -115.0001),
    (date(2025, 10, 1), 50.5121, 114.0104),  # positive lon (regex coverage)
    (date(2025, 12, 22), 52.0046, -116.7408),
]


@pytest.fixture(scope="module")
def parser() -> LandsatEvalDataset:
    """A ``LandsatEvalDataset`` whose only used method is the pure filename parser.

    ``prediction_month_from_file`` touches no instance state beyond ``tif_path``,
    so we bypass the data-folder-dependent ``__init__`` with
    ``__new__`` — this exercises the *real* downstream method, not a reimplementation.
    """
    return LandsatEvalDataset.__new__(LandsatEvalDataset)


@pytest.mark.parametrize(("window_end", "lat", "lon"), CASES)
def test_filename_matches_regex(window_end: date, lat: float, lon: float) -> None:
    """Every emitted filename matches the PR-prefix contract regex (AC-9)."""
    name = build_cube_filename(window_end=window_end, lat=lat, lon=lon)
    assert re.match(CUBE_FILENAME_REGEX, name), name
    assert name.startswith("PR_")
    assert name.endswith("_SC00.tif")


@pytest.mark.parametrize(("window_end", "lat", "lon"), CASES)
def test_prediction_month_roundtrips(
    parser: LandsatEvalDataset, window_end: date, lat: float, lon: float
) -> None:
    """Downstream parser recovers ``window_end.month`` from the filename (AC-9)."""
    from pathlib import Path

    name = build_cube_filename(window_end=window_end, lat=lat, lon=lon)
    assert parser.prediction_month_from_file(Path(name)) == window_end.month


@pytest.mark.parametrize(("window_end", "lat", "lon"), CASES)
def test_lat_lon_recoverable(window_end: date, lat: float, lon: float) -> None:
    """Lat/lon parse back from ``parts[3]``/``parts[4]`` (the loader's PR branch).

    ``_tif_to_array`` reads coords at ``parts[2]``/``parts[3]`` for the non-Landsat
    branch but the ``PR``-prefixed names take the Landsat branch
    (``parts[0].startswith("LC"/"LE")`` is False, so it falls to the else at
    ``landsat_eval.py:265`` reading ``parts[2]``/``parts[3]``). We assert the
    coordinates survive the round-trip so a future exporter cannot silently
    transpose them.
    """
    name = build_cube_filename(window_end=window_end, lat=lat, lon=lon)
    parts = name[: -len(".tif")].split("_")
    # PR _ YYYYMMDD _ LAT _ LON _ SC00
    assert parts[0] == "PR"
    assert float(parts[2]) == pytest.approx(lat)
    assert float(parts[3]) == pytest.approx(lon)
