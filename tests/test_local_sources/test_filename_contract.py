r"""Filename-contract tests for the local-source exporter (TASK-003, AC-9).

The ``LocalSourceExporter`` (TASK-004) writes one multiband GeoTIFF per
``(cell, window-end-day)`` whose **filename** must be parsed correctly by the
**unchanged** downstream loader ``LandsatEvalDataset.prediction_month_from_file``
(``src/snow_galileofsc/landsat_eval.py:171-181``). The filename is therefore a hard contract
between the new exporter and sacred downstream code.

Contract (PLAN §3 "Filename convention", SPEC FR-18 / AC-9):

    PR_{YYYYMMDD_window_end}_{LAT}_{LON}_SC00.tif

- regex ``^PR_\\d{8}_-?\\d+\\.\\d+_-?\\d+\\.\\d+_SC\\d+\\.tif$`` matches every name;
- ``prediction_month_from_file`` returns ``window_end.month`` (it reads the month
  from ``name.split("_")[1][4:6]`` on the ``PR`` branch).

The builder under test lives in :mod:`snow_galileo.data.local_sources.layout` so the
exporter and these tests share one definition (single source of truth — the
filename format is a layout concern, not exporter-internal).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from snow_galileo.data.local_sources.base import GridCell
from snow_galileo.data.local_sources.layout import (
    CUBE_FILENAME_REGEX,
    build_cube_filename,
    cell_centre_lat_lon,
)
from snow_galileo.fsc.landsat_eval import LandsatEvalDataset
from snow_galileo.inference.prebuilt import PrebuiltCubeSource

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


@pytest.fixture()
def bow_valley_cell() -> GridCell:
    """A real 1 km Bow Valley cell on the EPSG:32611 grid the exporter writes."""
    return GridCell.from_utm_bounds(
        cell_id=1, min_x=600000.0, min_y=5620000.0, max_x=601000.0, max_y=5621000.0
    )


def test_cell_centre_is_degrees_not_metres(bow_valley_cell: GridCell) -> None:
    """The centre comes back in the ±90/±180 band ``to_cartesian`` asserts on.

    ``landsat_eval.py`` parses these values out of the filename and feeds them to
    ``to_cartesian``, which builds the model's ``static_x`` location channels and
    rejects anything outside the degree range. Returning the cell's UTM metres here
    would fail there, far from the cause.
    """
    lat, lon = cell_centre_lat_lon(cell=bow_valley_cell)
    assert -90 <= lat <= 90
    assert -180 <= lon <= 180

    # Sanity-check the location itself, not just the range, against a value derived by
    # hand rather than from pyproj (asserting pyproj's own output back at it would prove
    # nothing). EPSG:32611's central meridian is -117 deg with a 500 km false easting, so
    # easting 600500 sits ~100.5 km east of it; at ~50.7N one degree of longitude spans
    # ~70.4 km, giving -117 + 100.5/70.4 ~= -115.57. Northing 5620500 is ~5620 km up from
    # the equator at ~111 km per degree ~= 50.7N.
    assert lat == pytest.approx(50.7, abs=0.2)
    assert lon == pytest.approx(-115.57, abs=0.2)


def test_prebuilt_reader_reconstructs_the_exporter_name(
    bow_valley_cell: GridCell, tmp_path: Path
) -> None:
    """``PrebuiltCubeSource.cube_name`` equals the name the exporter would write.

    This is the join key between a cube on disk and the cell it was exported for. If
    the two derivations ever diverge, every prebuilt cube becomes unfindable — so pin
    the agreement rather than trusting two copies of the arithmetic to stay in step.
    """
    window_end = date(2025, 4, 6)
    lat, lon = cell_centre_lat_lon(cell=bow_valley_cell)
    exporter_name = build_cube_filename(window_end=window_end, lat=lat, lon=lon)

    source = PrebuiltCubeSource(cube_dir=tmp_path)
    assert source.cube_name(cell=bow_valley_cell, window_end=window_end) == exporter_name
    assert re.match(CUBE_FILENAME_REGEX, exporter_name), exporter_name
