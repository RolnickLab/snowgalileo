"""Tests for the viewer output-listing + cube band-catalogue helpers (PLAN-V2 §8).

Pure functions over tiny synthetic GeoTIFFs — no Solara/leafmap, no real archive. The
Solara components themselves are smoke-tested manually via ``solara run`` (PLAN §8 style).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from snow_galileo.data.local_sources.viewer.outputs import (
    CubeRow,
    band_index,
    cube_availability,
    cube_variables,
    cubes_for_date,
    dates_for_cubes,
    list_cubes,
    list_fsc,
    timesteps_for_var,
    vars_at_timestep,
)
from snow_galileo.data.local_sources.viewer.settings import ViewerSettings

# A small EPSG:32611 grid (UTM 11N), matching the production cube CRS.
_TRANSFORM = Affine(10.0, 0.0, 547_000.0, 0.0, -10.0, 5_620_000.0)


def _settings(processing_root: Path) -> ViewerSettings:
    return ViewerSettings(processing_root=processing_root)  # type: ignore[call-arg]


def _write_cube(path: Path, descriptions: list[str]) -> None:
    """Write a tiny multi-band 32611 GeoTIFF with the given band descriptions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(descriptions)
    data = np.zeros((count, 4, 4), dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=count,
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
        nodata=-9999.0,
    ) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)


def _cube_descriptions(dynamic: list[str], statics: list[str], n_ts: int) -> list[str]:
    descs = [f"{var}_t{t}" for t in range(n_ts) for var in dynamic]
    return [*descs, *statics]


def _write_fsc(path: Path, value: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, 4, 4), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
        nodata=-9999.0,
    ) as dst:
        dst.write(data)


# --------------------------------------------------------------------------- #
# list_cubes / list_fsc
# --------------------------------------------------------------------------- #


def test_list_cubes_parses_and_sorts(tmp_path: Path) -> None:
    cubes = tmp_path / "cubes"
    # Two dates, out of order on disk; one bogus name that must be skipped.
    _write_cube(cubes / "PR_20250519_50.7306_-116.3218_SC00.tif", ["DEM"])
    _write_cube(cubes / "PR_20250406_50.1000_-115.0000_SC00.tif", ["DEM"])
    _write_cube(cubes / "PR_20250519_50.1000_-116.0000_SC00.tif", ["DEM"])
    (cubes / "not_a_cube.tif").write_bytes(b"junk")

    rows = list_cubes(_settings(tmp_path))

    assert [r.pred_date for r in rows] == [
        date(2025, 4, 6),
        date(2025, 5, 19),
        date(2025, 5, 19),
    ]
    # Within the 2025-05-19 group, sorted by (lat, lon).
    assert rows[1].lat == 50.1 and rows[2].lat == 50.7306
    assert rows[0].lon == -115.0
    assert rows[0].cell_label == "50.1000, -115.0000"


def test_list_cubes_empty_when_dir_absent(tmp_path: Path) -> None:
    assert list_cubes(_settings(tmp_path)) == []


def test_list_fsc_parses_and_sorts(tmp_path: Path) -> None:
    fsc = tmp_path / "daily_fsc"
    _write_fsc(fsc / "fsc_20250519.tif")
    _write_fsc(fsc / "fsc_20250406.tif")
    (fsc / "fsc_bogus.tif").write_bytes(b"junk")

    rows = list_fsc(_settings(tmp_path))

    assert [r.pred_date for r in rows] == [date(2025, 4, 6), date(2025, 5, 19)]


def test_list_fsc_empty_when_dir_absent(tmp_path: Path) -> None:
    assert list_fsc(_settings(tmp_path)) == []


def test_date_grouping_helpers() -> None:
    rows = [
        CubeRow(Path("a"), date(2025, 5, 19), 50.1, -116.0),
        CubeRow(Path("b"), date(2025, 4, 6), 50.2, -115.0),
        CubeRow(Path("c"), date(2025, 5, 19), 50.3, -114.0),
    ]
    assert dates_for_cubes(rows) == [date(2025, 4, 6), date(2025, 5, 19)]
    in_may = cubes_for_date(rows, date(2025, 5, 19))
    assert [r.lat for r in in_may] == [50.1, 50.3]


# --------------------------------------------------------------------------- #
# cube_variables / band_index
# --------------------------------------------------------------------------- #


def test_cube_variables_catalogues_dynamic_statics_timesteps(tmp_path: Path) -> None:
    dynamic = ["VV", "VH", "B2", "temperature_2m"]
    statics = ["DEM", "slope", "aspect", "Map"]
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, _cube_descriptions(dynamic, statics, n_ts=8))

    bands = cube_variables(path)

    assert bands.dynamic == dynamic  # first-seen order preserved
    assert bands.statics == statics
    assert bands.n_timesteps == 8


def test_cube_variables_raises_without_descriptions(tmp_path: Path) -> None:
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=2,
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(np.zeros((2, 4, 4), dtype="float32"))

    import pytest

    with pytest.raises(ValueError, match="no band descriptions"):
        cube_variables(path)


def test_band_index_resolves_by_description(tmp_path: Path) -> None:
    dynamic = ["VV", "VH", "B2"]
    statics = ["DEM", "Map"]
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, _cube_descriptions(dynamic, statics, n_ts=8))
    # Layout: [VV_t0, VH_t0, B2_t0, VV_t1, ...]; 3 dyn × 8 ts = 24 dynamic, then statics.
    assert band_index(path, var="VV", timestep=0) == 1
    assert band_index(path, var="B2", timestep=0) == 3
    assert band_index(path, var="VV", timestep=1) == 4
    assert band_index(path, var="B2", timestep=7) == 24
    # Statics: matched by exact name, timestep ignored.
    assert band_index(path, var="DEM", timestep=0) == 25
    assert band_index(path, var="Map", timestep=3) == 26


def test_band_index_raises_on_unknown_var(tmp_path: Path) -> None:
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, _cube_descriptions(["VV"], ["DEM"], n_ts=8))

    import pytest

    with pytest.raises(KeyError, match="nope"):
        band_index(path, var="nope", timestep=0)


# --------------------------------------------------------------------------- #
# cube_availability / vars_at_timestep / timesteps_for_var
# --------------------------------------------------------------------------- #


def _write_cube_with_fill(
    path: Path,
    *,
    dynamic: list[str],
    statics: list[str],
    n_ts: int,
    real: dict[str, set[int]],
) -> None:
    """Write a cube where each dynamic ``(var, t)`` band is all-nodata unless real.

    ``real[var]`` is the set of timesteps at which ``var``'s band carries real data (here a
    constant ``1.0``); every other dynamic band is filled with the ``-9999`` nodata. Static
    bands are written real.
    """
    descriptions = _cube_descriptions(dynamic, statics, n_ts)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodata = -9999.0
    arrays: list[np.ndarray] = []
    for t in range(n_ts):
        for var in dynamic:
            fill = 1.0 if t in real.get(var, set()) else nodata
            arrays.append(np.full((4, 4), fill, dtype="float32"))
    for _ in statics:
        arrays.append(np.full((4, 4), 1.0, dtype="float32"))
    data = np.stack(arrays, axis=0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=len(descriptions),
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
        nodata=nodata,
    ) as dst:
        dst.write(data)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)


def test_cube_availability_marks_real_timesteps(tmp_path: Path) -> None:
    dynamic = ["VV", "B2", "B2_landsat"]
    statics = ["DEM", "Map"]
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    # VV all-nodata everywhere; B2 (S2) real at t4,t6; Landsat real at t3 only.
    _write_cube_with_fill(
        path,
        dynamic=dynamic,
        statics=statics,
        n_ts=8,
        real={"B2": {4, 6}, "B2_landsat": {3}},
    )

    avail = cube_availability(path)

    assert avail.dynamic_order == dynamic
    assert avail.statics == statics
    assert avail.n_timesteps == 8
    assert avail.dynamic_real["VV"] == set()
    assert avail.dynamic_real["B2"] == {4, 6}
    assert avail.dynamic_real["B2_landsat"] == {3}


def test_vars_at_timestep_filters_dynamic_keeps_statics(tmp_path: Path) -> None:
    dynamic = ["VV", "B2", "B2_landsat"]
    statics = ["DEM", "Map"]
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube_with_fill(
        path,
        dynamic=dynamic,
        statics=statics,
        n_ts=8,
        real={"B2": {4, 6}, "B2_landsat": {3}},
    )
    avail = cube_availability(path)

    # t3: only Landsat is real; statics always appended.
    assert vars_at_timestep(avail, 3) == ["B2_landsat", "DEM", "Map"]
    # t4: only S2 is real.
    assert vars_at_timestep(avail, 4) == ["B2", "DEM", "Map"]
    # t0: no dynamic var is real → statics only.
    assert vars_at_timestep(avail, 0) == ["DEM", "Map"]


def test_timesteps_for_var_real_only_and_statics_empty(tmp_path: Path) -> None:
    dynamic = ["VV", "B2", "B2_landsat"]
    statics = ["DEM", "Map"]
    path = tmp_path / "cubes" / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube_with_fill(
        path,
        dynamic=dynamic,
        statics=statics,
        n_ts=8,
        real={"B2": {4, 6}, "B2_landsat": {3}},
    )
    avail = cube_availability(path)

    assert timesteps_for_var(avail, "B2") == [4, 6]
    assert timesteps_for_var(avail, "B2_landsat") == [3]
    assert timesteps_for_var(avail, "VV") == []  # all-nodata var
    assert timesteps_for_var(avail, "DEM") == []  # static: no timestep axis
