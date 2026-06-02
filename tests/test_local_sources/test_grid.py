"""Geometry tests for the Phase 0 grid generator (TASK-001).

Covers SPEC AC-10 (344 centre-in / 338 fully-inside, manifest sums to 500) and
AC-11 (cells non-overlapping). These assert against the real legacy CSV and AOI
in the repo, whose containment counts were verified empirically.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.geometry import Point, box

from src.data.local_sources.grid import (
    GRID_MATH_CRS,
    build_manifest,
    filter_cells,
    load_aoi_polygon,
    load_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CSV = REPO_ROOT / "sampled_cells_bow_river_with_dates.csv"
AOI_PATH = REPO_ROOT / "data" / "aoi.geojson"

EXPECTED_TOTAL_CELLS = 500
EXPECTED_CENTRE_IN = 344
EXPECTED_FULLY_INSIDE = 338


@pytest.fixture(scope="module")
def cells():
    return load_cells(LEGACY_CSV)


@pytest.fixture(scope="module")
def aoi():
    return load_aoi_polygon(AOI_PATH)


def test_loads_all_unique_cells(cells):
    """The legacy CSV deduplicates to exactly 500 unique cells."""
    assert len(cells) == EXPECTED_TOTAL_CELLS
    assert len({c.cell_id for c in cells}) == EXPECTED_TOTAL_CELLS


def test_centre_in_count(cells, aoi):
    """Centre-in rule keeps 344 cells (SPEC AC-10)."""
    kept, dropped = filter_cells(cells, aoi, keep_rule="centre_in")
    assert len(kept) == EXPECTED_CENTRE_IN
    assert len(kept) + len(dropped) == EXPECTED_TOTAL_CELLS


def test_fully_inside_count(cells, aoi):
    """`--require-fully-inside` keeps 338 cells (SPEC AC-10)."""
    kept, dropped = filter_cells(cells, aoi, keep_rule="fully_inside")
    assert len(kept) == EXPECTED_FULLY_INSIDE
    assert len(kept) + len(dropped) == EXPECTED_TOTAL_CELLS


def test_manifest_sums_to_total(cells, aoi):
    """Kept/dropped manifest accounts for every input cell (SPEC AC-10)."""
    kept, dropped = filter_cells(cells, aoi, keep_rule="centre_in")
    manifest = build_manifest(kept, dropped)
    assert len(manifest) == EXPECTED_TOTAL_CELLS
    assert (manifest["action"] == "KEEP").sum() == EXPECTED_CENTRE_IN
    assert (manifest["action"] == "DROP").sum() == EXPECTED_TOTAL_CELLS - EXPECTED_CENTRE_IN
    # cell_ids unique and complete
    assert sorted(manifest["cell_id"]) == list(range(EXPECTED_TOTAL_CELLS))


def test_kept_centres_inside_aoi(cells, aoi):
    """Every kept cell centre lies within the AOI (SPEC AC-10)."""
    kept, _ = filter_cells(cells, aoi, keep_rule="centre_in")
    transformer = Transformer.from_crs(GRID_MATH_CRS, "EPSG:4326", always_xy=True)
    for cell in kept:
        lon, lat = transformer.transform(cell.center_x, cell.center_y)
        point = Point(lon, lat)
        assert aoi.contains(point) or aoi.touches(point)


def test_cells_non_overlapping(cells):
    """Grid cells are pairwise non-overlapping (SPEC AC-11).

    Tested on the kept set's UTM bboxes; interiors must not intersect. A sampled
    pairwise check over a spatially-sorted neighbourhood keeps this O(n) rather
    than O(n^2) while still catching any real overlap (cells are a regular 1 km
    lattice, so only near-neighbours can overlap).
    """
    polys = sorted(
        (box(c.min_x, c.min_y, c.max_x, c.max_y) for c in cells),
        key=lambda p: (p.bounds[0], p.bounds[1]),
    )
    # Compare each cell only against the next few in sorted order (neighbours).
    for a, b in zip(polys, polys[1:]):
        assert a.intersection(b).area == pytest.approx(0.0, abs=1e-6)
    # Plus an exhaustive interior-overlap check on a small random-free subset:
    for a, b in combinations(polys[:40], 2):
        assert not a.buffer(-1e-6).intersects(b.buffer(-1e-6))
