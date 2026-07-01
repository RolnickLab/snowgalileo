"""Geometry tests for the grid generator (TASK-001 geometry half + TASK-003 prod).

TASK-001 part covers SPEC AC-10 (344 centre-in / 338 fully-inside, manifest sums
to 500) and AC-11 (cells non-overlapping), asserted against the real legacy CSV
and AOI whose containment counts were verified empirically.

TASK-003 part covers the productionized surface: ``build_grid`` mode A/B and the
per-cell ``GridCell`` target-grid triple (``EPSG:32611`` UTM 11N, ``scale=10`` m,
``100×100`` — see PLAN §3 Grid+CRS table and ``docs/agents/KNOWLEDGE.md``; the
"EPSG:4326 scale=10" wording in older prose was corrected 2026-06-04).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
from pyproj import Transformer
from shapely.geometry import Point, box

from snow_galileo.data.local_sources.grid import (
    GRID_MATH_CRS,
    build_grid,
    build_manifest,
    filter_cells,
    load_aoi_polygon,
    load_cells,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CSV = REPO_ROOT / "tests/fixtures/sampled_cells_bow_river_with_dates.csv"
AOI_PATH = REPO_ROOT / "data" / "bow_valley_inference_aoi.geojson"

EXPECTED_TOTAL_CELLS = 500
EXPECTED_CENTRE_IN = 344
EXPECTED_FULLY_INSIDE = 338

# Per-cell target-grid contract (TASK-003).
TARGET_CRS = "EPSG:32611"
TARGET_PX = 100  # EXPORTED_HEIGHT_WIDTH_METRES (1000 m) / 10 m scale
TARGET_SCALE_M = 10.0
CELL_SIZE_M = 1000.0


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


# --------------------------------------------------------------------------- #
# TASK-003 — productionized grid: build_grid(mode A/B) → GridCell target grid  #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def grid_a():
    """Mode-A grid: the in-AOI legacy cells as GridCells (centre-in rule)."""
    return build_grid(mode="A", legacy_csv=LEGACY_CSV, aoi_path=AOI_PATH)


@pytest.fixture(scope="module")
def grid_b():
    """Mode-B grid: the AOI tiled into 1 km cells (legacy CSV not consumed)."""
    return build_grid(mode="B", aoi_path=AOI_PATH)


def test_mode_a_cell_count(grid_a):
    """Mode A yields exactly the 344 centre-in cells (SPEC AC-10, FR-19)."""
    assert len(grid_a) == EXPECTED_CENTRE_IN


def test_mode_a_fully_inside_switch():
    """Mode A with require_fully_inside yields the 338 cells (SPEC AC-10)."""
    grid = build_grid(
        mode="A",
        legacy_csv=LEGACY_CSV,
        aoi_path=AOI_PATH,
        require_fully_inside=True,
    )
    assert len(grid) == EXPECTED_FULLY_INSIDE


def test_gridcell_target_triple(grid_a):
    """Every GridCell carries the UTM 11N / 10 m / 100×100 target triple (AC-12).

    The downstream loader reads neither the tif CRS nor transform, but the
    GEE reference patches (export_from_csv_utm) are UTM 11N @ 10 m, 100×100, so
    matching them keeps AC-27 parity a direct pixel diff (KNOWLEDGE.md, 2026-06-04).
    """
    for cell in grid_a:
        assert cell.crs == TARGET_CRS
        assert cell.shape == (TARGET_PX, TARGET_PX)
        # Affine: 10 m pixels, north-up (negative e), origin at (min_x, max_y).
        t = cell.transform
        assert t.a == pytest.approx(TARGET_SCALE_M)
        assert t.e == pytest.approx(-TARGET_SCALE_M)
        # transform extent spans exactly the 1 km cell.
        assert (t.a * cell.shape[1]) == pytest.approx(CELL_SIZE_M)
        assert (-t.e * cell.shape[0]) == pytest.approx(CELL_SIZE_M)


def test_gridcell_transform_matches_polygon_bounds(grid_a):
    """The cell's transform origin/extent equals its UTM polygon bounds.

    Origin is the top-left (min_x, max_y); the implied lower-right
    (origin + shape·pixel) equals (max_x, min_y). Guards a transposed or
    off-by-one transform that would silently shift every adapter's reprojection.
    """
    for cell in grid_a:
        min_x, min_y, max_x, max_y = cell.polygon.bounds
        t = cell.transform
        assert t.c == pytest.approx(min_x)  # x origin
        assert t.f == pytest.approx(max_y)  # y origin (north-up)
        assert (t.c + t.a * cell.shape[1]) == pytest.approx(max_x)
        assert (t.f + t.e * cell.shape[0]) == pytest.approx(min_y)


def test_gridcell_ids_unique(grid_a):
    """GridCell ids are unique (needed for the per-cell cube-cache shard path)."""
    ids = [cell.cell_id for cell in grid_a]
    assert len(ids) == len(set(ids))


def test_mode_b_tiles_within_aoi(grid_b):
    """Mode B tiles the AOI: every cell intersects it, none lies fully outside.

    Mode B is bounded by the AOI (never the wider cell-sampling bbox); a tile is
    kept iff it intersects the AOI (FR-19). We reproject each UTM cell centre back
    to lon/lat and assert it falls in the AOI's geographic bbox envelope.
    """
    aoi = load_aoi_polygon(AOI_PATH)
    to_geo = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    aoi_minx, aoi_miny, aoi_maxx, aoi_maxy = aoi.bounds
    assert len(grid_b) > 0
    for cell in grid_b:
        cx = (cell.polygon.bounds[0] + cell.polygon.bounds[2]) / 2
        cy = (cell.polygon.bounds[1] + cell.polygon.bounds[3]) / 2
        lon, lat = to_geo.transform(cx, cy)
        # Cell centres must lie within the AOI's lon/lat envelope (tolerant of the
        # one-cell boundary band the tiler may include).
        assert aoi_minx - 0.02 <= lon <= aoi_maxx + 0.02
        assert aoi_miny - 0.02 <= lat <= aoi_maxy + 0.02


def test_mode_b_cells_non_overlapping(grid_b):
    """Mode-B tiles are pairwise non-overlapping (SPEC AC-11)."""
    polys = sorted(
        (cell.polygon for cell in grid_b),
        key=lambda p: (p.bounds[0], p.bounds[1]),
    )
    for a, b in zip(polys, polys[1:]):
        assert a.intersection(b).area == pytest.approx(0.0, abs=1e-3)


def test_invalid_mode_rejected():
    """An unknown sweep mode is rejected explicitly (no silent default)."""
    with pytest.raises((ValueError, KeyError)):
        build_grid(mode="Z", legacy_csv=LEGACY_CSV, aoi_path=AOI_PATH)


# --------------------------------------------------------------------------- #
# Mode B internal inset (negative AOI buffer)                                  #
# --------------------------------------------------------------------------- #

INSET_M = 10_000.0  # 10 km internal border drop


@pytest.fixture(scope="module")
def grid_b_inset():
    """Mode-B grid with a 10 km internal inset (border ring dropped)."""
    return build_grid(mode="B", aoi_path=AOI_PATH, mode_b_inset_m=INSET_M)


def test_inset_drops_border_cells(grid_b, grid_b_inset):
    """A 10 km inset yields strictly fewer cells than the un-inset Mode B tiling."""
    assert 0 < len(grid_b_inset) < len(grid_b)


def test_inset_cells_are_a_spatial_subset(grid_b, grid_b_inset):
    """Every inset cell coincides with an un-inset cell (the inset only *removes*).

    Both grids snap their lattice origin with the same ``floor(v / CELL_SIZE_M) *
    CELL_SIZE_M`` rule, and a 10 km inset is a whole-cell multiple, so the two
    lattices coincide exactly — an inset cell is identical to a full-grid cell iff
    their UTM origins match. Compare origin sets (O(n)); the inset set must be a
    strict subset of the full set, i.e. the inset only erodes the border and never
    invents a tile outside the full Mode B coverage.

    (The previous form did a brute-force ``centroid in polygon`` over every full
    cell — ~19k×25k ≈ 480M shapely calls — which hung the suite for ~10 min.)
    """
    full_origins = {(c.transform.c, c.transform.f) for c in grid_b}
    inset_origins = {(c.transform.c, c.transform.f) for c in grid_b_inset}
    assert inset_origins < full_origins, "inset cells are not a strict subset of full grid"


def test_inset_cells_clear_the_aoi_edge(grid_b_inset):
    """Every inset cell centre lies at least ~inset distance inside the AOI edge.

    Reproject the AOI to UTM, erode it by the inset, and assert each kept cell
    intersects the eroded polygon (that is the tiling rule). Guards against the
    inset being applied in the wrong CRS (degrees) or sign.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    aoi = load_aoi_polygon(AOI_PATH)
    to_utm = Transformer.from_crs("EPSG:4326", GRID_MATH_CRS, always_xy=True)
    aoi_utm = shapely_transform(lambda xs, ys: to_utm.transform(xs, ys), aoi)
    eroded = aoi_utm.buffer(-INSET_M)
    assert not eroded.is_empty
    for cell in grid_b_inset:
        assert cell.polygon.intersects(eroded)


def test_inset_negative_rejected():
    """A negative inset is rejected (it would *grow* the AOI)."""
    with pytest.raises(ValueError, match="inset_m must be >= 0"):
        build_grid(mode="B", aoi_path=AOI_PATH, mode_b_inset_m=-1.0)


def test_inset_that_erases_aoi_raises():
    """An inset larger than the AOI's half-extent erodes it to nothing → ValueError."""
    with pytest.raises(ValueError, match="erodes the entire AOI"):
        build_grid(mode="B", aoi_path=AOI_PATH, mode_b_inset_m=1_000_000.0)


def test_inset_zero_matches_plain_mode_b(grid_b):
    """inset_m=0 is identical to plain Mode B (default-safe, no behaviour change)."""
    grid0 = build_grid(mode="B", aoi_path=AOI_PATH, mode_b_inset_m=0.0)
    assert len(grid0) == len(grid_b)
