"""AOI 1 km grid generator — Phase 0 geometry half.

This module ships the *geometry half* of the grid generator required by Phase 0
(TASK-001): load a cells CSV for **cell geometry only**, filter cells to an AOI,
emit a kept/dropped manifest, and emit the generated cross-product cube CSV that
drives both the inference sweep and the Phase 0 GEE reference-patch run.

There are two ways to obtain the cells, and they take different inputs:

- :func:`cells_from_csv` (**mode A**) — the cell list comes from a CSV; an AOI, if
  given, only *filters* it. Used for training/validation sampling.
- :func:`cells_from_aoi` (**mode B**) — an AOI is tiled into a seamless 1 km
  lattice; no CSV exists. Used for large-scale inference maps.

:func:`build_cells` / :func:`build_grid` dispatch between them for callers whose
mode arrives from ``cube.yaml``. Call the two directly when the mode is static.

**One CRS only.** Everything here is welded to :data:`GRID_MATH_CRS` (UTM 11N);
a cells CSV in another zone is rejected by :func:`load_cells`. Generalizing this
is parked — see
``docs/agents/planning/upcoming/TASK-generalize-grid-crs.md``.

Key contracts (verified against the codebase, see
``docs/agents/planning/bow_valley/020-data-ingestion/``):

- :func:`build_cube_csv` emits the **canonical UTM dialect** —
  ``date, crs, center_x, center_y, min_x, min_y, max_x, max_y`` — matching this
  module's :class:`CellGeometry` vocabulary and the legacy sampling CSV that
  :func:`load_cells` reads. Cell centres are UTM eastings/northings.
- ``EarthEngineExporterEval.export_from_csv_utm`` instead reads
  ``center_lat`` / ``center_lon`` (it uses them only to name the output tif, in
  decimal degrees). :func:`build_cube_dataframe_for_gee_utm` is the adapter that emits
  that dialect — identical UTM bounds, but with the centre reprojected to true
  degrees. Do **not** feed a :func:`build_cube_csv` frame straight to that reader
  (``KeyError`` on the missing ``center_lat`` column).
- Cell geometry stays in its native ``EPSG:32611`` (UTM 11N) in the CSV — the GEE
  exporter reprojects to 4326 itself. The AOI filter reprojects only the cell
  *centre* (and, for ``--require-fully-inside``, the corners) to 4326 for the
  point-in-polygon test. CRS is law: every transform is explicit.
- DataFrames use ``pandas`` (project standard; the GEE exporter reads the CSV via
  ``pd.read_csv``). See ``docs/agents/KNOWLEDGE.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform as shapely_transform

from snow_galileo.data.config import EXPORTED_HEIGHT_WIDTH_METRES
from snow_galileo.data.local_sources.base import GridCell

logger = structlog.get_logger(__name__)

#: Side length of one grid cell, in :data:`GRID_MATH_CRS` metres (1000 m).
CELL_SIZE_M: float = float(EXPORTED_HEIGHT_WIDTH_METRES)

SweepMode = Literal["A", "B"]

# --- Fixed contracts -------------------------------------------------------

#: Canonical UTM cube-CSV schema (:func:`build_cube_csv`). Centre columns are UTM
#: eastings/northings, matching :class:`CellGeometry` and the legacy sampling CSV.
CUBE_CSV_COLUMNS: list[str] = [
    "date",
    "crs",
    "center_x",
    "center_y",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
]

#: GEE UTM-reader schema (:func:`build_cube_dataframe_for_gee_utm`). Differs from
#: :data:`CUBE_CSV_COLUMNS` only in the centre columns: ``center_lat`` / ``center_lon``
#: in decimal degrees (what ``export_from_csv_utm`` reads to name the output tif).
GEE_UTM_CSV_COLUMNS: list[str] = [
    "date",
    "crs",
    "center_lat",
    "center_lon",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
]

# TODO Generalize CRS management for other regions

#: CRS the legacy cells (and therefore the generated CSV) are expressed in.
GRID_MATH_CRS: str = "EPSG:32611"

#: Geographic CRS used only for the AOI point-in-polygon test.
GEOGRAPHIC_CRS: str = "EPSG:4326"

KeepRule = Literal["centre_in", "fully_inside"]


@dataclass(frozen=True)
class CellGeometry:
    """Geometry of one sampling cell, in ``EPSG:32611`` metres.

    Attributes:
        cell_id: Stable identifier (row order in the deduplicated legacy CSV).
        center_x: Cell-centre easting (UTM 11N metres).
        center_y: Cell-centre northing (UTM 11N metres).
        min_x: Cell western bound (UTM 11N metres).
        min_y: Cell southern bound (UTM 11N metres).
        max_x: Cell eastern bound (UTM 11N metres).
        max_y: Cell northern bound (UTM 11N metres).
    """

    cell_id: int
    center_x: float
    center_y: float
    min_x: float
    min_y: float
    max_x: float
    max_y: float


def load_aoi_polygon(aoi_path: Path) -> Polygon:
    """Load the AOI boundary polygon from a GeoJSON file.

    Args:
        aoi_path: Path to ``bow_valley_inference_aoi.geojson`` (a single ``Polygon`` feature in
            CRS84 / ``EPSG:4326`` lon/lat order).

    Returns:
        The AOI as a shapely :class:`~shapely.geometry.Polygon` in lon/lat.

    Raises:
        ValueError: If the GeoJSON does not contain a single ``Polygon``.
    """
    raw = json.loads(aoi_path.read_text())
    features = raw.get("features", [])
    if len(features) != 1:
        raise ValueError(
            f"AOI {aoi_path} must contain exactly one feature, found {len(features)}."
        )
    geometry = features[0]["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError(f"AOI geometry must be a Polygon, found {geometry['type']!r}.")
    return Polygon(geometry["coordinates"][0])


def load_cells(cube_cells_csv: Path) -> list[CellGeometry]:
    """Load cell geometry from a cells CSV (geometry only).

    The CSV's ``date`` column is train/eval label-sampling metadata and is
    **not** read here (see PLAN §8 Q4). Rows are deduplicated on their full
    geometry so a cell sampled on multiple label dates is counted once.

    Args:
        cube_cells_csv: Path to a cells CSV in the :data:`CUBE_CSV_COLUMNS`
            geometry dialect, e.g.
            ``tests/fixtures/sampled_cells_bow_river_with_dates.csv``.

    Returns:
        One :class:`CellGeometry` per unique cell, ``cell_id`` assigned in
        stable row order.

    Raises:
        ValueError: If any cell CRS is not :data:`GRID_MATH_CRS`. Generalizing
            this beyond UTM 11N is parked — see
            ``docs/agents/planning/upcoming/TASK-generalize-grid-crs.md``.
    """
    df = pd.read_csv(cube_cells_csv)
    geom_cols = ["center_x", "center_y", "min_x", "min_y", "max_x", "max_y"]
    cells_df = df.drop_duplicates(subset=geom_cols).reset_index(drop=True)

    bad_crs = set(cells_df["crs"].unique()) - {GRID_MATH_CRS}
    if bad_crs:
        raise ValueError(f"Legacy CSV cells must be {GRID_MATH_CRS}; found unexpected {bad_crs}.")

    cells = [
        CellGeometry(
            cell_id=int(idx),
            center_x=float(row["center_x"]),
            center_y=float(row["center_y"]),
            min_x=float(row["min_x"]),
            min_y=float(row["min_y"]),
            max_x=float(row["max_x"]),
            max_y=float(row["max_y"]),
        )
        for idx, row in cells_df.iterrows()
    ]
    logger.info("loaded_cells", count=len(cells), source=str(cube_cells_csv))
    return cells


def _make_transformer() -> Transformer:
    """Build the ``EPSG:32611`` → ``EPSG:4326`` transformer (lon/lat order)."""
    return Transformer.from_crs(GRID_MATH_CRS, GEOGRAPHIC_CRS, always_xy=True)


def _centre_in_aoi(cell: CellGeometry, aoi: Polygon, transformer: Transformer) -> bool:
    """Return ``True`` if the cell *centre* lies within (or on) the AOI."""
    lon, lat = transformer.transform(cell.center_x, cell.center_y)
    point = Point(lon, lat)
    return aoi.contains(point) or aoi.touches(point)


def _fully_inside_aoi(cell: CellGeometry, aoi: Polygon, transformer: Transformer) -> bool:
    """Return ``True`` if every cell corner lies within the AOI.

    The cell bbox is reprojected corner-by-corner to lon/lat; UTM→geographic
    bends straight edges only slightly at this latitude, so the reprojected
    bounding box is a safe conservative envelope for the containment test.
    """
    lons_lats = [
        transformer.transform(x, y)
        for x in (cell.min_x, cell.max_x)
        for y in (cell.min_y, cell.max_y)
    ]
    cell_poly = box(
        min(p[0] for p in lons_lats),
        min(p[1] for p in lons_lats),
        max(p[0] for p in lons_lats),
        max(p[1] for p in lons_lats),
    )
    return aoi.contains(cell_poly)


def _filter_cells(
    cells: list[CellGeometry],
    aoi: Polygon,
    keep_rule: KeepRule = "centre_in",
) -> tuple[list[CellGeometry], list[CellGeometry]]:
    """Split cells into kept/dropped by the AOI containment rule.

    Args:
        cells: All candidate cells.
        aoi: AOI polygon in lon/lat (:data:`GEOGRAPHIC_CRS`).
        keep_rule: ``"centre_in"`` keeps a cell iff its centre is in the AOI
            (→ 344 cells); ``"fully_inside"`` requires every corner in the AOI
            (→ 338 cells).

    Returns:
        ``(kept, dropped)`` lists; their concatenation has the same length as
        ``cells``.
    """
    transformer = _make_transformer()
    predicate = _centre_in_aoi if keep_rule == "centre_in" else _fully_inside_aoi

    kept: list[CellGeometry] = []
    dropped: list[CellGeometry] = []
    for cell in cells:
        (kept if predicate(cell, aoi, transformer) else dropped).append(cell)

    logger.info(
        "filtered_cells",
        keep_rule=keep_rule,
        kept=len(kept),
        dropped=len(dropped),
        total=len(cells),
    )
    return kept, dropped


def build_manifest(
    kept: list[CellGeometry],
    dropped: list[CellGeometry],
) -> pd.DataFrame:
    """Build a kept/dropped audit manifest (one row per input cell).

    Args:
        kept: Cells inside the AOI.
        dropped: Cells outside the AOI.

    Returns:
        A DataFrame with columns ``cell_id, center_x, center_y, action`` where
        ``action`` is ``KEEP`` or ``DROP``, sorted by ``cell_id``.
    """
    rows = [
        {
            "cell_id": cell.cell_id,
            "center_x": cell.center_x,
            "center_y": cell.center_y,
            "action": action,
        }
        for action, group in (("KEEP", kept), ("DROP", dropped))
        for cell in group
    ]
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def _cell_to_gridcell(cell: CellGeometry) -> GridCell:
    """Build the productionized :class:`GridCell` from a UTM :class:`CellGeometry`.

    The target grid is ``EPSG:32611`` (UTM 11N) at ``scale=10`` m, ``100×100`` —
    matching the GEE inference reference patches (see ``docs/agents/KNOWLEDGE.md``).
    """
    return GridCell.from_utm_bounds(
        cell_id=cell.cell_id,
        min_x=cell.min_x,
        min_y=cell.min_y,
        max_x=cell.max_x,
        max_y=cell.max_y,
        crs=GRID_MATH_CRS,
    )


def _tile_aoi_to_cells(aoi: Polygon, inset_m: float = 0.0) -> list[CellGeometry]:
    """Tile the AOI into a regular 1 km grid in :data:`GRID_MATH_CRS` (mode B).

    The AOI (lon/lat) is reprojected to UTM 11N, optionally **eroded inward** by
    ``inset_m`` metres (a negative polygon buffer), snapped to a
    :data:`CELL_SIZE_M` lattice over its (inset) bounding box, and every tile whose
    geometry intersects the (inset) AOI is kept (so mode B is bounded by the AOI,
    never the wider cell-sampling bbox). The legacy CSV is not consumed.

    The inset follows the AOI's true shape, eroding ``inset_m`` from every edge: a
    100 km square becomes an 80 km square about the same centroid for
    ``inset_m=10_000``, while an irregular AOI shrinks uniformly inward (concave
    notches and necks narrower than ``2·inset_m`` may disappear). The buffer is
    applied in UTM metres — buffering in lon/lat degrees would be geometrically
    wrong (CRS is law).

    Args:
        aoi: AOI polygon in :data:`GEOGRAPHIC_CRS` (lon/lat).
        inset_m: Inward erosion in metres (``shapely`` ``buffer(-inset_m)`` in UTM).
            ``0.0`` (default) reproduces the un-inset tiling.

    Returns:
        One :class:`CellGeometry` per kept tile, ``cell_id`` in row-major order
        (south-to-north, west-to-east), in UTM metres.

    Raises:
        ValueError: If ``inset_m`` is negative, or if the inset erases the AOI
            entirely (no area left to tile).
    """
    if inset_m < 0:
        raise ValueError(f"inset_m must be >= 0, got {inset_m}.")

    to_utm = Transformer.from_crs(GEOGRAPHIC_CRS, GRID_MATH_CRS, always_xy=True)
    aoi_utm = shapely_transform(lambda xs, ys: to_utm.transform(xs, ys), aoi)
    if inset_m > 0:
        aoi_utm = aoi_utm.buffer(-inset_m)
        if aoi_utm.is_empty or aoi_utm.area == 0.0:
            raise ValueError(f"inset_m={inset_m} erodes the entire AOI; nothing left to tile.")

    min_x, min_y, max_x, max_y = aoi_utm.bounds

    # Snap the origin down to a whole-cell multiple so tiles align deterministically.
    start_x = (min_x // CELL_SIZE_M) * CELL_SIZE_M
    start_y = (min_y // CELL_SIZE_M) * CELL_SIZE_M

    cells: list[CellGeometry] = []
    cell_id = 0
    y = start_y
    while y < max_y:
        x = start_x
        while x < max_x:
            tile = box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M)
            if tile.intersects(aoi_utm):
                cells.append(
                    CellGeometry(
                        cell_id=cell_id,
                        center_x=x + CELL_SIZE_M / 2,
                        center_y=y + CELL_SIZE_M / 2,
                        min_x=x,
                        min_y=y,
                        max_x=x + CELL_SIZE_M,
                        max_y=y + CELL_SIZE_M,
                    )
                )
                cell_id += 1
            x += CELL_SIZE_M
        y += CELL_SIZE_M

    logger.info("tiled_aoi", mode="B", cells=len(cells), inset_m=inset_m)
    return cells


def cells_from_csv(
    cube_cells_csv: Path,
    aoi_path: Path | None = None,
    require_fully_inside: bool = False,
) -> list[CellGeometry]:
    """Mode A — take the cells from a CSV, optionally constrained by an AOI.

    The CSV is the authoritative cell list; the AOI is a *filter* over it, not a
    bound on it. Passing no AOI keeps every cell in the CSV, which is the
    training/validation-sampling case.

    Args:
        cube_cells_csv: Cells CSV (geometry only; see :func:`load_cells`).
        aoi_path: Optional AOI GeoJSON to filter the cells against. ``None``
            keeps every cell.
        require_fully_inside: Keep only fully-contained cells instead of the
            centre-in rule. Requires ``aoi_path``.

    Returns:
        The kept cell geometries, in CSV row order.

    Raises:
        ValueError: If ``require_fully_inside`` is set without an ``aoi_path``,
            or if filtering keeps no cells at all.
    """
    if require_fully_inside and aoi_path is None:
        raise ValueError("require_fully_inside needs an aoi_path to test containment against.")

    cells = load_cells(cube_cells_csv)
    if aoi_path is None:
        return cells

    keep_rule: KeepRule = "fully_inside" if require_fully_inside else "centre_in"
    kept, _ = _filter_cells(cells, load_aoi_polygon(aoi_path), keep_rule=keep_rule)
    if not kept:
        # Almost always a CSV/AOI region mismatch. Without this the sweep runs to
        # completion over an empty grid and writes nothing, silently.
        raise ValueError(
            f"AOI {aoi_path} keeps 0 of {len(cells)} cells from {cube_cells_csv} "
            f"(keep_rule={keep_rule!r}). Do the CSV and the AOI cover the same region?"
        )
    return kept


def cells_from_aoi(aoi_path: Path, inset_m: float = 0.0) -> list[CellGeometry]:
    """Mode B — tile an AOI into a seamless 1 km lattice. No CSV involved.

    Args:
        aoi_path: AOI GeoJSON to tile (authoritative bound).
        inset_m: Erode the AOI inward by this many metres before tiling,
            dropping a border ring of that width. ``0.0`` tiles the full AOI.

    Returns:
        One :class:`CellGeometry` per kept tile, in row-major order.

    Raises:
        ValueError: If ``inset_m`` is negative or erodes the entire AOI.
    """
    return _tile_aoi_to_cells(load_aoi_polygon(aoi_path), inset_m=inset_m)


def build_cells(
    mode: SweepMode = "A",
    cube_cells_csv: Path | None = None,
    aoi_path: Path | None = None,
    require_fully_inside: bool = False,
    mode_b_inset_m: float = 0.0,
) -> list[CellGeometry]:
    """Dispatch to the mode A/B cell builder named by a config value.

    Thin wrapper over :func:`cells_from_csv` / :func:`cells_from_aoi` for callers
    whose mode comes from ``cube.yaml`` rather than the call site. Call those two
    directly when the mode is known statically — their signatures already say
    which inputs are required.

    Args:
        mode: ``"A"`` (cells from CSV) or ``"B"`` (tile the AOI).
        cube_cells_csv: Cells CSV. **Required in mode A**, unused in mode B.
        aoi_path: AOI GeoJSON. **Required in mode B**, optional filter in mode A.
        require_fully_inside: Mode A only — see :func:`cells_from_csv`.
        mode_b_inset_m: Mode B only — see :func:`cells_from_aoi`.

    Returns:
        The list of cell geometries.

    Raises:
        ValueError: If ``mode`` is unknown, if the input that mode requires is
            missing, or if the underlying builder rejects its arguments.
    """
    if mode == "A":
        if cube_cells_csv is None:
            raise ValueError("mode 'A' builds cells from a CSV; cube_cells_csv is required.")
        return cells_from_csv(
            cube_cells_csv,
            aoi_path=aoi_path,
            require_fully_inside=require_fully_inside,
        )
    if mode == "B":
        if aoi_path is None:
            raise ValueError("mode 'B' tiles an AOI; aoi_path is required.")
        return cells_from_aoi(aoi_path, inset_m=mode_b_inset_m)
    raise ValueError(f"Unknown sweep mode {mode!r}; expected 'A' or 'B'.")


def build_grid(
    mode: SweepMode = "A",
    cube_cells_csv: Path | None = None,
    aoi_path: Path | None = None,
    require_fully_inside: bool = False,
    mode_b_inset_m: float = 0.0,
) -> list[GridCell]:
    """Build the inference sweep grid as productionized :class:`GridCell` objects.

    :func:`build_cells` with the geometries promoted to :class:`GridCell` — see it
    for the per-mode argument contract.

    Args:
        mode: ``"A"`` (cells from CSV) or ``"B"`` (tile the AOI).
        cube_cells_csv: Cells CSV. **Required in mode A**, unused in mode B.
        aoi_path: AOI GeoJSON. **Required in mode B**, optional filter in mode A.
        require_fully_inside: Mode A only — see :func:`cells_from_csv`.
        mode_b_inset_m: Mode B only — see :func:`cells_from_aoi`.

    Returns:
        The grid cells, each carrying the UTM 11N / 10 m / 100×100 target triple.

    Raises:
        ValueError: If ``mode`` is unknown, if the input that mode requires is
            missing, or if the underlying builder rejects its arguments.
    """
    cells = build_cells(
        mode=mode,
        aoi_path=aoi_path,
        cube_cells_csv=cube_cells_csv,
        require_fully_inside=require_fully_inside,
        mode_b_inset_m=mode_b_inset_m,
    )

    grid = [_cell_to_gridcell(cell) for cell in cells]
    logger.info("built_grid", mode=mode, cells=len(grid))
    return grid


def generate_date_list(window_start: date, window_end: date) -> list[date]:
    """Return every day in ``[window_start, window_end]`` inclusive."""
    if window_end < window_start:
        raise ValueError(f"window_end {window_end} precedes window_start {window_start}.")
    span = (window_end - window_start).days
    return [window_start + timedelta(days=offset) for offset in range(span + 1)]


def build_cube_dataframe(
    kept: list[CellGeometry],
    window_start: date | None = None,
    window_end: date | None = None,
    days: list[date] | None = None,
) -> pd.DataFrame:
    """Build the generated cube dataframe: full cross-product of cells × window days.

    Each row's ``date`` is a window-end day (``YYYYMMDD``); the GEE/export side
    derives ``window_start = date - (NUM_TIMESTEPS - 1)``. Cell geometry is
    passed through unchanged in :data:`GRID_MATH_CRS`.

    Args:
        kept: In-AOI cells (geometry only).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).
        days: List of individual dates - overrides 'window_start' and 'window_end'.

    Returns:
        A DataFrame with exactly :data:`CUBE_CSV_COLUMNS`, one row per
        ``(cell, day)`` pair, ordered by ``(date, cell_id)``.
    """
    date_list = []
    if window_start and window_end:
        date_list = generate_date_list(window_start, window_end)
    if days:
        date_list = days
        logger.warning("Argument 'day' provided -- Overriding 'window_start' and 'window_end'.")

    rows = [
        {
            "date": int(day.strftime("%Y%m%d")),
            "crs": GRID_MATH_CRS,
            "center_x": cell.center_x,
            "center_y": cell.center_y,
            "min_x": cell.min_x,
            "min_y": cell.min_y,
            "max_x": cell.max_x,
            "max_y": cell.max_y,
        }
        for day in date_list
        for cell in kept
    ]
    frame = pd.DataFrame(rows, columns=CUBE_CSV_COLUMNS)
    logger.info(
        "built_cube_csv",
        cells=len(kept),
        list_of_dates=len(date_list),
        rows=len(frame),
    )
    return frame


def build_cube_dataframe_for_gee_utm(
    kept: list[CellGeometry],
    window_start: date | None = None,
    window_end: date | None = None,
    days: list[date] | None = None,
) -> pd.DataFrame:
    """Build the cube CSV in the dialect ``export_from_csv_utm`` consumes.

    Wraps :func:`build_cube_csv` (single cross-product source of truth) and swaps the
    canonical UTM centre columns for the GEE UTM reader's: ``center_lat`` / ``center_lon``
    with the cell centre reprojected from :data:`GRID_MATH_CRS` to true
    :data:`GEOGRAPHIC_CRS` decimal degrees. The export **geometry** is unchanged — the
    reader derives the region from the (identical) ``min/max_x/y`` + per-row ``crs``; the
    centre columns feed only the output tif filename (``PR_{date}_{lat}_{lon}.tif``), which
    the downstream loader parses as lat/lon — so degrees, never eastings, is correct here.

    Args:
        kept: In-AOI cells (geometry only), in :data:`GRID_MATH_CRS`.
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).
        days: List of individual dates - overrides 'window_start' and 'window_end'.

    Returns:
        A DataFrame with exactly :data:`GEE_UTM_CSV_COLUMNS`, one row per ``(cell, day)``.
    """
    frame = build_cube_dataframe(kept, window_start=window_start, window_end=window_end, days=days)

    to_geo = Transformer.from_crs(GRID_MATH_CRS, GEOGRAPHIC_CRS, always_xy=True)
    lon, lat = to_geo.transform(frame["center_x"].to_numpy(), frame["center_y"].to_numpy())
    frame = frame.assign(center_lat=lat, center_lon=lon)

    logger.info("built_cube_csv_gee_utm", rows=len(frame))
    return frame[GEE_UTM_CSV_COLUMNS]


def generate(
    cube_cells_csv: Path,
    aoi_path: Path,
    output_csv: Path,
    manifest_path: Path,
    window_start: date,
    window_end: date,
    keep_rule: KeepRule = "centre_in",
) -> pd.DataFrame:
    """Run the full geometry pipeline and write the cube CSV + manifest.

    Args:
        cube_cells_csv: Cells CSV (geometry only).
        aoi_path: AOI GeoJSON (authoritative clip/inference boundary).
        output_csv: Destination for the generated cube CSV.
        manifest_path: Destination for the kept/dropped cell manifest.
        keep_rule: AOI containment rule (see :func:`_filter_cells`).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).

    Returns:
        The generated cube CSV DataFrame (also written to ``output_csv``).
    """
    aoi = load_aoi_polygon(aoi_path)
    cells = load_cells(cube_cells_csv)
    kept, dropped = _filter_cells(cells, aoi, keep_rule=keep_rule)

    manifest = build_manifest(kept, dropped)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    cube_csv = build_cube_dataframe(kept, window_start=window_start, window_end=window_end)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cube_csv.to_csv(output_csv, index=False)

    logger.info(
        "generate_complete",
        kept=len(kept),
        dropped=len(dropped),
        cube_rows=len(cube_csv),
        output_csv=str(output_csv),
        manifest=str(manifest_path),
    )
    return cube_csv
