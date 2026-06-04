"""AOI 1 km grid generator — Phase 0 geometry half.

This module ships the *geometry half* of the grid generator required by Phase 0
(TASK-001): load the legacy cell-sampling CSV for **cell geometry only**, filter
cells to ``data/bow_valley_inference_aoi.geojson``, emit a kept/dropped manifest, and emit the
generated cross-product cube CSV that drives both the inference sweep and the
Phase 0 GEE reference-patch run.

The mode A/B switch, ``cube_cache`` wiring, and the per-cell ``GridCell`` target
transform are productionized later in TASK-003; they are intentionally **not**
implemented here.

Key contracts (verified against the codebase, see
``docs/agents/planning/raw-data-ingestion/``):

- The generated CSV schema is fixed by ``EarthEngineExporterEval.export_from_csv_utm``
  (``src/data/earthengine/eo_eval.py:577-585``): exactly
  ``date, crs, center_x, center_y, min_x, min_y, max_x, max_y``.
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
from typing import Annotated, Literal

import pandas as pd
import structlog
import typer
from pyproj import Transformer
from shapely.geometry import Point, Polygon, box

logger = structlog.get_logger(__name__)

# --- Fixed contracts -------------------------------------------------------

#: Columns consumed verbatim by ``export_from_csv_utm`` (``eo_eval.py:577-585``).
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

#: CRS the legacy cells (and therefore the generated CSV) are expressed in.
GRID_MATH_CRS: str = "EPSG:32611"

#: Geographic CRS used only for the AOI point-in-polygon test.
GEOGRAPHIC_CRS: str = "EPSG:4326"

#: Default inference window (PLAN §3 Temporal window), inclusive of both ends.
DEFAULT_WINDOW_START: date = date(2025, 4, 6)
DEFAULT_WINDOW_END: date = date(2025, 5, 28)

#: Repo-root-relative default paths (resolved against the package's repo root).
DEFAULT_LEGACY_CSV: Path = Path("sampled_cells_bow_river_with_dates.csv")
DEFAULT_AOI_PATH: Path = Path("data/bow_valley_inference_aoi.geojson")
DEFAULT_OUTPUT_CSV: Path = Path("configs/bow_valley/cube_cells.csv")
DEFAULT_MANIFEST_PATH: Path = Path("configs/bow_valley/cell_filter_manifest.csv")

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
        raise ValueError(
            f"AOI geometry must be a Polygon, found {geometry['type']!r}."
        )
    return Polygon(geometry["coordinates"][0])


def load_cells(legacy_csv: Path) -> list[CellGeometry]:
    """Load cell geometry from the legacy sampling CSV (geometry only).

    The legacy ``date`` column is train/eval label-sampling metadata and is
    **not** read here (see PLAN §8 Q4). Rows are deduplicated on their full
    geometry so a cell sampled on multiple label dates is counted once.

    Args:
        legacy_csv: Path to ``sampled_cells_bow_river_with_dates.csv``.

    Returns:
        One :class:`CellGeometry` per unique cell, ``cell_id`` assigned in
        stable row order.

    Raises:
        ValueError: If any cell CRS is not :data:`GRID_MATH_CRS`.
    """
    df = pd.read_csv(legacy_csv)
    geom_cols = ["center_x", "center_y", "min_x", "min_y", "max_x", "max_y"]
    cells_df = df.drop_duplicates(subset=geom_cols).reset_index(drop=True)

    bad_crs = set(cells_df["crs"].unique()) - {GRID_MATH_CRS}
    if bad_crs:
        raise ValueError(
            f"Legacy CSV cells must be {GRID_MATH_CRS}; found unexpected {bad_crs}."
        )

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
    logger.info("loaded_cells", count=len(cells), source=str(legacy_csv))
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


def filter_cells(
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


def _window_days(window_start: date, window_end: date) -> list[date]:
    """Return every day in ``[window_start, window_end]`` inclusive."""
    if window_end < window_start:
        raise ValueError(
            f"window_end {window_end} precedes window_start {window_start}."
        )
    span = (window_end - window_start).days
    return [window_start + timedelta(days=offset) for offset in range(span + 1)]


def build_cube_csv(
    kept: list[CellGeometry],
    window_start: date = DEFAULT_WINDOW_START,
    window_end: date = DEFAULT_WINDOW_END,
) -> pd.DataFrame:
    """Build the generated cube CSV: full cross-product of cells × window days.

    Each row's ``date`` is a window-end day (``YYYYMMDD``); the GEE/export side
    derives ``window_start = date - (NUM_TIMESTEPS - 1)``. Cell geometry is
    passed through unchanged in :data:`GRID_MATH_CRS`.

    Args:
        kept: In-AOI cells (geometry only).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).

    Returns:
        A DataFrame with exactly :data:`CUBE_CSV_COLUMNS`, one row per
        ``(cell, day)`` pair, ordered by ``(date, cell_id)``.
    """
    days = _window_days(window_start, window_end)
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
        for day in days
        for cell in kept
    ]
    frame = pd.DataFrame(rows, columns=CUBE_CSV_COLUMNS)
    logger.info(
        "built_cube_csv",
        cells=len(kept),
        window_days=len(days),
        rows=len(frame),
    )
    return frame


def generate(
    legacy_csv: Path = DEFAULT_LEGACY_CSV,
    aoi_path: Path = DEFAULT_AOI_PATH,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    keep_rule: KeepRule = "centre_in",
    window_start: date = DEFAULT_WINDOW_START,
    window_end: date = DEFAULT_WINDOW_END,
) -> pd.DataFrame:
    """Run the full geometry pipeline and write the cube CSV + manifest.

    Args:
        legacy_csv: Legacy cell-sampling CSV (geometry only).
        aoi_path: AOI GeoJSON (authoritative clip/inference boundary).
        output_csv: Destination for the generated cube CSV.
        manifest_path: Destination for the kept/dropped cell manifest.
        keep_rule: AOI containment rule (see :func:`filter_cells`).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).

    Returns:
        The generated cube CSV DataFrame (also written to ``output_csv``).
    """
    aoi = load_aoi_polygon(aoi_path)
    cells = load_cells(legacy_csv)
    kept, dropped = filter_cells(cells, aoi, keep_rule=keep_rule)

    manifest = build_manifest(kept, dropped)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    cube_csv = build_cube_csv(kept, window_start=window_start, window_end=window_end)
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


def emit_csv(
    legacy_csv: Annotated[Path, typer.Option(help="Legacy cell-sampling CSV.")] = DEFAULT_LEGACY_CSV,
    aoi_path: Annotated[Path, typer.Option("--aoi", help="AOI GeoJSON.")] = DEFAULT_AOI_PATH,
    output_csv: Annotated[Path, typer.Option(help="Generated cube CSV output.")] = DEFAULT_OUTPUT_CSV,
    manifest_path: Annotated[
        Path, typer.Option(help="Kept/dropped manifest output.")
    ] = DEFAULT_MANIFEST_PATH,
    require_fully_inside: Annotated[
        bool, typer.Option("--require-fully-inside", help="Keep only fully-contained cells (→ 338).")
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
        legacy_csv=legacy_csv,
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
    typer.run(emit_csv)


if __name__ == "__main__":
    main()
