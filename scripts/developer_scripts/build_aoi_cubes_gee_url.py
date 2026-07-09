r"""Operator entry point — build FSC input cubes over an AOI via the GEE URL pipeline.

Given an AOI GeoJSON and a ``[start_date, end_date]`` range of window-end days, this
script tiles the AOI into a regular 1 km EPSG:32611 lattice (the **Mode B** tiler, the
same one :func:`snow_galileo.data.local_sources.grid.build_grid` uses), emits the cube
CSV in the exact schema the **original GEE exporter** consumes, and runs
:meth:`EarthEngineExporterEval.export_from_csv_utm` in ``url`` mode to download one 8-day
cube per ``(cell, day)`` pair.

**Why Mode B (not the legacy sampled cells).** A continuous AOI needs a gapless,
lattice-aligned tiling so the downloaded cubes tile seamlessly. Mode B snaps every tile to
a whole-``CELL_SIZE_M`` origin, guaranteeing a common lattice and uniform 1 km cells — the
prerequisite for stitching the results later like ``InferenceGridDriver`` /
``DailyMosaicWriter`` do. The legacy sampled-cell CSV (Mode A) is a scatter with holes and
must not be used for continuous coverage.

**CSV contract (verified against the live reader).**
:meth:`EarthEngineExporterEval.export_from_csv_utm` reads
``date, crs, center_lat, center_lon, min_x, max_x, min_y, max_y`` (``eo_eval.py:611-618``).
It uses ``center_lat`` / ``center_lon`` only to name the output file and derives the export
geometry from the UTM ``min/max`` bounds + per-row ``crs`` (reprojecting to 4326 itself).
NOTE: this differs from :func:`grid.build_cube_dataframe`, which emits ``center_x`` / ``center_y``.
This script delegates to :func:`grid.build_cube_csv_for_gee_utm` — the adapter that emits
what the reader consumes, with ``center_lat`` / ``center_lon`` populated with the cell
centre reprojected to true decimal degrees (CRS-correct, not mislabelled eastings).

**Downstream stitching seam (not handled here).** The GEE url output is named
``PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif``, which does **not** match the driver's
``build_cube_filename`` (``PR_{date}_{lat}_{lon}_SC00.tif``). A future stitcher must parse
these filenames (or map by the CSV's UTM bounds / ``cell_id``) itself. Cube download is
unaffected.

Example:
    uv run python scripts/developer_scripts/build_aoi_cubes_gee_url.py \\
        --aoi data/fortress_mountain_basin_aoi.geojson \\
        --start-date 2025-04-06 --end-date 2025-04-13 \\
        --tifs-folder fortress_cubes --dry-run
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Optional

import structlog
import typer
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform

from snow_galileo.data import EarthEngineExporterEval

# Reuse the canonical Mode-B tiler and the GEE-UTM cube-CSV builder — never re-implement
# the lattice-snap or the centre-reprojection logic (a second copy would be a
# seam-alignment / wrong-projection bug waiting to happen).
from snow_galileo.data.local_sources.grid import (
    GEOGRAPHIC_CRS,
    build_cells,
    build_cube_csv_for_gee_utm,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Build AOI FSC cubes via the GEE URL pipeline.")

#: The GeoJSON default CRS (RFC 7946): WGS84 lon/lat.
_DEFAULT_GEOJSON_CRS: str = "EPSG:4326"


def _load_aoi_geographic(aoi_path: Path) -> Polygon:
    """Load the AOI as a lon/lat (:data:`GEOGRAPHIC_CRS`) polygon, honouring its CRS.

    Reads the GeoJSON's declared CRS (``crs.properties.name``; default WGS84 lon/lat per
    RFC 7946) and reprojects the polygon to EPSG:4326 if it is anything else, so the AOI is
    always handed to the Mode-B tiler in the CRS it expects. CRS is law: a UTM AOI fed in
    as if it were degrees would produce garbage tiles.

    Args:
        aoi_path: Path to a GeoJSON with exactly one ``Polygon`` feature.

    Returns:
        The AOI as a shapely :class:`~shapely.geometry.Polygon` in lon/lat.

    Raises:
        ValueError: If the file does not hold exactly one ``Polygon`` feature.
    """
    raw = json.loads(aoi_path.read_text())
    features = raw.get("features", [])
    if len(features) != 1:
        raise ValueError(f"AOI {aoi_path} must hold exactly one feature, found {len(features)}.")
    geometry = features[0]["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError(f"AOI geometry must be a Polygon, found {geometry['type']!r}.")

    poly = Polygon(geometry["coordinates"][0])

    crs_name = raw.get("crs", {}).get("properties", {}).get("name")
    src_crs = (
        CRS.from_user_input(crs_name) if crs_name else CRS.from_user_input(_DEFAULT_GEOJSON_CRS)
    )
    if src_crs != CRS.from_user_input(GEOGRAPHIC_CRS):
        # always_xy=True: keep (lon, lat) order regardless of the source's declared axes.
        to_geo = Transformer.from_crs(src_crs, GEOGRAPHIC_CRS, always_xy=True)
        poly = shapely_transform(lambda xs, ys: to_geo.transform(xs, ys), poly)
        logger.info("reprojected_aoi", src_crs=src_crs.to_string(), dst_crs=GEOGRAPHIC_CRS)

    return poly


def _window_end_days(start: date, end: date) -> list[date]:
    """Return every window-end day in ``[start, end]`` inclusive.

    Each day becomes one cube per cell; the GEE exporter derives the 8-day window's
    start as ``end - (NUM_TIMESTEPS - 1)`` itself.

    Raises:
        ValueError: If ``end`` precedes ``start``.
    """
    if end < start:
        raise ValueError(f"end_date {end} precedes start_date {start}.")
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


@app.command()
def main(
    aoi: Annotated[Path, typer.Option(help="AOI GeoJSON (single Polygon; any declared CRS).")],
    start_date: Annotated[str, typer.Option(help="First window-end day, YYYY-MM-DD (inclusive).")],
    end_date: Annotated[str, typer.Option(help="Last window-end day, YYYY-MM-DD (inclusive).")],
    tifs_folder: Annotated[
        str,
        typer.Option(help="Cube download folder NAME (created under the configured DATA_FOLDER)."),
    ] = "aoi_cubes",
    out_csv: Annotated[Path, typer.Option(help="Where to write the generated cube CSV.")] = Path(
        "configs/aoi_cubes/cube_cells.csv"
    ),
    mode: Annotated[
        str, typer.Option(help="GEE export mode: 'url' (download), 'cloud', or 'drive'.")
    ] = "url",
    inset_m: Annotated[
        float,
        typer.Option(
            help="Erode the AOI inward by this many metres before tiling (0 = full AOI)."
        ),
    ] = 0.0,
    buffer_m: Annotated[
        float,
        typer.Option(
            help="Grow each cell outward by this many metres at export for seamless overlap "
            "(export geometry only; CSV cell bounds stay canonical). Neighbours overlap 2x this."
        ),
    ] = 40.0,
    max_workers: Annotated[
        int,
        typer.Option(
            help="Parallel download threads (url mode only). Keep low — GEE throttles "
            "getDownloadURL, so a big pool yields 429s, not speed. 1 = serial."
        ),
    ] = 4,
    limit: Annotated[
        Optional[int],
        typer.Option(help="Cap the number of CSV rows (smoke run); None = all (cell, day) pairs."),
    ] = None,
    check_gcp: Annotated[
        bool, typer.Option(help="Check Google Cloud Storage for already-exported tifs first.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Only build/write the cube CSV; do not touch Earth Engine."),
    ] = False,
) -> None:
    """Tile the AOI, emit the cube CSV, and (unless ``--dry-run``) download cubes via GEE."""
    if mode not in {"url", "cloud", "drive"}:
        raise typer.BadParameter(f"mode must be url/cloud/drive, got {mode!r}.")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    days = _window_end_days(start, end)

    list_of_cells = build_cells(mode="B", aoi_path=aoi)
    if not list_of_cells:
        raise typer.BadParameter(f"AOI {aoi} tiled to zero cells (check CRS / inset_m).")

    frame = build_cube_csv_for_gee_utm(list_of_cells, window_start=start, window_end=end)

    if limit is not None:
        frame = frame.head(limit)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    logger.info(
        "wrote_cube_csv",
        path=str(out_csv),
        cells=len(list_of_cells),
        days=len(days),
        cubes_to_export=len(frame),
        window=f"{start.isoformat()}..{end.isoformat()}",
    )

    if dry_run:
        typer.echo(
            f"[dry-run] Wrote {len(frame)} rows ({len(list_of_cells)} cells x {len(days)} days) "
            f"to {out_csv}. No cubes downloaded."
        )
        return

    exporter = EarthEngineExporterEval(check_gcp=check_gcp, mode=mode, tifs_folder=tifs_folder)
    logger.info(
        "export_start",
        mode=mode,
        tifs_folder=tifs_folder,
        cubes=len(frame),
        buffer_m=buffer_m,
        max_workers=max_workers,
    )
    # Native-UTM export: pins every tile to the shared UTM lattice (no EPSG:4326 round-trip)
    # and adds a buffer_m overlap halo, so tiles are seamless. See EarthEngineExporterEval.
    exporter.export_from_csv_utm_native(
        csv_file=str(out_csv), buffer_m=buffer_m, max_workers=max_workers
    )
    logger.info("export_complete", mode=mode, tifs_folder=tifs_folder, cubes=len(frame))


if __name__ == "__main__":
    app()
