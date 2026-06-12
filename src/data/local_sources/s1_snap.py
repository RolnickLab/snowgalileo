"""Sentinel-1 SNAP preprocessing cache builder (TASK-014, offline step).

The clip stage does **not** preprocess Sentinel-1: ``clip_sentinel1`` only
GCP-slices the raw range-geometry measurement TIFFs into a smaller SAFE ``.zip``
(raw DN, GCPs preserved — no calibration, terrain correction, or dB). Producing
the GEE ``COPERNICUS/S1_GRD`` value domain therefore needs the full ESA SNAP
Sentinel-1 Toolbox chain (the same engine GEE uses), which is heavy
(orbit + SRTM download + terrain correction).

To keep the per-cell :class:`~src.data.local_sources.s1.S1Adapter` ``fetch`` a
**pure raster** operation (no SNAP dependency, unit-testable, fast), SNAP runs
here **once per granule**, producing a 3-band dB+angle GeoTIFF cached on disk:

    clipped SAFE .zip ──(gpt, this module)──▶ s1_grd_<granule_stem>.tif
       (raw DN, GCPs)     production graph      (Sigma0_VV, Sigma0_VH in dB,
                                                 incidenceAngleFromEllipsoid in
                                                 degrees; EPSG:32611, 10 m, AOI)

The adapter then reads, coalesces, mosaics, and reprojects that cache like every
other scene source. The SNAP ``Subset`` to the AOI is the "windowed read" the
spec demands — terrain-correcting the full ~250 km IW swath at 10 m overflows
SNAP's classic-GeoTIFF 4 GB writer and wastes compute. (The clipped SAFEs are
already AOI-bbox slices, so the subset cost is bounded.)

This is an **idempotent, offline** step: run it once before exporting cubes;
re-running skips granules whose cache tif already exists.

    uv run python -m src.data.local_sources.s1_snap

Sentinel-1C note: the archive is all ``S1C_*`` (the satellite launched Dec 2024).
SNAP reads S1C natively; ``xarray-sentinel``/``sarsen`` do **not** (the ``s1[ab]``
regex bug), which is why this chain uses SNAP. See the
``xarray-sentinel-s1c-regex-bug`` memory note.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import structlog
from pyproj import Transformer
from shapely.geometry import Polygon, box

from src.data.local_sources.base import GridCell
from src.data.local_sources.clip.footprints import sentinel_safe_footprint

logger = structlog.get_logger(__name__)

#: Default ESA SNAP ``gpt`` location (NOT ``/usr/bin/snap``, which is snapd).
_DEFAULT_GPT = Path("/home/dev/esa-snap/bin/gpt")

#: The production SNAP graph (beside this module).
_DEFAULT_GRAPH = Path(__file__).with_name("s1_grd_graph.xml")

#: Cache-tif filename prefix.
_CACHE_PREFIX = "s1_grd_"

#: Margin (degrees) added around a cell's bbox so terrain correction has context at
#: the cell edges (no edge artefacts after the reproject crop).
_CELL_MARGIN_DEG: float = 0.02

#: Acquisition ``YYYYMMDD`` in a raw S1 granule stem (``S1C_IW_GRDH_1SDV_<acq>T...``).
_GRANULE_ACQ_PATTERN = re.compile(r"_(\d{8})T\d{6}_")


def _granule_acq_date(granule_zip: Path) -> datetime.date | None:
    """Parse the acquisition date from a raw S1 granule ``.zip`` stem.

    Returns ``None`` if the stem carries no parseable ``_<YYYYMMDD>T<HHMMSS>_`` token.
    """
    match = _GRANULE_ACQ_PATTERN.search(granule_zip.stem)
    if match is None:
        return None
    return datetime.datetime.strptime(match.group(1), "%Y%m%d").date()


def cache_tif_name(granule_stem: str, cell_id: int) -> str:
    """Return the cache-tif filename for a (granule, cell) pair.

    The cache is keyed by **(granule, cell)** because SNAP terrain correction must be
    bounded to where data exists: running it over the full AOI bbox or the full
    clipped scene (still the whole swath's geographic extent — the clip is a
    range-geometry pixel window) hits empty regions and NPE-corrupts the output. A
    small per-cell ``geoRegion`` subset runs clean.

    Args:
        granule_stem: The granule ``.zip`` stem (``S1C_IW_GRDH_...``).
        cell_id: The :class:`GridCell` id the subset is bounded to.

    Returns:
        ``s1_grd_<granule_stem>_cell{cell_id}.tif``.
    """
    return f"{_CACHE_PREFIX}{granule_stem}_cell{cell_id}.tif"


def _cell_region_wkt(cell: GridCell, *, margin_deg: float = _CELL_MARGIN_DEG) -> str:
    """Return the SNAP ``geoRegion`` WKT for a cell: its bbox (4326) + a small margin.

    The cell polygon is in its UTM CRS; this reprojects the bbox to EPSG:4326 (SNAP's
    geoRegion CRS) and pads it. Bounding the subset to one cell keeps SNAP's terrain
    correction over a small, fully-covered area (no empty-region NPEs).

    Args:
        cell: The target grid cell.
        margin_deg: Degrees of padding added around the cell bbox.

    Returns:
        A ``POLYGON((...))`` WKT string in lon/lat.
    """
    transformer = Transformer.from_crs(cell.crs, "EPSG:4326", always_xy=True)
    min_x, min_y, max_x, max_y = cell.polygon.bounds
    lon0, lat0 = transformer.transform(min_x, min_y)
    lon1, lat1 = transformer.transform(max_x, max_y)
    m = margin_deg
    lo0, la0 = min(lon0, lon1) - m, min(lat0, lat1) - m
    lo1, la1 = max(lon0, lon1) + m, max(lat0, lat1) + m
    return f"POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))"


def _cell_intersects_footprint(cell: GridCell, footprint_4326: Polygon | None) -> bool:
    """True if the cell bbox (in 4326) overlaps the granule footprint (or fp unknown)."""
    if footprint_4326 is None or not footprint_4326.is_valid:
        return True  # can't tell — attempt the subset (SNAP will skip if truly empty)
    transformer = Transformer.from_crs(cell.crs, "EPSG:4326", always_xy=True)
    min_x, min_y, max_x, max_y = cell.polygon.bounds
    lon0, lat0 = transformer.transform(min_x, min_y)
    lon1, lat1 = transformer.transform(max_x, max_y)
    cell_4326 = box(min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))
    return cell_4326.intersects(footprint_4326)


def _run_snap_chain(
    *,
    safe_manifest: Path,
    region_wkt: str,
    out_tif: Path,
    gpt: Path,
    graph: Path,
) -> Path:
    """Run the S1_GRD graph via ``gpt`` for one (granule, region); return the tif.

    Args:
        safe_manifest: The extracted ``.SAFE/manifest.safe`` to read.
        region_wkt: lon/lat WKT polygon to subset to (the ``${region}`` param).
        out_tif: Output GeoTIFF path (the ``${output}`` param).
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.

    Returns:
        ``out_tif``.

    Raises:
        FileNotFoundError: If ``gpt`` is not available.
        subprocess.CalledProcessError: If the SNAP chain fails.
    """
    if not gpt.exists():
        raise FileNotFoundError(
            f"ESA SNAP gpt not found at {gpt}; install SNAP or pass gpt=."
        )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gpt),
        str(graph),
        f"-Pinput={safe_manifest}",
        f"-Pregion={region_wkt}",
        f"-Poutput={out_tif}",
        "-c",
        "2G",
        "-q",
        "4",
    ]
    logger.info("s1_snap_chain_start", granule=safe_manifest.parent.name, out=out_tif.name)
    subprocess.run(cmd, check=True)
    return out_tif


def build_granule_cache(
    *,
    granule_zip: Path,
    cells: list[GridCell],
    cache_dir: Path,
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _DEFAULT_GRAPH,
    overwrite: bool = False,
) -> list[Path]:
    """Produce per-cell dB+angle cache tifs for one clipped S1 granule ``.zip``.

    Extracts the SAFE **once**, then for each cell whose bbox overlaps the granule
    footprint runs the SNAP graph subset to that cell's bbox+margin, writing
    ``{cache_dir}/s1_grd_<stem>_cell{id}.tif``. Idempotent per (granule, cell).

    Args:
        granule_zip: The clipped IW GRD SAFE ``.zip``.
        cells: Target grid cells (each gets its own bounded subset tif).
        cache_dir: Directory the cache tifs are written to.
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.
        overwrite: Re-run SNAP even where a cache tif already exists.

    Returns:
        The cache tif paths produced for this granule (cells outside the footprint
        are skipped and not included).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    footprint = sentinel_safe_footprint(granule_zip, "manifest.safe")
    covered = [c for c in cells if _cell_intersects_footprint(c, footprint)]
    if not covered:
        logger.info("s1_snap_no_cells", granule=granule_zip.stem)
        return []

    # All covered cells need the same extracted SAFE — extract once, reuse.
    pending = [
        c for c in covered
        if overwrite or not (cache_dir / cache_tif_name(granule_zip.stem, c.cell_id)).exists()
    ]
    out_paths = [cache_dir / cache_tif_name(granule_zip.stem, c.cell_id) for c in covered]
    if not pending:
        logger.info("s1_snap_cache_hit", granule=granule_zip.stem, cells=len(covered))
        return out_paths

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(granule_zip) as zf:
            zf.extractall(tmp_dir)
        safe_dirs = list(tmp_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise FileNotFoundError(f"No .SAFE directory inside {granule_zip}.")
        manifest = safe_dirs[0] / "manifest.safe"
        for cell in pending:
            _run_snap_chain(
                safe_manifest=manifest,
                region_wkt=_cell_region_wkt(cell),
                out_tif=cache_dir / cache_tif_name(granule_zip.stem, cell.cell_id),
                gpt=gpt,
                graph=graph,
            )
    return out_paths


def build_s1_cache(
    *,
    archive_root: Path,
    cells: list[GridCell],
    cache_dir: Path,
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _DEFAULT_GRAPH,
    overwrite: bool = False,
) -> list[Path]:
    """Build the per-(granule, cell) SNAP cache for every clipped S1 granule.

    The once/offline step analogous to the clip stage. Idempotent — already-cached
    (granule, cell) tifs are skipped (unless ``overwrite``).

    Args:
        archive_root: The clipped S1 archive (holds ``S1*_IW_GRDH_*.zip``).
        cells: Target grid cells each granule is subset to (per cell).
        cache_dir: Output directory for the ``s1_grd_*_cell*.tif`` cache.
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.
        overwrite: Re-run SNAP even where a cache tif already exists.

    Returns:
        All cache tif paths produced, in granule-name order.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    granules = sorted(archive_root.glob("S1*_IW_GRDH_*.zip"))
    logger.info(
        "s1_snap_build_start",
        n_granules=len(granules), n_cells=len(cells), cache_dir=str(cache_dir),
    )
    cached: list[Path] = []
    for granule_zip in granules:
        cached.extend(
            build_granule_cache(
                granule_zip=granule_zip,
                cells=cells,
                cache_dir=cache_dir,
                gpt=gpt,
                graph=graph,
                overwrite=overwrite,
            )
        )
    logger.info("s1_snap_build_done", n_cached=len(cached))
    return cached


class S1CacheUnavailableError(RuntimeError):
    """A needed S1 cache tif is missing and cannot be built (no gpt / no raw SAFEs).

    Raised by :func:`ensure_s1_cache` so cube export fails loudly instead of silently
    assembling an all-``-9999`` S1 block (the historical silent-dropout bug).
    """


def ensure_s1_cache(
    *,
    archive_root: Path,
    cells: list[GridCell],
    cache_dir: Path,
    window_days: list[datetime.date],
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _DEFAULT_GRAPH,
) -> None:
    """Guarantee the S1 cache covers every ``(cell, window-day)`` granule before export.

    Computes the **needed** ``(granule, cell)`` cache tifs — raw granules whose
    acquisition day is in ``window_days`` and whose footprint intersects the cell — and:

    * returns immediately if all needed tifs already exist (no SNAP run);
    * builds only the missing ones via :func:`build_granule_cache` when ``gpt`` and the
      raw SAFEs are present;
    * **raises** :class:`S1CacheUnavailableError` if any are missing and the cache cannot
      be built (``gpt`` absent or no raw granules) — so the cube is never silently filled
      with ``-9999`` S1.

    The check is footprint-aware: a window-day granule that does not cover a cell is **not**
    "missing" (it genuinely has no data there), so a legitimately S1-free cell does not
    trip the guard. By the same rule, a cell with **no covering granule in the window at
    all** (no S1 acquired over it this window) needs nothing built and is left with its
    ``-9999`` S1 block — genuinely-absent S1 is acceptable; only a *buildable-but-unbuilt*
    cache trips the guard.

    Args:
        archive_root: The clipped S1 archive (holds ``S1*_IW_GRDH_*.zip``).
        cells: The cells about to be exported (each needs its bounded subset).
        cache_dir: The SNAP cache directory the :class:`S1Adapter` reads.
        window_days: The export window's days (the adapter reads one timestep per day).
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.

    Raises:
        S1CacheUnavailableError: If a needed cache tif is missing and unbuildable.
    """
    window = set(window_days)
    granules = sorted(archive_root.glob("S1*_IW_GRDH_*.zip"))
    in_window = [g for g in granules if _granule_acq_date(g) in window]

    # Which (granule, cell) tifs are needed but absent? Footprint-gate per granule so an
    # uncovered cell is not counted as missing.
    missing_by_granule: dict[Path, list[GridCell]] = {}
    for granule_zip in in_window:
        footprint = sentinel_safe_footprint(granule_zip, "manifest.safe")
        for cell in cells:
            if not _cell_intersects_footprint(cell, footprint):
                continue
            tif = cache_dir / cache_tif_name(granule_zip.stem, cell.cell_id)
            if not tif.exists():
                missing_by_granule.setdefault(granule_zip, []).append(cell)

    if not missing_by_granule:
        return

    n_missing = sum(len(v) for v in missing_by_granule.values())
    if not gpt.exists():
        raise S1CacheUnavailableError(
            f"{n_missing} S1 cache tif(s) across {len(missing_by_granule)} granule(s) are "
            f"missing and ESA SNAP gpt was not found at {gpt}. Build the cache first: "
            f"`uv run python scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py` (or pass a valid --gpt). "
            f"Missing granules: {[g.stem for g in missing_by_granule]}."
        )

    logger.info(
        "s1_cache_ensure_building",
        n_granules=len(missing_by_granule), n_tifs=n_missing, cache_dir=str(cache_dir),
    )
    for granule_zip, granule_cells in missing_by_granule.items():
        try:
            build_granule_cache(
                granule_zip=granule_zip,
                cells=granule_cells,
                cache_dir=cache_dir,
                gpt=gpt,
                graph=graph,
            )
        except subprocess.CalledProcessError as exc:
            # A per-(granule, cell) SNAP failure — e.g. the known Subset "Empty region!"
            # anomaly where the footprint overlaps but the GCP-clipped scene has no pixels
            # over the cell — means S1 genuinely cannot be produced there. Log loudly and
            # leave that cell S1-free (the user's "absent S1 is OK" rule) rather than
            # aborting the whole cube. A *systemic* failure (no gpt) already raised above.
            logger.warning(
                "s1_cache_snap_failed",
                granule=granule_zip.stem,
                cells=[c.cell_id for c in granule_cells],
                returncode=exc.returncode,
            )


def _main() -> None:
    """CLI: build the per-cell S1 SNAP cache (the offline preprocessing step).

    Cells come from the production grid (:func:`build_grid`). SNAP terrain
    correction must be bounded per cell — a full-AOI/full-scene run NPE-corrupts on
    empty regions (the clip is a range-geometry pixel window, not a tight geographic
    mask).
    """
    import argparse

    from src.data.local_sources.grid import build_grid
    from src.data.local_sources.paths import LocalPaths

    paths = LocalPaths()
    parser = argparse.ArgumentParser(description="Build the Sentinel-1 per-cell SNAP cache.")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=paths.clipped_root / "sentinel1",
        help="Clipped S1 archive (holds S1*_IW_GRDH_*.zip).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=paths.clipped_root / "sentinel1_snap",
        help="Output directory for the s1_grd_*_cell*.tif cache.",
    )
    parser.add_argument("--gpt", type=Path, default=_DEFAULT_GPT)
    parser.add_argument("--graph", type=Path, default=_DEFAULT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cells = build_grid()
    cached = build_s1_cache(
        archive_root=args.archive_root,
        cells=cells,
        cache_dir=args.cache_dir,
        gpt=args.gpt,
        graph=args.graph,
        overwrite=args.overwrite,
    )
    for path in cached:
        print(path)


if __name__ == "__main__":
    _main()
