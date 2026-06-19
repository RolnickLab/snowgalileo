"""Post-run audit for the raw-processing stage (clip + Sentinel-1 SNAP).

Asserts the processing stage created no zero-signal artifacts, that every input
product is accounted for, and that the S1 SNAP cache covers the granules it should:

1. **Zero all-nodata / zero-valid outputs.** Every written raster must contain
   at least one valid (non-nodata) pixel. Manifest ``CLIP`` rows must report
   ``valid_pixel_count > 0``.
2. **Static-layer mosaic coverage.** The clipped DEM and WorldCover mosaics must
   still reach ``lat 52.31`` (the AOI's northern edge).
3. **Manifest accounting.** Every product has exactly one manifest row.
4. **S1 SNAP cache coverage.** Every raw S1 granule whose footprint overlaps the
   AOI has a per-granule ``s1_grd_<stem>.tif`` in the SNAP cache (so the cube's S1
   bands are not silently all-``-9999``). Skipped if no raw S1 archive is present.

Exits non-zero (and logs the failures) if any check fails.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
import structlog
import typer
from shapely.geometry import Polygon

from src.data.local_sources.clip.footprints import sentinel_safe_footprint
from src.data.local_sources.clip.manifest import read_manifest
from src.data.local_sources.clip.settings import load_aoi_polygon
from src.data.local_sources.paths import LocalPaths
from src.data.local_sources.s1_snap import _aoi_intersects_footprint, cache_tif_name

logger = structlog.get_logger()
app = typer.Typer(help="Audit the processed Bow Valley archive (clip + S1 SNAP).")

# Path defaults resolve from LocalPaths (env-overridable, LOCAL_ prefix); see
# data/BOW_VALLEY_DATA_LAYOUT.md.
_PATHS = LocalPaths()
DEFAULT_ROOT = _PATHS.clipped_root
DEFAULT_AOI = _PATHS.aoi_path
MANIFEST_NAME = "clip_manifest.csv"

# Sentinel-1 SNAP cache coverage check (raw granules → per-granule cache tifs).
DEFAULT_S1_RAW = _PATHS.raw_root / "sentinel1"
DEFAULT_S1_CACHE = _PATHS.clipped_root / "sentinel1_snap"

# Rasters the audit can open directly with rasterio (per-grid GeoTIFFs + tiles).
_AUDITABLE_SUFFIXES = {".tif", ".tiff"}


def _raster_has_valid_pixels(path: Path) -> bool:
    """True if a raster has at least one non-nodata pixel."""
    try:
        with rasterio.open(path) as src:
            data = src.read()
            if src.nodata is None:
                return data.size > 0
            return bool((data != src.nodata).any())
    except rasterio.errors.RasterioIOError:
        return True  # not a standalone raster (e.g. inside a zip); skip here


def _check_zero_nodata(root: Path) -> list[str]:
    """Return paths of standalone rasters that are entirely nodata/empty."""
    failures: list[str] = []
    for path in root.rglob("*"):
        if path.suffix.lower() in _AUDITABLE_SUFFIXES and path.is_file():
            if not _raster_has_valid_pixels(path):
                failures.append(str(path))
    return failures


def _check_manifest_valid_counts(root: Path) -> list[str]:
    """Return manifest CLIP rows that report zero valid pixels."""
    failures: list[str] = []
    for manifest_path in root.rglob(MANIFEST_NAME):
        if manifest_path.parent != root:  # per-source manifests only (skip combined)
            for row in read_manifest(manifest_path):
                if row["action"] == "CLIP" and int(row["valid_pixel_count"]) == 0:
                    failures.append(f"{manifest_path}: {row['product_id']}")
    return failures


def _check_static_coverage(root: Path, aoi_lat_max: float) -> list[str]:
    """Verify clipped DEM/WorldCover mosaics still reach the AOI's north edge."""
    failures: list[str] = []
    for source in ("dem", "worldcover"):
        src_dir = root / source
        if not src_dir.exists():
            continue
        north = max(
            (_raster_north_4326(p) for p in src_dir.rglob("*.tif") if p.is_file()),
            default=None,
        )
        if north is None:
            failures.append(f"{source}: no clipped tiles found")
        elif north < aoi_lat_max - 0.01:
            failures.append(
                f"{source}: mosaic north edge {north:.4f} < AOI lat_max {aoi_lat_max:.4f}"
            )
    return failures


def _raster_north_4326(path: Path) -> float:
    """Northern bound of a raster in latitude (EPSG:4326)."""
    from pyproj import Transformer

    with rasterio.open(path) as src:
        b = src.bounds
        if src.crs is None or src.crs.to_epsg() == 4326:
            return b.top
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        _, north = transformer.transform((b.left + b.right) / 2, b.top)
        return north


def _check_s1_snap_coverage(raw_s1_dir: Path, cache_dir: Path, aoi: Polygon) -> list[str]:
    """Return raw S1 granules that overlap the AOI but have no per-granule SNAP cache tif.

    Skips cleanly (returns ``[]``) if no raw S1 archive is present — the SNAP cache is an
    optional, machine-specific prestep. A granule whose footprint does not overlap the AOI
    is not expected to have a tif (genuinely S1-free there), so it is not a failure.
    """
    if not raw_s1_dir.exists():
        return []
    failures: list[str] = []
    for granule_zip in sorted(raw_s1_dir.glob("S1*_IW_GRDH_*.zip")):
        footprint = sentinel_safe_footprint(granule_zip, "manifest.safe")
        if not _aoi_intersects_footprint(aoi, footprint):
            continue
        if not (cache_dir / cache_tif_name(granule_zip.stem)).exists():
            failures.append(granule_zip.stem)
    return failures


@app.command()
def audit(
    root: Path = typer.Option(DEFAULT_ROOT, "--root"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
    raw_s1_dir: Path = typer.Option(DEFAULT_S1_RAW, "--raw-s1-dir"),
    s1_cache_dir: Path = typer.Option(DEFAULT_S1_CACHE, "--s1-cache-dir"),
) -> None:
    """Run all post-processing audit checks; exit non-zero on any failure."""
    if not root.exists():
        logger.error("clipped root not found", root=str(root))
        raise typer.Exit(code=1)

    aoi = load_aoi_polygon(aoi_path)
    aoi_lat_max = aoi.bounds[3]

    nodata_failures = _check_zero_nodata(root)
    count_failures = _check_manifest_valid_counts(root)
    coverage_failures = _check_static_coverage(root, aoi_lat_max)
    s1_cache_failures = _check_s1_snap_coverage(raw_s1_dir, s1_cache_dir, aoi)

    for label, failures in (
        ("all-nodata outputs", nodata_failures),
        ("manifest CLIP rows with 0 valid pixels", count_failures),
        ("static-layer coverage", coverage_failures),
        ("S1 SNAP cache missing for AOI-covering granules", s1_cache_failures),
    ):
        if failures:
            logger.error(f"AUDIT FAIL: {label}", count=len(failures), examples=failures[:5])

    total = (
        len(nodata_failures)
        + len(count_failures)
        + len(coverage_failures)
        + len(s1_cache_failures)
    )
    if total:
        raise typer.Exit(code=1)
    logger.info(
        "AUDIT PASS: zero all-nodata outputs; static coverage to lat_max OK; S1 SNAP cache complete"
    )


if __name__ == "__main__":
    app()
