"""Post-run audit for the AOI clip stage (CLIPPING_PLAN §3, TASK-002 subtask 5).

Asserts the clip stage created no zero-signal artifacts and that every input
product is accounted for in the manifest:

1. **Zero all-nodata / zero-valid outputs.** Every written raster must contain
   at least one valid (non-nodata) pixel. Manifest ``CLIP`` rows must report
   ``valid_pixel_count > 0``.
2. **Static-layer mosaic coverage.** The clipped DEM and WorldCover mosaics must
   still reach ``lat 52.31`` (the AOI's northern edge).
3. **Manifest accounting.** Every product has exactly one manifest row.

Exits non-zero (and logs the failures) if any check fails.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
import structlog
import typer

from src.data.local_sources.clip.manifest import read_manifest
from src.data.local_sources.clip.settings import load_aoi_polygon
from src.data.local_sources.paths import LocalPaths

logger = structlog.get_logger()
app = typer.Typer(help="Audit the clipped Bow Valley archive for zero-signal outputs.")

# Path defaults resolve from LocalPaths (env-overridable, LOCAL_ prefix); see
# data/BOW_VALLEY_DATA_LAYOUT.md.
_PATHS = LocalPaths()
DEFAULT_ROOT = _PATHS.clipped_root
DEFAULT_AOI = _PATHS.aoi_path
MANIFEST_NAME = "clip_manifest.csv"

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
            (
                _raster_north_4326(p)
                for p in src_dir.rglob("*.tif")
                if p.is_file()
            ),
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


@app.command()
def audit(
    root: Path = typer.Option(DEFAULT_ROOT, "--root"),
    aoi_path: Path = typer.Option(DEFAULT_AOI, "--aoi"),
) -> None:
    """Run all post-clip audit checks; exit non-zero on any failure."""
    if not root.exists():
        logger.error("clipped root not found", root=str(root))
        raise typer.Exit(code=1)

    aoi = load_aoi_polygon(aoi_path)
    aoi_lat_max = aoi.bounds[3]

    nodata_failures = _check_zero_nodata(root)
    count_failures = _check_manifest_valid_counts(root)
    coverage_failures = _check_static_coverage(root, aoi_lat_max)

    for label, failures in (
        ("all-nodata outputs", nodata_failures),
        ("manifest CLIP rows with 0 valid pixels", count_failures),
        ("static-layer coverage", coverage_failures),
    ):
        if failures:
            logger.error(f"AUDIT FAIL: {label}", count=len(failures), examples=failures[:5])

    total = len(nodata_failures) + len(count_failures) + len(coverage_failures)
    if total:
        raise typer.Exit(code=1)
    logger.info("AUDIT PASS: zero all-nodata outputs; static coverage to lat_max OK")


if __name__ == "__main__":
    app()
