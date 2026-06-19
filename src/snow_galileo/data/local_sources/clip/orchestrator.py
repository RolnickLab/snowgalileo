"""Source-walking orchestration: dispatch products to per-modality clippers.

Maps each raw source directory to its clip routine, iterates its products,
applies the gate + clip, and collects manifest rows. ``--dry-run`` runs only the
metadata gate (no pixels decoded, no files written).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import structlog
from shapely.geometry import Polygon

from . import clippers, footprints
from .gate import ClipAction, GateResult, evaluate_gate
from .manifest import ManifestRow, bbox_str
from .settings import ClipSettings

logger = structlog.get_logger()

#: Source directory names handled by the clip stage, in a stable order.
#: NOTE: Sentinel-1 is intentionally absent — S1 is *processed* from raw via ESA SNAP
#: (``process_raw_dataset.py process-s1`` → ``sentinel1_snap/``), NOT clipped. Everything
#: downstream (cube adapter AND viewer) reads the processed SNAP cache; there is no use
#: for raw-DN clipped S1. See PLAN-S1-PERGRANULE-SNAP.md.
SOURCES = [
    "dem",
    "worldcover",
    "era5",
    "landsat8",
    "landsat9",
    "modis",
    "sentinel2",
    "sentinel3",
    "viirs",
]


@dataclass(frozen=True)
class _Modality:
    """How to discover and clip the products of one source directory.

    Attributes:
        glob: Glob (relative to the source dir) selecting input products.
        per_grid_dir: True for MODIS/VIIRS, whose output is a directory of
            per-grid GeoTIFFs rather than a single file.
        clip: The clip routine to call for a single product.
        gate_footprint: Optional metadata-only footprint reader used by
            ``--dry-run`` (the clip routines re-read it internally on a real run).
    """

    glob: str
    per_grid_dir: bool
    clip: Callable[..., ManifestRow]
    gate_footprint: Optional[Callable[[Path], Optional[Polygon]]] = None


def _s2_footprint(p: Path) -> Optional[Polygon]:
    return footprints.sentinel_safe_footprint(p, "manifest.safe")


def _s3_footprint(p: Path) -> Optional[Polygon]:
    return footprints.sentinel_safe_footprint(p, "xfdumanifest.xml")


MODALITIES: dict[str, _Modality] = {
    "dem": _Modality(
        "**/*_DEM.tif", False, clippers.clip_geotiff, footprints.raster_footprint_4326
    ),
    "worldcover": _Modality(
        "**/*_Map.tif", False, clippers.clip_geotiff, footprints.raster_footprint_4326
    ),
    "era5": _Modality("**/*.nc", False, clippers.clip_era5, None),
    "landsat8": _Modality("*.tar", False, clippers.clip_landsat, footprints.landsat_footprint),
    "landsat9": _Modality("*.tar", False, clippers.clip_landsat, footprints.landsat_footprint),
    "sentinel2": _Modality("*.zip", False, clippers.clip_sentinel2, _s2_footprint),
    "sentinel3": _Modality("*.zip", False, clippers.clip_sentinel3, _s3_footprint),
    "modis": _Modality(
        "*.hdf", True, clippers.clip_sinusoidal, footprints.sinusoidal_tile_footprint
    ),
    "viirs": _Modality(
        "*.h5", True, clippers.clip_sinusoidal, footprints.sinusoidal_tile_footprint
    ),
}


def _dry_run_row(
    *, src_path: Path, source: str, aoi_4326: Polygon, modality: _Modality, settings: ClipSettings
) -> ManifestRow:
    """Evaluate only the metadata gate for one product (no clip, no write)."""
    footprint = modality.gate_footprint(src_path) if modality.gate_footprint else None
    if footprint is None:
        gate = GateResult(ClipAction.CLIP, True, 0.0)  # unknown footprint: would clip
    else:
        gate = evaluate_gate(
            footprint_4326=footprint,
            aoi_4326=aoi_4326,
            min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
        )
    return ManifestRow(
        product_id=src_path.stem,
        source=source,
        footprint_bbox=bbox_str(footprint.bounds if footprint else None),
        intersects=gate.intersects,
        aoi_overlap_km2=round(gate.aoi_overlap_km2, 6),
        valid_pixel_count=0,
        action=gate.action,
    )


def clip_one_source(
    *,
    source: str,
    input_dir: Path,
    output_dir: Path,
    aoi_4326: Polygon,
    settings: ClipSettings,
    dry_run: bool = False,
) -> list[ManifestRow]:
    """Clip every product in one source directory; return its manifest rows.

    Args:
        source: Source directory name (must be a key of :data:`MODALITIES`).
        input_dir: Root of the raw archive.
        output_dir: Root of the clipped archive.
        aoi_4326: Authoritative AOI polygon (EPSG:4326).
        settings: Clip-stage thresholds.
        dry_run: When True, only the metadata gate runs — no pixels are decoded
            and no output files are written.

    Returns:
        One :class:`ManifestRow` per input product.

    Raises:
        KeyError: If ``source`` is not a recognised modality.
        FileNotFoundError: If the source directory does not exist.
    """
    modality = MODALITIES[source]
    src_root = input_dir / source
    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    out_root = output_dir / source
    products = sorted(src_root.glob(modality.glob))
    logger.info("clip-source", source=source, products=len(products), dry_run=dry_run)

    rows: list[ManifestRow] = []
    for product in products:
        if dry_run:
            rows.append(
                _dry_run_row(
                    src_path=product,
                    source=source,
                    aoi_4326=aoi_4326,
                    modality=modality,
                    settings=settings,
                )
            )
            continue

        if modality.per_grid_dir:
            row = modality.clip(
                src_path=product,
                dst_dir=out_root,
                source=source,
                aoi_4326=aoi_4326,
                settings=settings,
            )
        else:
            rel = product.relative_to(src_root)
            dst_path = out_root / rel
            row = modality.clip(
                src_path=product,
                dst_path=dst_path,
                source=source,
                aoi_4326=aoi_4326,
                settings=settings,
            )
            # Restore the column's documented contract: output_path is the path
            # RELATIVE TO THE SOURCE ROOT, not a basename. Clippers emit
            # ``dst_path.name``, which collapses subdir-nested products that share a
            # basename (ERA5: the same var.nc lives under 202503/202504/202505) into
            # indistinguishable rows. Use the real relative path so every CLIP row
            # is uniquely resolvable. (Phase-0 finding F4.)
            if row.action is ClipAction.CLIP:
                row.output_path = rel.as_posix()
        rows.append(row)
    return rows
