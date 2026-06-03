"""Load and resolve products from ``clip_manifest.csv``.

The manifest is the single source of truth for product identity and location
(Phase-0 finding F1). ``output_path`` semantics vary by modality:

* flat file directly under ``<root>/<source>/`` (landsat tar, s2/s1/s3 zip, era5 nc);
* a *basename* whose real file is nested several dirs deep (DEM) — resolved via
  ``rglob``;
* a *directory* of per-grid GeoTIFFs (MODIS/VIIRS) — kept as the dir, the renderer
  picks a representative band inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import structlog

from src.viewer.settings import ViewerSettings

logger = structlog.get_logger(__name__)

Action = Literal["CLIP", "SKIP_NO_OVERLAP", "SKIP_DEGENERATE_OVERLAP"]


@dataclass(frozen=True)
class ProductRow:
    """One manifest row resolved to an on-disk location.

    Attributes:
        product_id: Stable product identifier from the manifest.
        source: Modality key (``dem``, ``landsat9``, ``sentinel2``, ...).
        footprint_bbox: ``(minx, miny, maxx, maxy)`` in EPSG:4326.
        intersects: Whether the footprint intersects the AOI.
        aoi_overlap_km2: Overlap area with the AOI in km².
        valid_pixel_count: Valid (non-nodata) pixel count after clip.
        action: Clip-stage decision.
        path: Resolved file or directory; ``None`` for SKIP rows.
    """

    product_id: str
    source: str
    footprint_bbox: tuple[float, float, float, float]
    intersects: bool
    aoi_overlap_km2: float
    valid_pixel_count: int
    action: Action
    path: Path | None


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = tuple(float(x) for x in str(raw).split(","))
    if len(parts) != 4:
        raise ValueError(f"footprint_bbox must have 4 values, got {raw!r}")
    return parts  # type: ignore[return-value]


def _resolve_path(
    *, source: str, output_path: str, action: str, root: Path
) -> Path | None:
    """Resolve a manifest ``output_path`` to a real file/dir under the clipped root.

    Returns ``None`` for SKIP rows (no output written) or when nothing is found.
    """
    if action != "CLIP" or not output_path or pd.isna(output_path):
        return None

    source_dir = root / source
    direct = source_dir / output_path
    if direct.exists():
        return direct

    # Nested case (DEM): output_path is a bare basename buried several dirs deep.
    hits = list(source_dir.rglob(Path(output_path).name))
    if hits:
        if len(hits) > 1:
            logger.warning(
                "multiple_path_matches", source=source, name=output_path, n=len(hits)
            )
        return hits[0]

    logger.warning("unresolved_product", source=source, output_path=output_path)
    return None


def load_products(settings: ViewerSettings | None = None) -> list[ProductRow]:
    """Read the clip manifest and resolve every row to a ``ProductRow``.

    Args:
        settings: Viewer settings; defaults to ``ViewerSettings()``.

    Returns:
        One ``ProductRow`` per manifest row, in file order.

    Raises:
        FileNotFoundError: If the manifest does not exist.
    """
    settings = settings or ViewerSettings()
    if not settings.manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {settings.manifest_path}")

    df = pd.read_csv(settings.manifest_path)
    rows: list[ProductRow] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            ProductRow(
                product_id=str(record["product_id"]),
                source=str(record["source"]),
                footprint_bbox=_parse_bbox(record["footprint_bbox"]),
                intersects=bool(record["intersects"]),
                aoi_overlap_km2=float(record["aoi_overlap_km2"]),
                valid_pixel_count=int(record["valid_pixel_count"]),
                action=str(record["action"]),  # type: ignore[arg-type]
                path=_resolve_path(
                    source=str(record["source"]),
                    output_path=record.get("output_path", ""),
                    action=str(record["action"]),
                    root=settings.clipped_root,
                ),
            )
        )
    return rows
