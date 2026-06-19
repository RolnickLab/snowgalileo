"""Load and resolve products for the viewer.

The clip manifest (``clip_manifest.csv``) is the source of truth for the **clipped**
modalities (Phase-0 finding F1). ``output_path`` semantics vary by modality:

* flat file directly under ``<root>/<source>/`` (landsat tar, s2/s3 zip, era5 nc);
* a *basename* whose real file is nested several dirs deep (DEM) — resolved via
  ``rglob``;
* a *directory* of per-grid GeoTIFFs (MODIS/VIIRS) — kept as the dir, the renderer
  picks a representative band inside.

**Sentinel-1 is the exception**: it is *processed* (ESA SNAP), not clipped, so it has
no manifest rows. Its products are discovered directly from the per-granule SNAP cache
(``sentinel1_snap/s1_grd_*.tif``) by :func:`_discover_s1_products` and appended — the
same processed tifs the cube ``S1Adapter`` reads.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import rasterio
import structlog
from pyproj import Transformer

from snow_galileo.data.local_sources.viewer.settings import ViewerSettings

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


def _resolve_path(*, source: str, output_path: str, action: str, root: Path) -> Path | None:
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
            logger.warning("multiple_path_matches", source=source, name=output_path, n=len(hits))
        return hits[0]

    logger.warning("unresolved_product", source=source, output_path=output_path)
    return None


#: Acquisition date token in a processed S1 cache stem (``s1_grd_S1C_..._<YYYYMMDD>T...``).
_S1_SNAP_ACQ = re.compile(r"_(\d{8})T\d{6}_")


def _tif_bbox_4326(path: Path) -> tuple[float, float, float, float]:
    """Return a raster's bounds as an EPSG:4326 ``(minx, miny, maxx, maxy)`` bbox."""
    with rasterio.open(path) as src:
        b = src.bounds
        if src.crs is None or src.crs.to_epsg() == 4326:
            return (b.left, b.bottom, b.right, b.top)
        tr = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        xs, ys = tr.transform(
            [b.left, b.right, b.left, b.right], [b.bottom, b.bottom, b.top, b.top]
        )
    return (min(xs), min(ys), max(xs), max(ys))


def _discover_s1_products(settings: ViewerSettings) -> list[ProductRow]:
    """Synthesize ``ProductRow``s for processed S1 directly from the SNAP cache dir.

    S1 is processed (ESA SNAP), not clipped, so it has no clip-manifest rows. Each
    per-granule ``s1_grd_*.tif`` becomes one ``CLIP`` ``ProductRow`` (``source="sentinel1"``)
    pointing at the processed tif — what the viewer's S1 renderer reads. Returns ``[]`` if
    the SNAP cache dir does not exist (S1 not processed yet).
    """
    snap_dir = settings.s1_snap_dir
    if not snap_dir.exists():
        return []
    rows: list[ProductRow] = []
    for tif in sorted(snap_dir.glob("s1_grd_*.tif")):
        m = _S1_SNAP_ACQ.search(tif.stem)
        acq = (
            datetime.datetime.strptime(m.group(1), "%Y%m%d").date().isoformat() if m else tif.stem
        )
        try:
            bbox = _tif_bbox_4326(tif)
        except rasterio.errors.RasterioIOError:
            logger.warning("s1_snap_unreadable", path=str(tif))
            continue
        rows.append(
            ProductRow(
                product_id=f"S1 {acq} ({tif.stem})",
                source="sentinel1",
                footprint_bbox=bbox,
                intersects=True,
                aoi_overlap_km2=0.0,  # not gated by area — processed, not clipped
                valid_pixel_count=0,  # unknown/irrelevant for the processed product
                action="CLIP",
                path=tif,
            )
        )
    return rows


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
        # Skip any ``sentinel1`` manifest rows. S1 is now *processed* (ESA SNAP), not
        # clipped, so the live product is the per-granule ``sentinel1_snap/s1_grd_*.tif``
        # discovered below — never the clip manifest. Some manifests still carry **legacy**
        # S1 rows whose ``output_path`` points at the raw ``sentinel1/*.zip`` SAFE archive
        # (a leftover from before the SNAP migration); those zips are not a raster, so the
        # renderer raised ``RasterioIOError`` and fell back to a broken ``plain_image``.
        # Honour the documented contract ("S1 has no manifest rows") at the viewer boundary,
        # robust to whatever the manifest happens to contain.
        if str(record["source"]) == "sentinel1":
            continue
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
    # S1 is processed (SNAP), not clipped — any clip-manifest S1 rows were skipped above.
    # Discover its live products from the SNAP cache dir and append them so the viewer lists
    # processed S1.
    rows.extend(_discover_s1_products(settings))
    return rows
