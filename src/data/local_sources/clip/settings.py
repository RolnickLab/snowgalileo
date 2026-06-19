"""Configuration and AOI loading for the clip stage.

All thresholds are pydantic-settings fields (env-overridable via the ``CLIP_``
prefix), never magic numbers scattered in the clip routines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from shapely.geometry import Polygon, shape

#: Authoritative geographic CRS of ``data/bow_valley_inference_aoi.geojson`` and of every footprint
#: comparison performed by the gate.
AOI_CRS = "EPSG:4326"

#: MODIS / VIIRS sinusoidal projection (sphere of radius 6371007.181 m).
SINUSOIDAL_PROJ4 = "+proj=sinu +R=6371007.181 +units=m +no_defs"


class ClipSettings(BaseSettings):
    """Tunable thresholds for the AOI clip stage.

    Overridable from the environment with the ``CLIP_`` prefix, e.g.
    ``CLIP_MIN_AOI_OVERLAP_AREA_KM2=2.5``.

    Attributes:
        min_aoi_overlap_area_km2: Minimum footprint∩AOI area (km²) for a product
            to be clipped. Below this, the product is skipped as
            ``SKIP_DEGENERATE_OVERLAP`` — it is smaller than a single grid cell
            (1 km × 1 km) and can populate no full cell.
        require_valid_pixels: When True, a clip yielding zero non-nodata pixels
            is suppressed (no output file) even if its footprint passed the area
            test — a degenerate sliver on the product's own border nodata.
        swath_buffer_pixels: Padding (pixels) added around the AOI-overlapping
            grid window for Sentinel-3 tie-point-grid slicing.
        era5_pad_degrees: Buffer (degrees) added to the AOI bounds before the
            ERA5-Land lat/lon ``sel`` slice. ERA5 clips by **pixel centre** (xarray
            label slice), so without a pad the southernmost kept centre can sit
            *inside* the AOI, leaving edge cells beyond that pixel's nearest-resample
            catchment (centre ± half-resolution) with no source → all-nodata. Must be
            ≥ one native ERA5-Land pixel (0.1°); the default 0.15° clears a full pixel
            plus slack on every side. Mirrors GEE, which samples the unbounded global
            ERA5 collection (our clip is the only artificial boundary).
    """

    model_config = SettingsConfigDict(env_prefix="CLIP_", frozen=True)

    min_aoi_overlap_area_km2: Annotated[float, Field(gt=0)] = 1.0
    require_valid_pixels: bool = True
    swath_buffer_pixels: Annotated[int, Field(ge=0)] = 10
    era5_pad_degrees: Annotated[float, Field(ge=0.1)] = 0.15


def load_aoi_polygon(aoi_path: Path) -> Polygon:
    """Load the authoritative AOI polygon from a GeoJSON file.

    Args:
        aoi_path: Path to ``data/bow_valley_inference_aoi.geojson`` (EPSG:4326, single Polygon
            feature).

    Returns:
        The AOI as a shapely ``Polygon`` in EPSG:4326.

    Raises:
        FileNotFoundError: If ``aoi_path`` does not exist.
        ValueError: If the file has no features or the first geometry is not a
            Polygon.
    """
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    geojson = json.loads(aoi_path.read_text())
    features = geojson.get("features", [])
    if not features:
        raise ValueError(f"No features in AOI GeoJSON: {aoi_path}")

    geom = features[0].get("geometry")
    if geom is None or geom.get("type") != "Polygon":
        raise ValueError(f"AOI first feature must be a Polygon: {aoi_path}")

    polygon = shape(geom)
    if not isinstance(polygon, Polygon):
        raise ValueError(f"AOI geometry did not resolve to a Polygon: {aoi_path}")
    return polygon
