"""The mandatory two-stage intersect gate (CLIPPING_PLAN §2.0).

Every product passes through :func:`evaluate_gate` *before* any per-modality clip
runs. This is the one place footprint-vs-AOI filtering happens; adapters never
re-implement it. The gate is fail-safe toward skipping: a product that fails
produces **no output file at all**, keeping the clipped archive free of
zero-signal artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pyproj import Geod
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

# WGS84 ellipsoid — used to measure intersection area in km² independent of the
# product's native CRS (footprints are compared in EPSG:4326).
_GEOD = Geod(ellps="WGS84")


class ClipAction(str, Enum):
    """Outcome of the intersect gate for a single product."""

    CLIP = "CLIP"
    SKIP_NO_OVERLAP = "SKIP_NO_OVERLAP"
    SKIP_DEGENERATE_OVERLAP = "SKIP_DEGENERATE_OVERLAP"


@dataclass(frozen=True)
class GateResult:
    """Result of stage 1 of the gate (geometry-only, no pixels decoded).

    Attributes:
        action: ``CLIP`` if the footprint overlaps the AOI by at least
            ``min_aoi_overlap_area_km2``; otherwise a ``SKIP_*`` action.
        intersects: Whether footprint and AOI polygons intersect at all.
        aoi_overlap_km2: Area of footprint∩AOI in km² (0.0 when disjoint).
    """

    action: ClipAction
    intersects: bool
    aoi_overlap_km2: float


def geodesic_area_km2(polygon: BaseGeometry) -> float:
    """Geodesic area of a lon/lat polygon in km².

    Args:
        polygon: A shapely geometry in EPSG:4326. Empty geometries yield 0.0.

    Returns:
        The absolute geodesic area in square kilometres.
    """
    if polygon.is_empty:
        return 0.0
    area_m2, _ = _GEOD.geometry_area_perimeter(polygon)
    return abs(area_m2) / 1.0e6


def evaluate_gate(
    *,
    footprint_4326: Polygon,
    aoi_4326: Polygon,
    min_aoi_overlap_area_km2: float,
) -> GateResult:
    """Run stage 1 of the intersect gate (CLIPPING_PLAN §2.0 step 1–2a).

    Both polygons must be in EPSG:4326. Uses true polygon intersection — not a
    bbox-corner test — because swath footprints are non-rectangular.

    Args:
        footprint_4326: Product footprint polygon in EPSG:4326.
        aoi_4326: Authoritative AOI polygon in EPSG:4326.
        min_aoi_overlap_area_km2: Minimum useful overlap area (km²).

    Returns:
        A :class:`GateResult`. ``SKIP_NO_OVERLAP`` when the polygons do not
        intersect; ``SKIP_DEGENERATE_OVERLAP`` when they intersect but the
        overlap area is below the threshold; otherwise ``CLIP``.
    """
    if not footprint_4326.intersects(aoi_4326):
        return GateResult(action=ClipAction.SKIP_NO_OVERLAP, intersects=False, aoi_overlap_km2=0.0)

    overlap = footprint_4326.intersection(aoi_4326)
    overlap_km2 = geodesic_area_km2(overlap)

    if overlap_km2 < min_aoi_overlap_area_km2:
        return GateResult(
            action=ClipAction.SKIP_DEGENERATE_OVERLAP,
            intersects=True,
            aoi_overlap_km2=overlap_km2,
        )

    return GateResult(action=ClipAction.CLIP, intersects=True, aoi_overlap_km2=overlap_km2)
