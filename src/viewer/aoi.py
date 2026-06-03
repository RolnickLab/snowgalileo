"""AOI loading for overlay on every georeferenced product."""

from __future__ import annotations

import json
from pathlib import Path

from src.viewer.settings import ViewerSettings


def load_aoi_geojson(settings: ViewerSettings | None = None) -> dict:
    """Load ``data/aoi.geojson`` as a GeoJSON dict for a leafmap layer.

    Args:
        settings: Viewer settings; defaults to ``ViewerSettings()``.

    Returns:
        Parsed GeoJSON FeatureCollection / geometry.

    Raises:
        FileNotFoundError: If the AOI file does not exist.
    """
    settings = settings or ViewerSettings()
    if not settings.aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {settings.aoi_path}")
    return json.loads(Path(settings.aoi_path).read_text())


def aoi_bounds_4326(geojson: dict) -> tuple[float, float, float, float]:
    """Compute ``(minx, miny, maxx, maxy)`` of a GeoJSON in EPSG:4326.

    Walks coordinate arrays directly to avoid a geopandas dependency for a bbox.
    """
    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords: object) -> None:
        if (
            isinstance(coords, (list, tuple))
            and len(coords) == 2
            and all(isinstance(c, (int, float)) for c in coords)
        ):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        if isinstance(coords, (list, tuple)):
            for c in coords:
                _walk(c)

    feats = geojson.get("features", [geojson])
    for feat in feats:
        geom = feat.get("geometry", feat)
        _walk(geom.get("coordinates", []))

    if not xs or not ys:
        raise ValueError("no coordinates found in AOI geojson")
    return (min(xs), min(ys), max(xs), max(ys))
