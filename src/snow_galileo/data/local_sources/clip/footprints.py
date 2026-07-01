"""Metadata-only footprint readers (CLIPPING_PLAN §2.0 step 1).

Each reader returns a footprint polygon in EPSG:4326 **without decoding pixel
data** — Landsat ``MTL.json`` corners, Sentinel ``manifest.safe`` / ``xfdumanifest.xml``
GML, MODIS/VIIRS sinusoidal tile bounds, NetCDF coordinate ranges. A reader that
cannot resolve a footprint returns ``None``; the caller treats that as
``SKIP_NO_OVERLAP`` (fail-safe toward skipping).
"""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import rasterio
import rasterio.errors
from pyproj import Transformer
from shapely.geometry import Polygon, box

from .settings import SINUSOIDAL_PROJ4


def _read_tar_member_text(tar_path: Path, suffix: str) -> Optional[str]:
    """Return the decoded text of the first tar member ending with ``suffix``."""
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(suffix):
                fh = tar.extractfile(member)
                if fh is None:
                    return None
                return fh.read().decode("utf-8", "ignore")
    return None


def _read_zip_member_text(zip_path: Path, name_contains: str) -> Optional[str]:
    """Return the decoded text of the first zip member containing ``name_contains``."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name_contains in name:
                return zf.read(name).decode("utf-8", "ignore")
    return None


def landsat_footprint(tar_path: Path) -> Optional[Polygon]:
    """Footprint polygon from a Landsat tar's ``MTL.json`` product corners.

    Args:
        tar_path: Path to a Landsat Collection-2 ``.tar`` archive.

    Returns:
        The scene footprint in EPSG:4326, or ``None`` if no ``MTL.json`` is found.
    """
    text = _read_tar_member_text(tar_path, "MTL.json")
    if text is None:
        return None
    attrs = json.loads(text)["LANDSAT_METADATA_FILE"]["PROJECTION_ATTRIBUTES"]
    corners = [
        (float(attrs["CORNER_UL_LON_PRODUCT"]), float(attrs["CORNER_UL_LAT_PRODUCT"])),
        (float(attrs["CORNER_UR_LON_PRODUCT"]), float(attrs["CORNER_UR_LAT_PRODUCT"])),
        (float(attrs["CORNER_LR_LON_PRODUCT"]), float(attrs["CORNER_LR_LAT_PRODUCT"])),
        (float(attrs["CORNER_LL_LON_PRODUCT"]), float(attrs["CORNER_LL_LAT_PRODUCT"])),
    ]
    return Polygon(corners)


def _parse_gml_coordinates(text: str) -> Optional[Polygon]:
    """Parse a SAFE manifest GML footprint into a ``(lon, lat)`` polygon.

    SAFE products encode the footprint with one of two GML elements, both listing
    ``lat`` before ``lon`` (the GML default axis order):

    * **Sentinel-1 / Sentinel-2**: ``<gml:coordinates>``. S1 uses
      comma-within-pair, space-between-pairs (``"lat,lon lat,lon ..."``); S2 uses
      pure whitespace (``"lat lon lat lon ..."``).
    * **Sentinel-3**: ``<gml:posList>`` with pure whitespace
      (``"lat lon lat lon ..."``).

    Both are normalised to a flat ``lat lon lat lon`` token list, then re-ordered
    to the ``(lon, lat)`` tuples shapely expects.

    Args:
        text: The manifest XML text.

    Returns:
        The footprint as a shapely ``Polygon`` in EPSG:4326, or ``None`` if
        neither element with a valid coordinate list is present.
    """
    match = re.search(r"<gml:coordinates>(.*?)</gml:coordinates>", text, re.DOTALL)
    if match is None:
        match = re.search(r"<gml:posList[^>]*>(.*?)</gml:posList>", text, re.DOTALL)
    if match is None:
        match = re.search(r"coordinates>(.*?)<", text, re.DOTALL)
    if match is None:
        return None
    # Normalise both conventions: commas (S1 intra-pair separator) become spaces,
    # then split on any whitespace into a flat scalar list.
    tokens = match.group(1).replace(",", " ").split()
    if len(tokens) < 6 or len(tokens) % 2 != 0:
        return None
    # SAFE GML lists coordinates as "lat lon lat lon ..."; shapely wants (lon, lat).
    coords = [(float(tokens[i + 1]), float(tokens[i])) for i in range(0, len(tokens), 2)]
    return Polygon(coords)


def sentinel_safe_footprint(zip_path: Path, manifest_name: str) -> Optional[Polygon]:
    """Footprint polygon from a Sentinel SAFE manifest GML footprint.

    Args:
        zip_path: Path to a Sentinel ``.zip`` archive (S1/S2/S3 SAFE product).
        manifest_name: Substring identifying the manifest member, e.g.
            ``"manifest.safe"`` (S1/S2) or ``"xfdumanifest.xml"`` (S3).

    Returns:
        The product footprint in EPSG:4326, or ``None`` if no GML footprint is
        present in the manifest.
    """
    text = _read_zip_member_text(zip_path, manifest_name)
    if text is None:
        return None
    return _parse_gml_coordinates(text)


def sinusoidal_tile_footprint(reference_path: Path) -> Optional[Polygon]:
    """Footprint of a MODIS/VIIRS sinusoidal tile, reprojected to EPSG:4326.

    Reads the tile's sinusoidal bounds from the file's geotransform/CRS via
    ``rasterio`` (or its first georeferenced subdataset) and reprojects the
    corner grid to lon/lat. Densified along each edge so the curved sinusoidal
    boundary is not under-sampled.

    Args:
        reference_path: Path to a MODIS HDF4 / VIIRS HDF5 file, or a single
            georeferenced sinusoidal GeoTIFF subdataset.

    Returns:
        The tile footprint in EPSG:4326, or ``None`` if no sinusoidal bounds can
        be resolved.
    """
    bounds = _sinusoidal_bounds(reference_path)
    if bounds is None:
        return None
    x_min, y_min, x_max, y_max = bounds

    transformer = Transformer.from_crs(SINUSOIDAL_PROJ4, "EPSG:4326", always_xy=True)
    steps = 20
    edge: list[tuple[float, float]] = []
    for i in range(steps + 1):  # bottom edge, west -> east
        edge.append((x_min + (x_max - x_min) * i / steps, y_min))
    for i in range(steps + 1):  # right edge, south -> north
        edge.append((x_max, y_min + (y_max - y_min) * i / steps))
    for i in range(steps + 1):  # top edge, east -> west
        edge.append((x_max - (x_max - x_min) * i / steps, y_max))
    for i in range(steps + 1):  # left edge, north -> south
        edge.append((x_min, y_max - (y_max - y_min) * i / steps))

    lonlat = [transformer.transform(x, y) for x, y in edge]
    return Polygon(lonlat)


def _sinusoidal_bounds(
    reference_path: Path,
) -> Optional[tuple[float, float, float, float]]:
    """Return ``(x_min, y_min, x_max, y_max)`` in sinusoidal metres for a tile.

    Tries ``rasterio`` first (works for HDF5/VIIRS and plain GeoTIFFs). Falls
    back to system ``gdalinfo`` for HDF4 containers, whose driver rasterio's GDAL
    build lacks.
    """
    try:
        with rasterio.open(reference_path) as src:
            if src.crs is not None and src.transform is not None and src.width:
                b = src.bounds
                return (b.left, b.bottom, b.right, b.top)
            subdatasets = list(src.subdatasets)
        for sub in subdatasets:  # HDF5 container: first georeferenced subdataset.
            with rasterio.open(sub) as sub_src:
                if sub_src.crs is not None and sub_src.transform is not None:
                    b = sub_src.bounds
                    return (b.left, b.bottom, b.right, b.top)
    except rasterio.errors.RasterioIOError:
        pass  # HDF4 (MODIS) — rasterio can't open it; use gdalinfo below.

    return _sinusoidal_bounds_via_gdalinfo(reference_path)


def _sinusoidal_bounds_via_gdalinfo(
    reference_path: Path,
) -> Optional[tuple[float, float, float, float]]:
    """``_sinusoidal_bounds`` fallback using system ``gdalinfo`` (HDF4)."""
    from .gdal_io import gdalinfo_json

    meta = gdalinfo_json(reference_path)
    subs = meta.get("metadata", {}).get("SUBDATASETS", {})
    names = [v for k, v in sorted(subs.items()) if k.endswith("_NAME")]
    for name in names:
        sub_meta = gdalinfo_json(name)
        corner = sub_meta.get("cornerCoordinates")
        if not corner:
            continue
        ul = corner["upperLeft"]
        lr = corner["lowerRight"]
        x_min, x_max = min(ul[0], lr[0]), max(ul[0], lr[0])
        y_min, y_max = min(ul[1], lr[1]), max(ul[1], lr[1])
        return (x_min, y_min, x_max, y_max)
    return None


def netcdf_footprint(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> Polygon:
    """Footprint of a regular lat/lon NetCDF grid (e.g. ERA5-Land) as a bbox.

    Args:
        lon_min: Western coordinate bound.
        lat_min: Southern coordinate bound.
        lon_max: Eastern coordinate bound.
        lat_max: Northern coordinate bound.

    Returns:
        The grid extent as an axis-aligned polygon in EPSG:4326.
    """
    return box(lon_min, lat_min, lon_max, lat_max)


def raster_footprint_4326(raster_path: Path) -> Optional[Polygon]:
    """Footprint of a georeferenced raster, reprojected to EPSG:4326.

    For DEM/WorldCover tiles whose CRS is already geographic this is just the
    bounds; for projected rasters the four corners are transformed to lon/lat.

    Args:
        raster_path: Path to a georeferenced raster readable by ``rasterio``.

    Returns:
        The raster footprint in EPSG:4326, or ``None`` if it is not georeferenced.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None or src.transform is None:
            return None
        b = src.bounds
        if src.crs.to_epsg() == 4326:
            return box(b.left, b.bottom, b.right, b.top)
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        corners = [
            transformer.transform(b.left, b.bottom),
            transformer.transform(b.right, b.bottom),
            transformer.transform(b.right, b.top),
            transformer.transform(b.left, b.top),
        ]
        return Polygon(corners)
