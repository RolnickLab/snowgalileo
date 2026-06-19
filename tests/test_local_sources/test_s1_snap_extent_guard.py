"""Extent sanity guard for S1 SNAP outputs (truncation defence).

An interrupted SNAP run can exit 0 yet write a tiny truncated raster; 3 such ~0.4%
slivers once got published as valid cache hits (the cache-hit logic then refused to
overwrite them). ``_output_extent_is_plausible`` rejects an output too small for the
AOI∩footprint overlap so the truncated tif is never published. These tests exercise the
guard directly with synthetic UTM GeoTIFFs — no SNAP, no gpt.
"""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from src.data.local_sources.s1_snap import (
    _MIN_EXTENT_RATIO,
    _aoi_region_wkt,
    _output_extent_is_plausible,
)

# A Bow-Valley-like AOI bbox in 4326 (lon/lat) and a footprint that covers all of it.
_AOI = box(-116.562, 50.73, -114.528, 52.307)
_FOOTPRINT_FULL = box(-117.0, 50.5, -114.0, 52.6)  # encloses the AOI


def _utm_bounds_of(geom_4326):
    """Reproject a 4326 geom to EPSG:32611 and return (minx, miny, maxx, maxy)."""
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True).transform
    return shapely_transform(to_utm, geom_4326).bounds


def _write_utm_tif(path, *, minx, miny, maxx, maxy, px=1000.0):
    """Write a 1-band EPSG:32611 GeoTIFF spanning the given UTM bbox (coarse px)."""
    width = max(1, int((maxx - minx) / px))
    height = max(1, int((maxy - miny) / px))
    transform = from_origin(minx, maxy, px, px)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=transform,
    ) as ds:
        ds.write(np.ones((height, width), dtype="float32"), 1)


def test_full_extent_output_is_plausible(tmp_path):
    """An output covering the AOI region passes the guard (publish)."""
    region_wkt = _aoi_region_wkt(_AOI)
    minx, miny, maxx, maxy = _utm_bounds_of(_AOI)
    tif = tmp_path / "full.tif"
    _write_utm_tif(tif, minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    assert _output_extent_is_plausible(
        out_tif=tif, aoi_4326=_AOI, footprint_4326=_FOOTPRINT_FULL, region_wkt=region_wkt
    )


def test_sliver_output_is_rejected(tmp_path):
    """A ~4×5.5 km sliver (the observed failure) is rejected as truncated."""
    region_wkt = _aoi_region_wkt(_AOI)
    minx, miny, _, _ = _utm_bounds_of(_AOI)
    tif = tmp_path / "sliver.tif"
    # ~3.9 km × 5.5 km, matching the real truncated outputs.
    _write_utm_tif(tif, minx=minx, miny=miny, maxx=minx + 3900, maxy=miny + 5500, px=100.0)

    assert not _output_extent_is_plausible(
        out_tif=tif, aoi_4326=_AOI, footprint_4326=_FOOTPRINT_FULL, region_wkt=region_wkt
    )


def test_partial_swath_above_threshold_passes(tmp_path):
    """A legitimately partial swath (≥ min ratio of the overlap) still passes."""
    region_wkt = _aoi_region_wkt(_AOI)
    minx, miny, maxx, maxy = _utm_bounds_of(_AOI)
    # Cover ~40% of the region in x — above the 0.25 floor.
    tif = tmp_path / "partial.tif"
    _write_utm_tif(tif, minx=minx, miny=miny, maxx=minx + 0.40 * (maxx - minx), maxy=maxy)

    assert _MIN_EXTENT_RATIO < 0.40  # sanity: the test's 0.40 is genuinely above the floor
    assert _output_extent_is_plausible(
        out_tif=tif, aoi_4326=_AOI, footprint_4326=_FOOTPRINT_FULL, region_wkt=region_wkt
    )


def test_no_footprint_falls_back_to_region(tmp_path):
    """With an unreadable footprint, the guard falls back to the region area and still
    rejects a sliver.
    """
    region_wkt = _aoi_region_wkt(_AOI)
    minx, miny, _, _ = _utm_bounds_of(_AOI)
    tif = tmp_path / "sliver_nofp.tif"
    _write_utm_tif(tif, minx=minx, miny=miny, maxx=minx + 3900, maxy=miny + 5500, px=100.0)

    assert not _output_extent_is_plausible(
        out_tif=tif, aoi_4326=_AOI, footprint_4326=None, region_wkt=region_wkt
    )


def test_unreadable_output_is_rejected(tmp_path):
    """A non-raster / corrupt output (rasterio can't open it) is rejected, not raised."""
    region_wkt = _aoi_region_wkt(_AOI)
    bogus = tmp_path / "corrupt.tif"
    bogus.write_bytes(b"not-a-geotiff")  # SNAP exited 0 but wrote garbage

    assert not _output_extent_is_plausible(
        out_tif=bogus, aoi_4326=_AOI, footprint_4326=_FOOTPRINT_FULL, region_wkt=region_wkt
    )
