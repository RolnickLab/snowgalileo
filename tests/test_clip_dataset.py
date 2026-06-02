"""Tests for the AOI clip stage (TASK-002 / CLIPPING_PLAN §2).

Two tiers:

* **Gate geometry** (always run): synthetic footprints exercise the §2.0
  intersect gate — no archive data required.
* **Real-archive clips** (skipped when ``data/bow_valley_selection_raw`` is
  absent): end-to-end clip of a Landsat scene, a MODIS tile (per-grid index
  ratio), DEM CRS preservation, non-destructive pixel equality, and the
  post-run audit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rasterio
from shapely.geometry import Polygon, box

from src.data.local_sources.clip import clippers
from src.data.local_sources.clip.footprints import _parse_gml_coordinates
from src.data.local_sources.clip.gate import (
    ClipAction,
    evaluate_gate,
    geodesic_area_km2,
)
from src.data.local_sources.clip.gdal_io import _parse_grid_band
from src.data.local_sources.clip.settings import ClipSettings, load_aoi_polygon

REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = REPO_ROOT / "data" / "bow_valley_selection_raw"
AOI_PATH = REPO_ROOT / "data" / "aoi.geojson"

requires_archive = pytest.mark.skipif(
    not RAW_ROOT.exists(), reason="raw archive not present on this machine"
)


@pytest.fixture(scope="module")
def aoi() -> Polygon:
    return load_aoi_polygon(AOI_PATH)


@pytest.fixture(scope="module")
def settings() -> ClipSettings:
    return ClipSettings()


# --------------------------------------------------------------------------- #
# Gate geometry (no archive needed)
# --------------------------------------------------------------------------- #
def test_gate_skips_disjoint_footprint(aoi, settings):
    """AC-1: a footprint fully outside the AOI yields SKIP_NO_OVERLAP."""
    far = box(10.0, 10.0, 11.0, 11.0)
    result = evaluate_gate(
        footprint_4326=far,
        aoi_4326=aoi,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    assert result.action is ClipAction.SKIP_NO_OVERLAP
    assert result.intersects is False
    assert result.aoi_overlap_km2 == 0.0


def test_gate_skips_degenerate_overlap(aoi, settings):
    """AC-2: a sub-threshold overlap yields SKIP_DEGENERATE_OVERLAP."""
    lon_min, lat_min, lon_max, lat_max = aoi.bounds
    sliver = box(lon_min, lat_max - 0.001, lon_min + 0.001, lat_max)
    result = evaluate_gate(
        footprint_4326=sliver,
        aoi_4326=aoi,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    assert result.action is ClipAction.SKIP_DEGENERATE_OVERLAP
    assert result.intersects is True
    assert 0.0 < result.aoi_overlap_km2 < settings.min_aoi_overlap_area_km2


def test_gate_clips_substantial_overlap(aoi, settings):
    """A footprint covering a large part of the AOI yields CLIP."""
    lon_min, lat_min, lon_max, lat_max = aoi.bounds
    big = box(lon_min, lat_min, (lon_min + lon_max) / 2, lat_max)
    result = evaluate_gate(
        footprint_4326=big,
        aoi_4326=aoi,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    assert result.action is ClipAction.CLIP
    assert result.aoi_overlap_km2 > settings.min_aoi_overlap_area_km2


def test_geodesic_area_positive(aoi):
    """The AOI has a plausible geodesic area (~24 800 km²)."""
    area = geodesic_area_km2(aoi)
    assert 20_000 < area < 30_000


# --------------------------------------------------------------------------- #
# Manifest GML footprint parsing (no archive needed)
#
# Regression guards for two SAFE-manifest dialects that the live full-archive
# run revealed the parser silently dropped (returning None -> every product
# wrongly SKIP_NO_OVERLAP):
#   * Sentinel-1: <gml:coordinates> with comma-within-pair "lat,lon lat,lon".
#   * Sentinel-3: <gml:posList> instead of <gml:coordinates>.
# --------------------------------------------------------------------------- #
def test_parse_gml_coordinates_s1_comma_pairs():
    """S1 manifest lists ``lat,lon`` comma-pairs; parser must yield (lon, lat)."""
    xml = (
        "<safe:footPrint><gml:coordinates>"
        "51.355087,-119.420021 51.760925,-115.716866 "
        "50.265179,-115.352760 49.861259,-118.940010"
        "</gml:coordinates></safe:footPrint>"
    )
    poly = _parse_gml_coordinates(xml)
    assert poly is not None
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    # Axis order: longitudes are the negative values, latitudes ~50.
    assert -119.5 < lon_min < -115.3
    assert 49.8 < lat_min < 51.8


def test_parse_gml_coordinates_s2_whitespace_pairs():
    """S2 manifest lists pure-whitespace ``lat lon`` pairs; must still parse."""
    xml = (
        "<gml:coordinates>"
        "51.94998 -115.40237 52.00999 -115.69954 "
        "52.01011 -115.69948 51.95001 -115.40240"
        "</gml:coordinates>"
    )
    poly = _parse_gml_coordinates(xml)
    assert poly is not None
    lon_min, _, lon_max, lat_max = poly.bounds
    assert -115.7 < lon_min < -115.39
    assert 51.9 < lat_max < 52.1


def test_parse_gml_coordinates_s3_poslist():
    """S3 manifest uses ``<gml:posList>`` (not coordinates); must parse it."""
    xml = (
        '<sentinel-safe:footPrint><gml:posList srsName="EPSG:4326">'
        "52.4625 -132.682 52.4015 -131.676 50.1045 -115.216 49.8966 -114.294"
        "</gml:posList></sentinel-safe:footPrint>"
    )
    poly = _parse_gml_coordinates(xml)
    assert poly is not None
    lon_min, lat_min, lon_max, lat_max = poly.bounds
    assert -132.7 < lon_min < -114.2
    assert 49.8 < lat_min < 52.5


def test_parse_gml_coordinates_returns_none_when_absent():
    """No coordinate element -> None (fail-safe toward SKIP_NO_OVERLAP)."""
    assert _parse_gml_coordinates("<manifest><noCoords/></manifest>") is None


# --------------------------------------------------------------------------- #
# HDF subdataset grid/band parsing (no archive needed)
#
# Regression guard for the HDF5 (VIIRS) descriptor, whose "://group/path" form
# the MODIS-shaped ``:``-split mangled — leaking quotes and slashes into output
# filenames and crashing ``gdal_translate`` on the first product.
# --------------------------------------------------------------------------- #
def test_parse_grid_band_hdf5_viirs():
    """VIIRS HDF5 ``://...`` descriptor yields clean grid/band identifiers."""
    name = (
        'HDF5:"data/.../VNP09GA.A2025060.h10v03.h5"://HDFEOS/GRIDS/'
        "VIIRS_Grid_1km_2D/Data_Fields/SurfReflect_M4_1"
    )
    grid, band = _parse_grid_band(name)
    assert grid == "VIIRS_Grid_1km_2D"
    assert band == "SurfReflect_M4_1"
    # No quote/slash may leak into tokens used to build output filenames.
    for token in (grid, band):
        assert '"' not in token and "/" not in token


def test_parse_grid_band_hdf4_modis():
    """MODIS HDF4 ``:``-delimited descriptor still parses (no regression)."""
    name = (
        'HDF4_EOS:EOS_GRID:"data/.../MOD09GA.A2025060.h10v03.hdf":'
        "MODIS_Grid_500m_2D:sur_refl_b01_1"
    )
    grid, band = _parse_grid_band(name)
    assert grid == "MODIS_Grid_500m_2D"
    assert band == "sur_refl_b01_1"


# --------------------------------------------------------------------------- #
# Real-archive clips
# --------------------------------------------------------------------------- #
@requires_archive
def test_landsat_clip_keeps_zone_and_pixels(tmp_path, aoi, settings):
    """AC-3 + AC-7: a partial Landsat scene clips with >0 valid px, stays 32612."""
    import tarfile

    src = sorted((RAW_ROOT / "landsat9").glob("*.tar"))[0]
    dst = tmp_path / "ls9.tar"
    row = clippers.clip_landsat(
        src_path=src, dst_path=dst, source="landsat9", aoi_4326=aoi, settings=settings
    )
    assert row.action is ClipAction.CLIP
    assert row.valid_pixel_count > 0
    assert dst.exists()

    with tarfile.open(dst) as tar:
        band = next(n for n in tar.getnames() if n.endswith("_B4.TIF"))
        tar.extract(band, path=tmp_path)
    with rasterio.open(tmp_path / band) as clipped:
        assert clipped.crs.to_epsg() == 32612


@requires_archive
def test_sentinel2_clip_stays_utm11(tmp_path, aoi, settings):
    """AC-7: clipped Sentinel-2 JP2 bands stay EPSG:32611."""
    import zipfile

    src = sorted((RAW_ROOT / "sentinel2").glob("*.zip"))[0]
    dst = tmp_path / "s2.zip"
    row = clippers.clip_sentinel2(
        src_path=src, dst_path=dst, source="sentinel2", aoi_4326=aoi, settings=settings
    )
    assert row.action is ClipAction.CLIP
    with zipfile.ZipFile(dst) as zf:
        jp2 = next(n for n in zf.namelist() if n.lower().endswith(".jp2"))
        zf.extract(jp2, path=tmp_path)
    with rasterio.open(tmp_path / jp2) as clipped:
        assert clipped.crs.to_epsg() == 32611


@requires_archive
def test_modis_per_grid_index_ratio(tmp_path, aoi, settings):
    """AC-6: MODIS 500 m grid clips to ~2× the 1 km grid dimensions (no 1200 clamp)."""
    src = sorted((RAW_ROOT / "modis").glob("*.hdf"))[0]
    row = clippers.clip_sinusoidal(
        src_path=src, dst_dir=tmp_path, source="modis", aoi_4326=aoi, settings=settings
    )
    assert row.action is ClipAction.CLIP

    out = tmp_path / src.stem
    km1 = next(out.glob("MODIS_Grid_1km_2D__*.tif"))
    m500 = next(out.glob("MODIS_Grid_500m_2D__*.tif"))
    with rasterio.open(km1) as r1, rasterio.open(m500) as r5:
        h1, w1 = r1.shape
        h5, w5 = r5.shape
    # 500 m grid must be ~2x the 1 km grid on both axes (allow ±2 px rounding).
    # The old hardcoded `min(1200, ...)` clamp + `*2` could not produce this
    # independent per-grid ratio; it derived the 500 m window from the 1 km one.
    assert abs(h5 - 2 * h1) <= 2
    assert abs(w5 - 2 * w1) <= 2
    assert h1 > 0 and w1 > 0 and h5 > 0 and w5 > 0


@requires_archive
def test_dem_clip_is_non_destructive(tmp_path, aoi, settings):
    """AC-8: clipped DEM pixels inside the AOI equal the raw pixels (no resample)."""
    tile = next(
        p for p in (RAW_ROOT / "dem").rglob("*N51*_DEM.tif") if p.is_file()
    )
    dst = tmp_path / "dem.tif"
    row = clippers.clip_geotiff(
        src_path=tile, dst_path=dst, source="dem", aoi_4326=aoi, settings=settings
    )
    assert row.action is ClipAction.CLIP

    with rasterio.open(tile) as src, rasterio.open(dst) as clipped:
        assert clipped.crs == src.crs
        # Sample the clipped centre and read the same lon/lat from the source.
        cx = (clipped.bounds.left + clipped.bounds.right) / 2
        cy = (clipped.bounds.top + clipped.bounds.bottom) / 2
        clipped_val = next(clipped.sample([(cx, cy)]))[0]
        src_val = next(src.sample([(cx, cy)]))[0]
    assert clipped_val == src_val


@requires_archive
def test_manifest_one_row_per_product(tmp_path, aoi, settings):
    """AC-5: clipping WorldCover yields one manifest row per input tile."""
    from src.data.local_sources.clip.orchestrator import clip_one_source

    rows = clip_one_source(
        source="worldcover",
        input_dir=RAW_ROOT,
        output_dir=tmp_path,
        aoi_4326=aoi,
        settings=settings,
    )
    n_tiles = len(list((RAW_ROOT / "worldcover").rglob("*_Map.tif")))
    assert len(rows) == n_tiles
    product_ids = [r.product_id for r in rows]
    assert len(product_ids) == len(set(product_ids))
    assert all(r.action in ClipAction for r in rows)


@requires_archive
def test_audit_passes_on_clipped_worldcover(tmp_path, aoi, settings):
    """AC-4: the post-run audit finds zero all-nodata outputs after a real clip."""
    from src.data.local_sources.clip.manifest import write_manifest
    from src.data.local_sources.clip.orchestrator import clip_one_source

    rows = clip_one_source(
        source="worldcover",
        input_dir=RAW_ROOT,
        output_dir=tmp_path,
        aoi_4326=aoi,
        settings=settings,
    )
    write_manifest(rows, tmp_path / "worldcover" / "clip_manifest.csv")

    # No clipped raster may be entirely nodata.
    for tif in (tmp_path / "worldcover").rglob("*.tif"):
        with rasterio.open(tif) as src:
            data = src.read()
            assert (data != src.nodata).any() if src.nodata is not None else data.size


@requires_archive
def test_viirs_clip_writes_per_grid_tifs(tmp_path, aoi, settings):
    """A VIIRS HDF5 granule clips to per-grid GeoTIFFs (HDF5 descriptor parse).

    Guards the HDF5 ``://group/path`` subdataset descriptor that crashed
    ``gdal_translate`` on the live run: the test fails fast if the parse
    regresses (no tiles written) and asserts the 500 m grid clips to ~2× the
    1 km grid, mirroring the MODIS per-grid check.
    """
    src = sorted((RAW_ROOT / "viirs").glob("*.h5"))[0]
    row = clippers.clip_sinusoidal(
        src_path=src, dst_dir=tmp_path, source="viirs", aoi_4326=aoi, settings=settings
    )
    assert row.action is ClipAction.CLIP

    out = tmp_path / src.stem
    km1 = next(out.glob("VIIRS_Grid_1km_2D__*.tif"))
    m500 = next(out.glob("VIIRS_Grid_500m_2D__*.tif"))
    with rasterio.open(km1) as r1, rasterio.open(m500) as r5:
        h1, w1 = r1.shape
        h5, w5 = r5.shape
    assert h1 > 0 and w1 > 0 and h5 > 0 and w5 > 0
    assert abs(h5 - 2 * h1) <= 2
    assert abs(w5 - 2 * w1) <= 2
