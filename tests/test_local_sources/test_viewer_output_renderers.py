"""Tests for the cube-band + daily-FSC renderers (PLAN-V2 §8).

Synthetic 32611 GeoTIFFs only — no Solara/leafmap, no real archive. Verifies the
``QuicklookResult`` contract (georef, bounds, RGB shape) and the FSC nodata→transparent
handling, plus the round-trip through ``result_to_geotiff``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from src.data.local_sources.viewer.manifest import _discover_s1_products
from src.data.local_sources.viewer.quicklook import RENDERERS
from src.data.local_sources.viewer.renderers import (
    render_cube_band,
    render_fsc,
    result_to_geotiff,
)
from src.data.local_sources.viewer.settings import ViewerSettings

# EPSG:32611 (UTM 11N) grid — production cube/FSC CRS.
_TRANSFORM = Affine(10.0, 0.0, 547_000.0, 0.0, -10.0, 5_620_000.0)
_NODATA = -9999.0


def _write_cube(path: Path, band_arrays: dict[str, np.ndarray]) -> None:
    """Write a described multi-band 32611 cube from ``{description: HxW array}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descs = list(band_arrays)
    h, w = next(iter(band_arrays.values())).shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=len(descs),
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
        nodata=_NODATA,
    ) as dst:
        for i, desc in enumerate(descs, start=1):
            dst.write(band_arrays[desc].astype("float32"), i)
            dst.set_band_description(i, desc)


def _write_fsc(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
        nodata=_NODATA,
    ) as dst:
        dst.write(arr.astype("float32"), 1)


# --------------------------------------------------------------------------- #
# render_cube_band
# --------------------------------------------------------------------------- #


def test_render_cube_band_dynamic_returns_georef(tmp_path: Path) -> None:
    grad = np.tile(np.linspace(-30, 5, 16, dtype="float32"), (16, 1))
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"VV_t0": grad, "VV_t1": grad * 0.5, "DEM": grad + 100})

    result = render_cube_band(path=path, var="VV", timestep=1, is_static=False, long_edge=64)

    assert result.kind == "georef_raster"
    assert result.bounds_4326 is not None
    assert result.src_crs is not None and "32611" in result.src_crs
    assert result.image.dtype == np.uint8  # stretched diagnostic look
    assert result.label == "VV @ t1"


def test_render_cube_band_static_label(tmp_path: Path) -> None:
    arr = np.tile(np.linspace(1000, 3000, 16, dtype="float32"), (16, 1))
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"VV_t0": arr, "DEM": arr})

    result = render_cube_band(path=path, var="DEM", timestep=0, is_static=True, long_edge=64)
    assert result.label == "DEM (static)"
    assert result.kind == "georef_raster"


def test_render_cube_band_masks_nodata(tmp_path: Path) -> None:
    arr = np.full((16, 16), 0.5, dtype="float32")
    arr[:8, :] = _NODATA  # top half is nodata
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"B2_t0": arr, "DEM": arr})

    result = render_cube_band(path=path, var="B2", timestep=0, is_static=False, long_edge=64)
    # The nodata half stretches to 0 (NaN→0 in _stretch_uint8); valid half is uniform.
    assert result.image.shape[0] > 0


def test_render_cube_band_uniform_valid_stays_opaque(tmp_path: Path) -> None:
    """A uniform-valued valid band must render opaque, not be dropped as nodata.

    Regression: a constant field (e.g. ERA5 total_precipitation_sum) and dark-but-valid
    pixels (S3 radiance) stretch to 0 in ``_stretch_uint8`` — the old all-zero-RGB alpha
    heuristic dropped every such pixel as transparent, so the whole band read as nodata
    (ERA5) or pocked with false holes (S3). The explicit ``alpha_mask`` (real NaN-nodata)
    must keep the valid pixels opaque even though they stretch to 0.
    """
    arr = np.full((16, 16), 0.5, dtype="float32")  # constant valid field → stretches to 0
    arr[:8, :] = _NODATA  # top half genuine nodata
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"B2_t0": arr, "DEM": arr})

    result = render_cube_band(path=path, var="B2", timestep=0, is_static=False, long_edge=64)

    # The renderer must hand the writer a real validity mask, not rely on pixel value.
    assert result.alpha_mask is not None
    assert result.alpha_mask.shape == result.image.shape[:2]

    out = result_to_geotiff(result, tmp_path / "out.tif")
    with rasterio.open(out) as src:
        assert src.count == 2  # gray + alpha
        assert src.colorinterp[-1].name == "alpha"
        alpha = src.read(2)

    # Alpha must track the validity mask: every valid pixel opaque (even the zero-stretched
    # uniform half), every nodata pixel transparent.
    valid = result.alpha_mask
    assert np.all(alpha[valid] > 0), "valid pixels were falsely dropped as nodata"
    assert np.all(alpha[~valid] == 0), "nodata pixels were not made transparent"


# --------------------------------------------------------------------------- #
# render_fsc
# --------------------------------------------------------------------------- #


def test_render_fsc_fixed_colormap_rgb(tmp_path: Path) -> None:
    fsc = np.tile(np.linspace(0.0, 1.0, 16, dtype="float32"), (16, 1))
    path = tmp_path / "fsc_20250519.tif"
    _write_fsc(path, fsc)

    result = render_fsc(path=path, long_edge=64)

    assert result.kind == "georef_raster"
    assert result.image.ndim == 3 and result.image.shape[-1] == 3  # RGB
    assert result.image.dtype == np.uint8
    assert result.bounds_4326 is not None
    assert "FSC" in result.label


def test_render_fsc_nodata_becomes_black_for_transparency(tmp_path: Path) -> None:
    fsc = np.full((16, 16), 0.0, dtype="float32")  # valid FSC=0 everywhere
    fsc[:8, :] = _NODATA  # top half nodata
    path = tmp_path / "fsc_20250519.tif"
    _write_fsc(path, fsc)

    result = render_fsc(path=path, long_edge=64)
    rgb = result.image
    # Valid FSC=0 → viridis(0) = (68,1,84), NOT pure black → survives the alpha heuristic.
    valid_rows = rgb[rgb.shape[0] // 2 :]
    assert valid_rows.max() > 0, "valid FSC=0 must not be pure black"
    # Nodata rows must be pure black (so result_to_geotiff makes them transparent).
    nodata_rows = rgb[: rgb.shape[0] // 4]  # safely inside the nodata band post-decimation
    assert nodata_rows.max() == 0, "nodata must be pure black for transparency"


def test_render_fsc_roundtrips_through_geotiff(tmp_path: Path) -> None:
    fsc = np.tile(np.linspace(0.0, 1.0, 16, dtype="float32"), (16, 1))
    fsc[0, :] = _NODATA
    path = tmp_path / "fsc_20250519.tif"
    _write_fsc(path, fsc)

    result = render_fsc(path=path, long_edge=64)
    out = result_to_geotiff(result, tmp_path / "out.tif")

    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 4326
        assert src.count == 4  # RGB + alpha
        assert src.colorinterp[-1].name == "alpha"


# --------------------------------------------------------------------------- #
# Sentinel-1 — processed SNAP tif (not clipped): discovery + renderer
# --------------------------------------------------------------------------- #
_S1_STEM = "s1_grd_S1C_IW_GRDH_1SDV_20250519T012124_20250519T012149_002393_005073_3D71"


def _write_s1_snap(path: Path) -> None:
    """Write a 3-band processed-S1 SNAP tif: VH(1), VV(2) linear σ⁰, angle(3) deg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = w = 16
    vh = np.full((h, w), 10.0 ** (-1.4), dtype="float32")  # ~-14 dB linear
    vv = np.full((h, w), 10.0 ** (-0.8), dtype="float32")  # ~-8 dB linear
    angle = np.full((h, w), 43.6, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="float32",
        crs="EPSG:32611",
        transform=_TRANSFORM,
    ) as dst:
        dst.write(vh, 1)
        dst.write(vv, 2)
        dst.write(angle, 3)


def test_discover_s1_products_from_snap_cache(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """S1 products are synthesized from sentinel1_snap/ tifs (no clip manifest rows)."""
    snap_dir = tmp_path / "clipped" / "sentinel1_snap"
    _write_s1_snap(snap_dir / f"{_S1_STEM}.tif")
    settings = ViewerSettings(clipped_root=tmp_path / "clipped")

    rows = _discover_s1_products(settings)

    assert len(rows) == 1
    row = rows[0]
    assert row.source == "sentinel1"
    assert row.action == "CLIP"
    assert row.path is not None and row.path.name == f"{_S1_STEM}.tif"
    assert "2025-05-19" in row.product_id  # acq date parsed from the stem
    # bbox is a real 4326 extent reprojected from the 32611 tif.
    minx, miny, maxx, maxy = row.footprint_bbox
    assert minx < maxx and miny < maxy and -180 <= minx <= 180


def test_discover_s1_products_empty_when_no_cache(tmp_path: Path) -> None:
    """No SNAP cache dir → no S1 products (S1 simply not processed yet), no error."""
    settings = ViewerSettings(clipped_root=tmp_path / "clipped")  # dir absent
    assert _discover_s1_products(settings) == []


def test_sentinel1_renderer_reads_processed_tif(tmp_path: Path) -> None:
    """The S1 renderer reads VV (band 2) from the processed tif → dB georef raster."""
    path = tmp_path / f"{_S1_STEM}.tif"
    _write_s1_snap(path)
    from src.data.local_sources.viewer.manifest import ProductRow

    row = ProductRow(
        product_id="S1 test",
        source="sentinel1",
        footprint_bbox=(-116.5, 50.7, -114.5, 52.3),
        intersects=True,
        aoi_overlap_km2=0.0,
        valid_pixel_count=0,
        action="CLIP",
        path=path,
    )
    result = RENDERERS["sentinel1"].render(row, long_edge=64)

    assert result.kind == "georef_raster"
    assert result.src_crs is not None and "32611" in result.src_crs
    assert result.image.dtype == np.uint8
    assert result.bounds_4326 is not None
    assert "VV" in result.label and "dB" in result.label
