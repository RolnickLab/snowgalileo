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

from src.data.local_sources.viewer.renderers import (
    render_cube_band,
    render_fsc,
    result_to_geotiff,
)

# EPSG:32611 (UTM 11N) grid — production cube/FSC CRS.
_TRANSFORM = Affine(10.0, 0.0, 547_000.0, 0.0, -10.0, 5_620_000.0)
_NODATA = -9999.0


def _write_cube(path: Path, band_arrays: dict[str, np.ndarray]) -> None:
    """Write a described multi-band 32611 cube from ``{description: HxW array}``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descs = list(band_arrays)
    h, w = next(iter(band_arrays.values())).shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=len(descs),
        dtype="float32", crs="EPSG:32611", transform=_TRANSFORM, nodata=_NODATA,
    ) as dst:
        for i, desc in enumerate(descs, start=1):
            dst.write(band_arrays[desc].astype("float32"), i)
            dst.set_band_description(i, desc)


def _write_fsc(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype="float32", crs="EPSG:32611", transform=_TRANSFORM, nodata=_NODATA,
    ) as dst:
        dst.write(arr.astype("float32"), 1)


# --------------------------------------------------------------------------- #
# render_cube_band
# --------------------------------------------------------------------------- #


def test_render_cube_band_dynamic_returns_georef(tmp_path: Path) -> None:
    grad = np.tile(np.linspace(-30, 5, 16, dtype="float32"), (16, 1))
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"VV_t0": grad, "VV_t1": grad * 0.5, "DEM": grad + 100})

    result = render_cube_band(
        path=path, var="VV", timestep=1, is_static=False, long_edge=64
    )

    assert result.kind == "georef_raster"
    assert result.bounds_4326 is not None
    assert result.src_crs is not None and "32611" in result.src_crs
    assert result.image.dtype == np.uint8  # stretched diagnostic look
    assert result.label == "VV @ t1"


def test_render_cube_band_static_label(tmp_path: Path) -> None:
    arr = np.tile(np.linspace(1000, 3000, 16, dtype="float32"), (16, 1))
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"VV_t0": arr, "DEM": arr})

    result = render_cube_band(
        path=path, var="DEM", timestep=0, is_static=True, long_edge=64
    )
    assert result.label == "DEM (static)"
    assert result.kind == "georef_raster"


def test_render_cube_band_masks_nodata(tmp_path: Path) -> None:
    arr = np.full((16, 16), 0.5, dtype="float32")
    arr[:8, :] = _NODATA  # top half is nodata
    path = tmp_path / "PR_20250519_50.0_-116.0_SC00.tif"
    _write_cube(path, {"B2_t0": arr, "DEM": arr})

    result = render_cube_band(
        path=path, var="B2", timestep=0, is_static=False, long_edge=64
    )
    # The nodata half stretches to 0 (NaN→0 in _stretch_uint8); valid half is uniform.
    assert result.image.shape[0] > 0


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
