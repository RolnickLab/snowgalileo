"""Phase-0 throwaway spike: prove GDAL can read each hard clipped format with
decimated reads, and that solara/leafmap import. NOT shipped — lives under docs/.

Run: uv run python docs/agents/planning/clip-viewer/spike_reads.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

ROOT = Path("data/clipped_bow_valley_selection_raw")
LONG_EDGE = 512  # decimation target for the spike


def _decimated_read(uri: str, band: int = 1) -> tuple[np.ndarray, str, tuple]:
    with rasterio.open(uri) as src:
        scale = max(src.width, src.height) / LONG_EDGE
        out_w = max(1, int(src.width / scale))
        out_h = max(1, int(src.height / scale))
        arr = src.read(band, out_shape=(out_h, out_w), resampling=Resampling.bilinear)
        crs = str(src.crs)
        return arr, crs, src.bounds


def spike_imports() -> None:
    import leafmap  # noqa: F401
    import solara  # noqa: F401

    print("[imports] solara + leafmap import OK")


def spike_geotiff() -> None:
    dem = next(ROOT.glob("dem/*.tif"))
    arr, crs, bounds = _decimated_read(str(dem))
    print(f"[dem] {dem.name} -> {arr.shape} crs={crs} valid={np.isfinite(arr).sum()}")


def spike_landsat_tar() -> None:
    tar = next(ROOT.glob("landsat9/*.tar"))
    # find a B4 member name
    import tarfile

    with tarfile.open(tar) as tf:
        b4 = next(n for n in tf.getnames() if n.upper().endswith("_B4.TIF"))
    uri = f"/vsitar/{tar}/{b4}"
    arr, crs, bounds = _decimated_read(uri)
    print(f"[landsat] {b4} via /vsitar/ -> {arr.shape} crs={crs} max={arr.max()}")


def spike_s2_jp2_zip() -> None:
    zp = next(ROOT.glob("sentinel2/*.zip"))
    import zipfile

    with zipfile.ZipFile(zp) as zf:
        b04 = next(n for n in zf.namelist() if n.endswith("_B04.jp2"))
    uri = f"/vsizip/{zp}/{b04}"
    arr, crs, bounds = _decimated_read(uri)
    print(f"[s2] {Path(b04).name} via /vsizip/ -> {arr.shape} crs={crs} max={arr.max()}")


def spike_s1_tiff_zip() -> None:
    zp = next(ROOT.glob("sentinel1/*.zip"))
    import zipfile

    with zipfile.ZipFile(zp) as zf:
        vv = next(n for n in zf.namelist() if "-vv-" in n and n.endswith(".tiff"))
    uri = f"/vsizip/{zp}/{vv}"
    arr, crs, bounds = _decimated_read(uri)
    print(f"[s1] {Path(vv).name} via /vsizip/ -> {arr.shape} crs={crs} max={arr.max()}")


def spike_era5() -> None:
    import rioxarray  # noqa: F401
    import xarray as xr

    f = next(ROOT.glob("era5/*_totalprecip.nc"))
    ds = xr.open_dataset(f)
    var = "tp"
    da = ds[var].isel(valid_time=0)
    print(
        f"[era5] {f.name} {var}[t=0] -> {tuple(da.sizes.values())} "
        f"lat={float(ds.latitude.min()):.2f}..{float(ds.latitude.max()):.2f}"
    )


def spike_s3() -> None:

    zp = next(ROOT.glob("sentinel3/*.zip"))
    import zipfile

    with zipfile.ZipFile(zp) as zf:
        rad = next(n for n in zf.namelist() if n.endswith("Oa08_radiance.nc"))
        geo = [n for n in zf.namelist() if n.endswith("geo_coordinates.nc")]
    print(
        f"[s3] {Path(rad).name} present; geo_coordinates present={bool(geo)} "
        f"(=> non-georeferenced quicklook path confirmed)"
    )


if __name__ == "__main__":
    for fn in (
        spike_imports,
        spike_geotiff,
        spike_landsat_tar,
        spike_s2_jp2_zip,
        spike_s1_tiff_zip,
        spike_era5,
        spike_s3,
    ):
        try:
            fn()
        except Exception as exc:  # spike: report, don't abort
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
