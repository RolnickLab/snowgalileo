"""Per-modality quicklook renderers (PLAN §5).

Phase 2 covers the plain-GeoTIFF set: ``dem``, ``worldcover``, ``modis``,
``viirs``. Each renderer reads decimated (overview/``out_shape``) — never full-res
(geospatial skill: no eager multi-GB loads) — returns native ``src_crs`` plus
``bounds_4326`` (transformed, never assumed), and is registered under its
``source`` key so the dispatcher in ``quicklook.py`` can find it.

Archive (landsat/S2/S1) and ERA5/S3 renderers land in later phases.
"""

from __future__ import annotations

import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
import xarray as xr
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import array_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from src.viewer.archives import (
    find_member,
    list_tar_members,
    list_zip_members,
    vsitar_path,
    vsizip_path,
)
from src.viewer.manifest import ProductRow
from src.viewer.quicklook import QuicklookResult, register

logger = structlog.get_logger(__name__)

# MODIS/VIIRS surface-reflectance scale + valid range (int16, scale 1e-4).
_SR_SCALE = 1.0e-4
_SR_FILL = -28672  # nodata sentinel in the SR products

# Finite nodata for float display GeoTIFFs. NaN cannot be used: the tile server's
# JSON metadata endpoint rejects non-finite floats (see result_to_geotiff).
_FLOAT_NODATA = -9999.0

# ESA WorldCover v200 class → RGB (the official discrete palette).
_WORLDCOVER_PALETTE: dict[int, tuple[int, int, int]] = {
    10: (0, 100, 0),  # tree cover
    20: (255, 187, 34),  # shrubland
    30: (255, 255, 76),  # grassland
    40: (240, 150, 255),  # cropland
    50: (250, 0, 0),  # built-up
    60: (180, 180, 180),  # bare / sparse veg
    70: (240, 240, 240),  # snow & ice
    80: (0, 100, 200),  # permanent water
    90: (0, 150, 160),  # herbaceous wetland
    95: (0, 207, 117),  # mangroves
    100: (250, 230, 160),  # moss & lichen
}


def _decimated_shape(*, width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Scale ``(height, width)`` so the long edge is ``≤ long_edge`` px.

    Returns ``(out_h, out_w)`` for a rasterio ``out_shape`` read.
    """
    longest = max(width, height)
    if longest <= long_edge:
        return (height, width)
    scale = long_edge / longest
    return (max(1, round(height * scale)), max(1, round(width * scale)))


def _bounds_4326(src: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    """Transform a dataset's native bounds to EPSG:4326 (CRS is law: never assume).

    Raises:
        ValueError: If the dataset carries no CRS to transform from.
    """
    if src.crs is None:
        raise ValueError("dataset has no CRS; cannot compute 4326 bounds")
    minx, miny, maxx, maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    return (minx, miny, maxx, maxy)


@dataclass(frozen=True)
class _Decimated:
    """A decimated read kept in its NATIVE grid (CRS + affine), pre-warp.

    Renderers reproject to EPSG:4326 with :func:`_to_4326` only when the source is
    not already 4326 (MODIS/VIIRS are MODIS Sinusoidal); writing a sinusoidal array
    onto a 4326 affine stretches it horizontally (the native E–W pixel size is not
    constant in degrees).
    """

    array: npt.NDArray[np.floating]  # HxW (band) or HxWxC
    transform: Affine
    crs: CRS
    bounds_4326: tuple[float, float, float, float]


def _read_band_decimated(path: str | Path, *, band: int, long_edge: int) -> _Decimated:
    """Read one band decimated, keeping the native grid (CRS + transform).

    ``path`` may be a real file path or a GDAL ``/vsizip/`` / ``/vsitar/`` string;
    the latter MUST be passed through as ``str`` — wrapping it in ``Path`` collapses
    the ``//`` after the ``/vsi*/`` prefix and GDAL can no longer find the archive.
    """
    with rasterio.open(path) as src:
        out_h, out_w = _decimated_shape(width=src.width, height=src.height, long_edge=long_edge)
        arr = src.read(
            band,
            out_shape=(out_h, out_w),
            resampling=Resampling.average,
            masked=True,
        ).astype("float32")
        # Affine of the *decimated* grid: scale the native transform by the read
        # ratio so it still maps pixels → native CRS coordinates.
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        crs = src.crs
        bounds_4326 = _bounds_4326(src)
    return _Decimated(
        array=np.ma.filled(arr, np.nan),
        transform=transform,
        crs=crs,
        bounds_4326=bounds_4326,
    )


def _to_4326(
    dec: _Decimated,
) -> tuple[npt.NDArray[np.floating], tuple[float, float, float, float]]:
    """Reproject a decimated native-grid array to an EPSG:4326 raster grid.

    Returns ``(array_4326, bounds_4326)`` where ``array_4326`` is a true 4326 grid
    (so a plain 4326 affine over its bounds is undistorted). Pass-through when the
    source is already 4326. Multi-band ``HxWxC`` arrays are warped band-by-band.
    """
    if dec.crs.to_epsg() == 4326:
        return dec.array, dec.bounds_4326

    src_h, src_w = dec.array.shape[:2]
    dst_transform, dst_w, dst_h = calculate_default_transform(
        dec.crs, "EPSG:4326", src_w, src_h, *array_bounds(src_h, src_w, dec.transform)
    )

    def _warp_one(band: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        out = np.full((dst_h, dst_w), np.nan, dtype="float32")
        reproject(
            source=np.ascontiguousarray(band),
            destination=out,
            src_transform=dec.transform,
            src_crs=dec.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        return out

    if dec.array.ndim == 2:
        warped = _warp_one(dec.array)
    else:
        warped = np.dstack([_warp_one(dec.array[..., i]) for i in range(dec.array.shape[-1])])

    bounds = array_bounds(dst_h, dst_w, dst_transform)
    return warped, bounds


def _read_rgb_decimated(
    paths: tuple[Path, Path, Path], *, long_edge: int
) -> tuple[npt.NDArray[np.uint8], tuple[float, float, float, float], str]:
    """Read three single-band SR files, reproject to 4326, stretch to uint8 RGB.

    All three bands must share the same grid (true for MODIS 500 m SR bands).
    """
    chans: list[npt.NDArray[np.floating]] = []
    bounds: tuple[float, float, float, float] | None = None
    crs = ""
    for p in paths:
        dec = _read_band_decimated(p, band=1, long_edge=long_edge)
        a = np.where(dec.array == _SR_FILL, np.nan, dec.array) * _SR_SCALE
        arr_4326, bounds = _to_4326(
            _Decimated(array=a, transform=dec.transform, crs=dec.crs, bounds_4326=dec.bounds_4326)
        )
        chans.append(arr_4326)
        crs = str(dec.crs)
    assert bounds is not None
    rgb = np.dstack(chans)
    return _stretch_uint8(rgb), bounds, crs


def _stretch_uint8(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    """2–98 percentile per-band contrast stretch to uint8, NaN → 0.

    Accepts ``HxW`` (single-band) or ``HxWxC`` (multi-band); the returned array
    keeps the input's dimensionality.
    """
    is_2d = arr.ndim == 2
    src = arr[..., None] if is_2d else arr
    out = np.zeros(src.shape, dtype=np.uint8)
    for i in range(src.shape[-1]):
        band = src[..., i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, (2, 98))
        if hi <= lo:
            continue
        scaled = np.clip((band - lo) / (hi - lo), 0, 1) * 255
        out[..., i] = np.nan_to_num(scaled).astype(np.uint8)
    return out[..., 0] if is_2d else out


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


class _DemRenderer:
    source = "dem"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        dec = _read_band_decimated(row.path, band=1, long_edge=long_edge)
        arr, bounds = _to_4326(dec)
        return QuicklookResult(
            kind="georef_raster",
            image=arr,
            bounds_4326=bounds,
            src_crs=str(dec.crs),
            label="DEM elevation (m)",
        )


class _WorldCoverRenderer:
    source = "worldcover"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        with rasterio.open(row.path) as src:
            out_h, out_w = _decimated_shape(
                width=src.width, height=src.height, long_edge=long_edge
            )
            classes = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
            bounds = _bounds_4326(src)
            crs = str(src.crs)
        rgb = np.zeros((*classes.shape, 3), dtype=np.uint8)
        for value, color in _WORLDCOVER_PALETTE.items():
            rgb[classes == value] = color
        return QuicklookResult(
            kind="georef_raster",
            image=rgb,
            bounds_4326=bounds,
            src_crs=crs,
            label="ESA WorldCover (class palette)",
        )


def _grid_band(directory: Path, token: str) -> Path:
    """Find the single GeoTIFF in a MODIS/VIIRS grid dir matching ``token``.

    Raises:
        FileNotFoundError: If no band matches ``token``.
    """
    hits = sorted(directory.glob(f"*{token}*.tif"))
    if not hits:
        raise FileNotFoundError(f"no band matching {token!r} in {directory}")
    return hits[0]


class _ModisRenderer:
    source = "modis"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        # True-ish color from 500 m SR: b01 (red), b04 (green), b03 (blue).
        paths = (
            _grid_band(row.path, "sur_refl_b01"),
            _grid_band(row.path, "sur_refl_b04"),
            _grid_band(row.path, "sur_refl_b03"),
        )
        rgb, bounds, crs = _read_rgb_decimated(paths, long_edge=long_edge)
        return QuicklookResult(
            kind="georef_raster",
            image=rgb,
            bounds_4326=bounds,
            src_crs=crs,
            label="MODIS SR true-color (b01/b04/b03)",
        )


class _ViirsRenderer:
    source = "viirs"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        # Single 500 m I1 SR band, stretched (PLAN §5).
        band = _grid_band(row.path, "SurfReflect_I1")
        dec = _read_band_decimated(band, band=1, long_edge=long_edge)
        scaled = np.where(dec.array == _SR_FILL, np.nan, dec.array) * _SR_SCALE
        arr_4326, bounds = _to_4326(
            _Decimated(
                array=scaled, transform=dec.transform, crs=dec.crs, bounds_4326=dec.bounds_4326
            )
        )
        return QuicklookResult(
            kind="georef_raster",
            image=_stretch_uint8(arr_4326),
            bounds_4326=bounds,
            src_crs=str(dec.crs),
            label="VIIRS SR I1 (stretched)",
        )


# --------------------------------------------------------------------------- #
# Phase 3 — archive renderers (Landsat tar, S2 zip, S1 zip via /vsi*/)
# --------------------------------------------------------------------------- #


def _read_rgb_paths_4326(
    paths: tuple[str, str, str], *, long_edge: int
) -> tuple[npt.NDArray[np.uint8], tuple[float, float, float, float], str]:
    """Read three georeferenced single-band rasters → 4326 RGB, stretched to uint8.

    Each path is a GDAL-openable source (incl. ``/vsizip/`` / ``/vsitar/``). Bands
    are read decimated, reprojected to EPSG:4326, then percentile-stretched. The
    three bands must share a grid (true for Landsat/S2 band stacks). ``0`` is the
    Landsat/S2 fill value → masked to NaN so it neither stretches nor reprojects in.
    """
    chans: list[npt.NDArray[np.floating]] = []
    bounds: tuple[float, float, float, float] | None = None
    crs = ""
    for p in paths:
        dec = _read_band_decimated(p, band=1, long_edge=long_edge)
        masked = np.where(dec.array == 0, np.nan, dec.array)
        arr_4326, bounds = _to_4326(
            _Decimated(
                array=masked,
                transform=dec.transform,
                crs=dec.crs,
                bounds_4326=dec.bounds_4326,
            )
        )
        chans.append(arr_4326)
        crs = str(dec.crs)
    assert bounds is not None
    rgb = np.dstack(chans)
    return _stretch_uint8(rgb), bounds, crs


def _read_gcp_band_4326(
    src_path: str, *, long_edge: int
) -> tuple[npt.NDArray[np.floating], tuple[float, float, float, float], str]:
    """Decimated read of a GCP-georeferenced band, warped to EPSG:4326.

    Sentinel-1 measurement TIFFs report ``crs=None`` + an identity transform but
    carry GCPs in EPSG:4326 (F3). We compute a GCP-based source transform, read the
    band decimated, then ``reproject`` to 4326 — never a full-res load (~146 MB).

    Returns ``(array_4326, bounds_4326, gcp_crs)``.

    Raises:
        ValueError: If the source carries no GCPs.
    """
    with rasterio.open(src_path) as src:
        gcps, gcp_crs = src.gcps
        if not gcps:
            raise ValueError("source has no GCPs; cannot georeference")

        out_h, out_w = _decimated_shape(width=src.width, height=src.height, long_edge=long_edge)
        arr = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average).astype(
            "float32"
        )
        # GCP transform for the *decimated* grid: derive the full-res GCP transform,
        # then scale it by the read ratio (GCP pixel coords are in full-res space).
        full_transform = rasterio.transform.from_gcps(gcps)
        src_transform = full_transform * full_transform.scale(
            src.width / out_w, src.height / out_h
        )

    dec = _Decimated(
        array=arr,
        transform=src_transform,
        crs=CRS.from_user_input(gcp_crs),
        bounds_4326=array_bounds(out_h, out_w, src_transform),
    )
    arr_4326, bounds = _to_4326(dec)
    return arr_4326, bounds, str(gcp_crs)


def _landsat_band(archive: Path, band_token: str) -> str:
    members = list_tar_members(archive)
    return vsitar_path(archive, find_member(members, suffix=f"_{band_token}.TIF"))


class _LandsatRenderer:
    def __init__(self, source: str) -> None:
        self.source = source

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        # True-colour B4/B3/B2 from the tar via /vsitar/ (per-scene UTM, F2).
        paths = (
            _landsat_band(row.path, "B4"),
            _landsat_band(row.path, "B3"),
            _landsat_band(row.path, "B2"),
        )
        rgb, bounds, crs = _read_rgb_paths_4326(paths, long_edge=long_edge)
        return QuicklookResult(
            kind="georef_raster",
            image=rgb,
            bounds_4326=bounds,
            src_crs=crs,
            label=f"{self.source} true-colour (B4/B3/B2)",
        )


class _Sentinel2Renderer:
    source = "sentinel2"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        archive = row.path
        members = list_zip_members(archive)

        def band(token: str) -> str:
            member = find_member(members, suffix=f"_{token}.jp2", contains="IMG_DATA")
            return vsizip_path(archive, member)

        # True-colour B04/B03/B02 jp2 via /vsizip/ (EPSG:32611, F3).
        paths = (band("B04"), band("B03"), band("B02"))
        rgb, bounds, crs = _read_rgb_paths_4326(paths, long_edge=long_edge)
        return QuicklookResult(
            kind="georef_raster",
            image=rgb,
            bounds_4326=bounds,
            src_crs=crs,
            label="S2 true-colour (B04/B03/B02)",
        )


class _Sentinel1Renderer:
    source = "sentinel1"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        members = list_zip_members(row.path)
        # VV measurement TIFF (fallback VH); GCP-warped (F3 corrected).
        try:
            member = find_member(members, suffix=".tiff", contains="-vv-")
            pol = "VV"
        except FileNotFoundError:
            member = find_member(members, suffix=".tiff", contains="-vh-")
            pol = "VH"
        path = vsizip_path(row.path, member)
        arr_4326, bounds, crs = _read_gcp_band_4326(path, long_edge=long_edge)
        # dB-stretch the backscatter amplitude for display.
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(np.where(arr_4326 > 0, arr_4326, np.nan))
        return QuicklookResult(
            kind="georef_raster",
            image=_stretch_uint8(db),
            bounds_4326=bounds,
            src_crs=crs,
            label=f"S1 GRD {pol} (dB, GCP-warped)",
        )


# --------------------------------------------------------------------------- #
# Phase 4 — ERA5 (date-stepped NetCDF) + S3 (non-georeferenced radiance)
# --------------------------------------------------------------------------- #


def era5_time_steps(path: Path) -> list[str]:
    """Return ISO date strings for an ERA5 NetCDF's ``valid_time`` axis (slider)."""
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return [str(np.datetime_as_string(t, unit="D")) for t in ds["valid_time"].values]


class _Era5Renderer:
    source = "era5"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        with xr.open_dataset(row.path, engine="h5netcdf") as ds:
            var = next(iter(ds.data_vars))
            n_times = ds.sizes["valid_time"]
            idx = max(0, min(date_idx, n_times - 1))
            slab = ds[var].isel(valid_time=idx)
            arr = np.asarray(slab.values, dtype="float32")
            lats = np.asarray(ds["latitude"].values, dtype="float64")
            lons = np.asarray(ds["longitude"].values, dtype="float64")
            when = str(np.datetime_as_string(ds["valid_time"].values[idx], unit="D"))

        # ERA5-Land is a regular EPSG:4326 grid; build bounds from coord edges.
        dlat = abs(float(lats[1] - lats[0])) if lats.size > 1 else 0.1
        dlon = abs(float(lons[1] - lons[0])) if lons.size > 1 else 0.1
        bounds = (
            float(lons.min()) - dlon / 2,
            float(lats.min()) - dlat / 2,
            float(lons.max()) + dlon / 2,
            float(lats.max()) + dlat / 2,
        )
        # Orient north-up (descending latitude is the ERA5 convention).
        if lats.size > 1 and lats[0] < lats[-1]:
            arr = arr[::-1, :]
        return QuicklookResult(
            kind="georef_raster",
            image=arr,
            bounds_4326=bounds,
            src_crs="EPSG:4326",
            label=f"ERA5 {var} @ {when} (idx {idx}/{n_times - 1})",
        )


class _Sentinel3Renderer:
    source = "sentinel3"

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        assert row.path is not None
        # S3 OLCI geolocation lives in a separate per-pixel geo_coordinates.nc with
        # no affine/CRS — render a NON-georeferenced radiance quicklook (CONTRACT
        # §"Non-georeferenced set"). Read one radiance band via h5py (xarray's
        # reference handling fails on these files).
        import h5py

        with zipfile.ZipFile(row.path) as zf:
            member = find_member(zf.namelist(), suffix="Oa08_radiance.nc")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "radiance.nc"
                out.write_bytes(zf.read(member))
                with h5py.File(out, "r") as f:
                    key = next(k for k in f if "radiance" in k.lower())
                    raw = np.asarray(f[key][()], dtype="float32")

        if raw.size == 0:
            return QuicklookResult(
                kind="plain_image",
                image=np.zeros((1, 1), dtype=np.uint8),
                bounds_4326=None,
                src_crs=None,
                label="S3 OLCI Oa08 radiance",
                note="empty radiance array (clip produced no pixels)",
            )
        # Decimate by striding to respect long_edge without a georeferenced read.
        step = max(1, max(raw.shape) // long_edge)
        decimated = raw[::step, ::step]
        return QuicklookResult(
            kind="plain_image",
            image=_stretch_uint8(decimated),
            bounds_4326=None,
            src_crs=None,
            label="S3 OLCI Oa08 radiance",
            note="non-georeferenced (geolocation is a separate per-pixel grid)",
        )


def result_to_geotiff(result: QuicklookResult, dst: Path) -> Path:
    """Write a ``georef_raster`` quicklook to a small EPSG:4326 GeoTIFF for display.

    The decimated image is written with an affine derived from ``bounds_4326`` so a
    map layer (leafmap ``add_raster``) can place it directly; we ship the already
    decimated array, never the full-res source.

    Args:
        result: A ``georef_raster`` result (its ``bounds_4326`` must be set).
        dst: Output path for the GeoTIFF.

    Returns:
        ``dst``.

    Raises:
        ValueError: If ``result`` is not a ``georef_raster`` or lacks bounds.
    """
    if result.kind != "georef_raster" or result.bounds_4326 is None:
        raise ValueError("result_to_geotiff requires a georef_raster with bounds")

    image = result.image
    is_rgb = image.ndim == 3
    height, width = image.shape[:2]
    transform = transform_from_bounds(*result.bounds_4326, width, height)

    # uint8 for RGB/stretched single-band; float32 for continuous (DEM).
    # A finite nodata sentinel (not NaN) is required: the tile server's
    # ``/api/metadata`` route serializes band stats as JSON, and NaN/inf are not
    # JSON-compliant — a NaN nodata makes that endpoint 500.
    if is_rgb or image.dtype == np.uint8:
        data = image.astype(np.uint8)
        bands = [data[..., i] for i in range(3)] if is_rgb else [data]
        # Reprojecting a sheared (e.g. sinusoidal) AOI leaves large fill regions
        # that ``_stretch_uint8`` zeroed. Add an alpha band so those areas render
        # transparent (basemap shows through) instead of as a black rectangle.
        # Heuristic: a pixel is masked iff every colour band is 0 (genuinely pure
        # black valid pixels are vanishingly rare in stretched reflectance).
        opaque = np.zeros((height, width), dtype=bool)
        for b in bands:
            opaque |= b > 0
        alpha = np.where(opaque, 255, 0).astype(np.uint8)
        out_bands = [*bands, alpha]
        dtype: str = "uint8"
        nodata: float | None = None
    else:
        filled = np.where(np.isfinite(image), image, _FLOAT_NODATA).astype("float32")
        out_bands = [filled]
        dtype = "float32"
        nodata = _FLOAT_NODATA

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(out_bands),
        "dtype": dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "deflate",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(dst, "w", **profile) as out:
        for i, band in enumerate(out_bands, start=1):
            out.write(band, i)
        if is_rgb or (image.dtype == np.uint8 and image.ndim == 2):
            # Tag the trailing band as alpha so the tile server honours it.
            out.colorinterp = [*out.colorinterp[:-1], ColorInterp.alpha]
    return dst


def register_renderers() -> None:
    """Register every implemented renderer in the dispatch registry."""
    for renderer in (
        # Phase 2 — plain GeoTIFFs.
        _DemRenderer(),
        _WorldCoverRenderer(),
        _ModisRenderer(),
        _ViirsRenderer(),
        # Phase 3 — archives via /vsi*/.
        _LandsatRenderer("landsat8"),
        _LandsatRenderer("landsat9"),
        _Sentinel2Renderer(),
        _Sentinel1Renderer(),
        # Phase 4 — ERA5 (date-stepped) + S3 (non-georeferenced).
        _Era5Renderer(),
        _Sentinel3Renderer(),
    ):
        register(renderer)


register_renderers()
