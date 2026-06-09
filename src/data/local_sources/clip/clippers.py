"""Per-modality clip routines (CLIPPING_PLAN §2.1–§2.7).

Every routine:

1. Reads the product footprint (metadata only) and runs the §2.0 intersect gate.
2. On a ``SKIP_*`` action, writes **no output file** and returns a manifest row.
3. On ``CLIP``, crops to the AOI preserving native CRS / pixel values / format,
   counts valid (non-nodata) pixels, and — if ``require_valid_pixels`` and the
   clip is empty — deletes the output and downgrades to
   ``SKIP_DEGENERATE_OVERLAP``.

Routines never reproject to the cell grid, fabricate ``-9999`` placeholders, or
pad to the full AOI; that is the adapter layer's job. This stage only crops.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.mask
import structlog
import xarray as xr
from pyproj import Transformer
from rasterio.control import GroundControlPoint
from rasterio.windows import Window
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as shapely_transform

from . import footprints
from .gate import ClipAction, GateResult, evaluate_gate
from .gdal_io import list_subdatasets, translate_subdataset
from .manifest import ManifestRow, bbox_str
from .settings import AOI_CRS, ClipSettings

logger = structlog.get_logger()


def _skip_row(
    *, product_id: str, source: str, footprint: Optional[Polygon], gate: GateResult
) -> ManifestRow:
    """Build a manifest row for a gated-out (skipped) product."""
    return ManifestRow(
        product_id=product_id,
        source=source,
        footprint_bbox=bbox_str(footprint.bounds if footprint else None),
        intersects=gate.intersects,
        aoi_overlap_km2=round(gate.aoi_overlap_km2, 6),
        valid_pixel_count=0,
        action=gate.action,
    )


def _reproject_aoi(aoi_4326: Polygon, dst_crs: object) -> Polygon:
    """Reproject the AOI polygon from EPSG:4326 to ``dst_crs``."""
    if dst_crs is None or rasterio.crs.CRS.from_user_input(dst_crs).to_epsg() == 4326:
        return aoi_4326
    transformer = Transformer.from_crs(AOI_CRS, dst_crs, always_xy=True)
    return shapely_transform(
        lambda xs, ys, transformer=transformer: transformer.transform(xs, ys),
        aoi_4326,
    )


def _count_valid_pixels(array: np.ndarray, nodata: Optional[float]) -> int:
    """Count non-nodata pixels in a raster array."""
    if nodata is None:
        return int(array.size)
    return int(np.count_nonzero(array != nodata))


def _clip_geotiff_to(src_path: Path, dst_path: Path, aoi_4326: Polygon) -> tuple[int, object]:
    """Crop a georeferenced raster to the AOI, preserving CRS, profile, and pixel values.

    The clip is non-destructive: ``rasterio.mask.mask`` only crops (nearest, no
    resampling), so the written pixels must equal the raw pixels inside the AOI.

    **Lossless-JP2 guard.** Sentinel-2 bands are JPEG-2000. GDAL's ``JP2OpenJPEG``
    writer defaults to *lossy* even when the source profile came from a lossless
    JP2, which silently corrupts both reflectance (±~2 DN) and the categorical
    ``MSK_CLASSI`` cloud mask (class flips). When the output is a ``.jp2`` we force
    reversible (lossless) wavelet + full quality so the crop stays bit-exact. The
    same setting the S2 test fixtures use (``REVERSIBLE``/``QUALITY=100``). Other
    sources (GeoTIFF: DEFLATE/LZW/none) are already lossless and unaffected.

    Returns:
        ``(valid_pixel_count, crs)`` of the written output.
    """
    with rasterio.open(src_path) as src:
        aoi_in_crs = _reproject_aoi(aoi_4326, src.crs)
        out_image, out_transform = rasterio.mask.mask(src, [mapping(aoi_in_crs)], crop=True)
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )
        crs = src.crs
        nodata = src.nodata

    if str(dst_path).lower().endswith(".jp2"):
        # Force lossless JPEG-2000; GDAL's JP2OpenJPEG default is lossy.
        profile.update(driver="JP2OpenJPEG", REVERSIBLE="YES", QUALITY="100")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out_image)
    return _count_valid_pixels(out_image, nodata), crs


# --------------------------------------------------------------------------- #
# Standard GeoTIFFs — DEM & WorldCover (§2.1)
# --------------------------------------------------------------------------- #
def clip_geotiff(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a single DEM / WorldCover GeoTIFF tile through the gate."""
    footprint = footprints.raster_footprint_4326(src_path)
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    product_id = src_path.stem
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    valid, crs = _clip_geotiff_to(src_path, dst_path, aoi_4326)
    return _finalize_clip(
        product_id=product_id,
        source=source,
        footprint=footprint,
        gate=gate,
        valid=valid,
        outputs=[dst_path],
        rel_output=dst_path.name,
        settings=settings,
    )


# --------------------------------------------------------------------------- #
# ERA5-Land NetCDF (§2.2)
# --------------------------------------------------------------------------- #
def clip_era5(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip an ERA5-Land NetCDF by slicing its lat/lon coordinate dimensions."""
    lon_min, lat_min, lon_max, lat_max = aoi_4326.bounds
    product_id = src_path.stem

    with xr.open_dataset(src_path, engine="h5netcdf") as ds:
        lat = ds.coords["latitude"].values
        if lat[0] > lat[-1]:  # descending latitude (ERA5 convention)
            ds_clip = ds.sel(latitude=slice(lat_max, lat_min))
        else:
            ds_clip = ds.sel(latitude=slice(lat_min, lat_max))
        ds_clip = ds_clip.sel(longitude=slice(lon_min, lon_max))
        n_lat = ds_clip.sizes.get("latitude", 0)
        n_lon = ds_clip.sizes.get("longitude", 0)

        footprint = (
            footprints.netcdf_footprint(
                float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())
            )
            if (lon := ds.coords["longitude"].values) is not None
            else None
        )

        if n_lat == 0 or n_lon == 0:
            gate = GateResult(ClipAction.SKIP_NO_OVERLAP, False, 0.0)
            logger.info("gated", product=product_id, action=gate.action.value)
            return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ds_clip.to_netcdf(dst_path, engine="h5netcdf")

    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    return ManifestRow(
        product_id=product_id,
        source=source,
        footprint_bbox=bbox_str(footprint.bounds if footprint else None),
        intersects=True,
        aoi_overlap_km2=round(gate.aoi_overlap_km2, 6),
        valid_pixel_count=int(n_lat * n_lon),
        action=ClipAction.CLIP,
        output_path=dst_path.name,
    )


# --------------------------------------------------------------------------- #
# Landsat tarballs — landsat8 / landsat9 (§2.3)
# --------------------------------------------------------------------------- #
def clip_landsat(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a Landsat ``.tar``: gate on MTL footprint, crop each band in-zone."""
    product_id = src_path.stem
    footprint = footprints.landsat_footprint(src_path)
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    with (
        tarfile.open(src_path, "r") as src_tar,
        tarfile.open(dst_path, "w") as dst_tar,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp_dir = Path(tmp)
        for member in src_tar.getmembers():
            if not member.isfile():
                continue
            src_tar.extract(member, path=tmp_dir)
            extracted = tmp_dir / member.name
            if member.name.lower().endswith((".tif", ".tiff")):
                clipped = tmp_dir / f"clipped_{Path(member.name).name}"
                valid, _ = _clip_geotiff_to(extracted, clipped, aoi_4326)
                total_valid += valid
                dst_tar.add(clipped, arcname=member.name)
            else:  # MTL / ANG / XML metadata — copy unchanged
                dst_tar.add(extracted, arcname=member.name)

    return _finalize_clip(
        product_id=product_id,
        source=source,
        footprint=footprint,
        gate=gate,
        valid=total_valid,
        outputs=[dst_path],
        rel_output=dst_path.name,
        settings=settings,
    )


# --------------------------------------------------------------------------- #
# Sentinel-2 — JP2 bands in a SAFE zip (§2.4)
# --------------------------------------------------------------------------- #
def clip_sentinel2(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a Sentinel-2 ``.zip``: gate on SAFE footprint, crop each JP2 band."""
    product_id = src_path.stem
    footprint = footprints.sentinel_safe_footprint(src_path, "manifest.safe")
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    with (
        zipfile.ZipFile(src_path, "r") as src_zip,
        zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as dst_zip,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp_dir = Path(tmp)
        for name in src_zip.namelist():
            if name.endswith("/"):
                continue
            src_zip.extract(name, path=tmp_dir)
            extracted = tmp_dir / name
            if name.lower().endswith(".jp2"):
                clipped = tmp_dir / f"clipped_{Path(name).name}"
                valid, _ = _clip_geotiff_to(extracted, clipped, aoi_4326)
                total_valid += valid
                dst_zip.write(clipped, arcname=name)
            else:  # XML / manifest — copy unchanged
                dst_zip.write(extracted, arcname=name)

    return _finalize_clip(
        product_id=product_id,
        source=source,
        footprint=footprint,
        gate=gate,
        valid=total_valid,
        outputs=[dst_path],
        rel_output=dst_path.name,
        settings=settings,
    )


# --------------------------------------------------------------------------- #
# Sentinel-1 — range-geometry GCP TIFFs in a SAFE zip (§2.5)
# --------------------------------------------------------------------------- #
def clip_sentinel1(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a Sentinel-1 ``.zip``: GCP-window slice each range-geometry TIFF."""
    product_id = src_path.stem
    footprint = footprints.sentinel_safe_footprint(src_path, "manifest.safe")
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    lon_min, lat_min, lon_max, lat_max = aoi_4326.bounds
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    with (
        zipfile.ZipFile(src_path, "r") as src_zip,
        zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as dst_zip,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp_dir = Path(tmp)
        for name in src_zip.namelist():
            if name.endswith("/"):
                continue
            src_zip.extract(name, path=tmp_dir)
            extracted = tmp_dir / name
            is_measurement = name.lower().endswith((".tiff", ".tif")) and "measurement/" in name
            if is_measurement:
                clipped = tmp_dir / f"clipped_{Path(name).name}"
                valid = _clip_s1_measurement(
                    extracted,
                    clipped,
                    lon_min,
                    lat_min,
                    lon_max,
                    lat_max,
                    settings.gcp_buffer_pixels,
                )
                total_valid += valid
                dst_zip.write(clipped, arcname=name)
            else:
                dst_zip.write(extracted, arcname=name)

    return _finalize_clip(
        product_id=product_id,
        source=source,
        footprint=footprint,
        gate=gate,
        valid=total_valid,
        outputs=[dst_path],
        rel_output=dst_path.name,
        settings=settings,
    )


def _clip_s1_measurement(
    src_tiff: Path,
    dst_tiff: Path,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    buffer_px: int,
) -> int:
    """Slice one S1 measurement TIFF to the AOI via its GCP grid.

    Defensive: if the TIFF resolves a real CRS + affine transform it is clipped
    as a standard GeoTIFF; otherwise (the verified range-geometry case) the AOI
    is intersected against the GCP lon/lat grid and the pixel array is sliced to
    the bounding GCP window.

    Returns:
        Count of non-zero pixels in the written window.
    """
    with rasterio.open(src_tiff) as src:
        if src.crs is not None and src.transform is not None and not src.gcps[0]:
            data = src.read()
            with rasterio.open(dst_tiff, "w", **src.profile) as dst:
                dst.write(data)
            return int(np.count_nonzero(data))

        gcps, gcp_crs = src.gcps
        in_aoi = [g for g in gcps if lon_min <= g.x <= lon_max and lat_min <= g.y <= lat_max]
        if not in_aoi:  # gate passed on footprint but no GCP lands in AOI bbox
            col_min, row_min, col_max, row_max = 0, 0, 0, 0
        else:
            col_min = max(0, int(min(g.col for g in in_aoi)) - buffer_px)
            row_min = max(0, int(min(g.row for g in in_aoi)) - buffer_px)
            col_max = min(src.width, int(max(g.col for g in in_aoi)) + buffer_px)
            row_max = min(src.height, int(max(g.row for g in in_aoi)) + buffer_px)

        window = Window(col_min, row_min, max(0, col_max - col_min), max(0, row_max - row_min))
        data = src.read(window=window)

        shifted = [
            GroundControlPoint(
                row=g.row - row_min, col=g.col - col_min, x=g.x, y=g.y, z=g.z, id=g.id
            )
            for g in gcps
            if col_min <= g.col <= col_max and row_min <= g.row <= row_max
        ]
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2])

    with rasterio.open(dst_tiff, "w", **profile) as dst:
        dst.write(data)
        if shifted:
            dst.gcps = (shifted, gcp_crs)
    return int(np.count_nonzero(data))


# --------------------------------------------------------------------------- #
# Sentinel-3 — tie-point-grid NetCDF in a SAFE zip (§2.6)
# --------------------------------------------------------------------------- #
def clip_sentinel3(
    *,
    src_path: Path,
    dst_path: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a Sentinel-3 ``.zip``: slice each band NetCDF on its tie-point grid."""
    import h5py

    product_id = src_path.stem
    footprint = footprints.sentinel_safe_footprint(src_path, "xfdumanifest.xml")
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    lon_min, lat_min, lon_max, lat_max = aoi_4326.bounds
    buf = settings.swath_buffer_pixels
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    with (
        zipfile.ZipFile(src_path, "r") as src_zip,
        zipfile.ZipFile(dst_path, "w", zipfile.ZIP_DEFLATED) as dst_zip,
        tempfile.TemporaryDirectory() as tmp,
    ):
        tmp_dir = Path(tmp)
        geo_name = next(n for n in src_zip.namelist() if "geo_coordinates.nc" in n)
        src_zip.extract(geo_name, path=tmp_dir)
        with h5py.File(tmp_dir / geo_name, "r") as geo:
            # S3 geo_coordinates store lat/lon as scaled int32 (scale_factor ≈ 1e-6,
            # optional add_offset). Compare in DEGREES — applying the CF scaling —
            # never the raw integers, or the AOI mask is empty and the radiance
            # clips to (0, 0) while unrelated full-copied datasets still inflate the
            # valid-pixel count (silent all-empty clip).
            lats = _cf_scaled(geo["latitude"])
            lons = _cf_scaled(geo["longitude"])
        n_rows, n_cols = int(lats.shape[0]), int(lats.shape[1])
        grid_shape: tuple[int, int] = (n_rows, n_cols)
        mask = (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)
        rows, cols = mask.nonzero()
        if len(rows) == 0:
            r0, r1, c0, c1 = 0, 0, 0, 0
        else:
            r0 = max(0, int(rows.min()) - buf)
            r1 = min(n_rows, int(rows.max()) + buf)
            c0 = max(0, int(cols.min()) - buf)
            c1 = min(n_cols, int(cols.max()) + buf)

        for name in src_zip.namelist():
            if name.endswith("/"):
                continue
            src_zip.extract(name, path=tmp_dir)
            extracted = tmp_dir / name
            if name.lower().endswith(".nc"):
                clipped = tmp_dir / f"clipped_{Path(name).name}"
                total_valid += _slice_s3_netcdf(extracted, clipped, grid_shape, r0, r1, c0, c1)
                dst_zip.write(clipped, arcname=name)
            else:
                dst_zip.write(extracted, arcname=name)

    return _finalize_clip(
        product_id=product_id,
        source=source,
        footprint=footprint,
        gate=gate,
        valid=total_valid,
        outputs=[dst_path],
        rel_output=dst_path.name,
        settings=settings,
    )


def _cf_scaled(dataset: object) -> np.ndarray:
    """Decode a CF-scaled HDF5 dataset to physical units (``scale_factor`` /
    ``add_offset``).

    S3 ``geo_coordinates`` lat/lon are int32 with ``scale_factor = 1e-6``; the raw
    integers are meaningless against degree bounds. Returns a ``float64`` array in
    physical units; a dataset without scaling attrs is returned unchanged (as float).
    """
    values = dataset[:].astype("float64")  # type: ignore[index]
    attrs = dataset.attrs  # type: ignore[attr-defined]
    scale = attrs.get("scale_factor")
    offset = attrs.get("add_offset")
    if scale is not None:
        values = values * float(np.asarray(scale).reshape(-1)[0])
    if offset is not None:
        values = values + float(np.asarray(offset).reshape(-1)[0])
    return values


def _slice_s3_netcdf(
    src_nc: Path,
    dst_nc: Path,
    grid_shape: tuple[int, int],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> int:
    """Slice every grid-shaped 2D dataset in an S3 band NetCDF to the AOI window."""
    import h5py

    n_rows, n_cols = grid_shape
    valid = 0
    with h5py.File(src_nc, "r") as src, h5py.File(dst_nc, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def _visit(obj_name: str, obj: object) -> None:
            nonlocal valid
            if not isinstance(obj, h5py.Dataset):
                return
            shp = obj.shape
            if len(shp) == 2 and shp == (n_rows, n_cols):
                data = obj[r0:r1, c0:c1]
            elif len(shp) == 1 and shp[0] == n_rows:
                data = obj[r0:r1]
            elif len(shp) == 1 and shp[0] == n_cols:
                data = obj[c0:c1]
            else:
                data = obj[()]
            ds = dst.create_dataset(obj_name, data=data, dtype=obj.dtype)
            for k, v in obj.attrs.items():
                ds.attrs[k] = v
            if getattr(data, "ndim", 0) == 2:
                valid += int(data.size)

        src.visititems(_visit)
    return valid


# --------------------------------------------------------------------------- #
# MODIS / VIIRS — per-grid sinusoidal subdatasets (§2.7)
# --------------------------------------------------------------------------- #
def clip_sinusoidal(
    *,
    src_path: Path,
    dst_dir: Path,
    source: str,
    aoi_4326: Polygon,
    settings: ClipSettings,
) -> ManifestRow:
    """Clip a MODIS/VIIRS tile per-grid to AOI; one GeoTIFF per subdataset.

    Each science/QA subdataset is extracted to GeoTIFF (preserving its native
    sinusoidal CRS + geotransform), then cropped using indices computed from
    *that subdataset's own* geotransform — never a hardcoded 1200 clamp, so the
    500 m grid (2400²) is indexed at ~2× the 1 km grid (1200²).
    """
    product_id = src_path.stem
    footprint = footprints.sinusoidal_tile_footprint(src_path)
    gate = evaluate_gate(
        footprint_4326=footprint if footprint else Polygon(),
        aoi_4326=aoi_4326,
        min_aoi_overlap_area_km2=settings.min_aoi_overlap_area_km2,
    )
    if gate.action is not ClipAction.CLIP:
        logger.info("gated", product=product_id, action=gate.action.value)
        return _skip_row(product_id=product_id, source=source, footprint=footprint, gate=gate)

    out_subdir = dst_dir / product_id
    out_subdir.mkdir(parents=True, exist_ok=True)
    total_valid = 0
    written: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for sub in list_subdatasets(src_path):
            tmp_tif = tmp_dir / f"{sub.grid}__{sub.band}.tif"
            translate_subdataset(sub.name, tmp_tif)
            out_tif = out_subdir / f"{sub.grid}__{sub.band}.tif"
            valid = _clip_sinusoidal_subdataset(tmp_tif, out_tif, aoi_4326)
            if valid == 0 and out_tif.exists():
                out_tif.unlink()  # do not keep empty per-band slivers
                continue
            total_valid += valid
            written.append(out_tif)

    if settings.require_valid_pixels and total_valid == 0:
        shutil.rmtree(out_subdir, ignore_errors=True)
        downgraded = GateResult(ClipAction.SKIP_DEGENERATE_OVERLAP, True, gate.aoi_overlap_km2)
        return _skip_row(
            product_id=product_id, source=source, footprint=footprint, gate=downgraded
        )

    return ManifestRow(
        product_id=product_id,
        source=source,
        footprint_bbox=bbox_str(footprint.bounds if footprint else None),
        intersects=True,
        aoi_overlap_km2=round(gate.aoi_overlap_km2, 6),
        valid_pixel_count=total_valid,
        action=ClipAction.CLIP,
        output_path=str(out_subdir.name),
    )


def _clip_sinusoidal_subdataset(src_tif: Path, dst_tif: Path, aoi_4326: Polygon) -> int:
    """Crop one sinusoidal-grid GeoTIFF to the AOI by the reprojected geometry.

    Crops with ``rasterio.mask.mask(crop=True)`` against the AOI reprojected into
    the subdataset's own sinusoidal CRS — the same geometry crop the GeoTIFF path
    uses (:func:`_clip_geotiff_to`). A bounding-box window of the reprojected AOI
    *corners* is wrong here: the Sinusoidal projection (``x = R·λ·cos φ``) shears
    the AOI rectangle, so its axis-aligned pixel window is several times wider than
    the AOI in X. Masking by the actual geometry crops to its true footprint.

    Returns:
        Count of non-nodata pixels written (0 if the AOI window is empty).
    """
    with rasterio.open(src_tif) as src:
        aoi_in_crs = _reproject_aoi(aoi_4326, src.crs)
        try:
            out_image, out_transform = rasterio.mask.mask(src, [mapping(aoi_in_crs)], crop=True)
        except ValueError:
            # rasterio raises when the AOI does not overlap the raster at all.
            return 0
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )
        nodata = src.nodata

    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_tif, "w", **profile) as dst:
        dst.write(out_image)
    return _count_valid_pixels(out_image, nodata)


# --------------------------------------------------------------------------- #
# Shared finalisation: post-clip valid-pixel gate (§2.0 step 2b)
# --------------------------------------------------------------------------- #
def _finalize_clip(
    *,
    product_id: str,
    source: str,
    footprint: Optional[Polygon],
    gate: GateResult,
    valid: int,
    outputs: list[Path],
    rel_output: str,
    settings: ClipSettings,
) -> ManifestRow:
    """Apply the post-clip valid-pixel test, deleting empty outputs.

    A product whose footprint passed the area gate but whose clip yields zero
    valid pixels is downgraded to ``SKIP_DEGENERATE_OVERLAP`` and its output
    files are removed — the gate's guarantee that the archive holds no
    zero-signal artifacts.
    """
    if settings.require_valid_pixels and valid == 0:
        for path in outputs:
            if path.exists():
                path.unlink()
        downgraded = GateResult(
            ClipAction.SKIP_DEGENERATE_OVERLAP, gate.intersects, gate.aoi_overlap_km2
        )
        logger.info("post-clip empty", product=product_id, action=downgraded.action.value)
        return _skip_row(
            product_id=product_id, source=source, footprint=footprint, gate=downgraded
        )

    return ManifestRow(
        product_id=product_id,
        source=source,
        footprint_bbox=bbox_str(footprint.bounds if footprint else None),
        intersects=True,
        aoi_overlap_km2=round(gate.aoi_overlap_km2, 6),
        valid_pixel_count=valid,
        action=ClipAction.CLIP,
        output_path=rel_output,
    )
