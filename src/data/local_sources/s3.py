"""Sentinel-3 OLCI adapter — ``[Oa17_radiance, Oa21_radiance]`` (TASK-011).

Replaces the S3 placeholders with top-of-atmosphere radiance on the cell grid. OLCI
EFR products are a **curvilinear swath**: each radiance ``.nc`` is paired with a
per-pixel ``geo_coordinates.nc`` (latitude/longitude). We decode the radiance and geo
CF scaling, pick the overpass covering the cell, and warp the swath onto the cell's
regular UTM grid with :func:`scipy.interpolate.griddata`.

**The CF-scaling landmine.** ``geo_coordinates`` lat/lon are ``int32`` with
``scale_factor = 1e-6``; comparing the raw integers to degree bounds (the TASK-002 bug)
yields an empty mask and ``(0,0)`` radiance. We multiply by ``scale_factor`` before any
geographic use. Radiance is ``uint16`` with a per-band ``scale_factor`` (Oa17 0.00493004,
Oa21 0.00324118) — identical to the GEE catalog's listed scale — giving
W m⁻² sr⁻¹ µm⁻¹. Identity normalization downstream, so the scale flows through unchanged.

**Parity caveat (documented, accepted).** GEE terrain-orthorectifies OLCI in SNAP; this
swath-warp does not, leaving OLCI's intrinsic ~300–450 m geolocation gap (corr ~0.66–0.73
vs the reference patch; a rigid shift does not close it). The scale is exact and the
geolocation input is the best available (the full per-pixel grid, not the ×64-subsampled
tie points). The residual is owned by the out-of-scope S3-ortho/normalization follow-up
(PARITY_SPIKE_NOTES §10). Reading SEN3 NetCDF goes through ``h5py`` directly —
``h5netcdf``/``xarray`` cannot resolve these files' HDF5 dimension-scale references.
"""

from __future__ import annotations

import datetime
import io
import zipfile
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import structlog
from pyproj import Transformer
from rasterio.transform import xy
from scipy.interpolate import griddata

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell, LocalSourceAdapter, create_placeholder

logger = structlog.get_logger(__name__)

_GEO_CRS: str = "EPSG:4326"

#: Per-band radiance ``.nc`` filename and the variable inside it.
_RADIANCE_FILES: dict[str, str] = {
    "Oa17_radiance": "Oa17_radiance.nc",
    "Oa21_radiance": "Oa21_radiance.nc",
}

#: The valid-radiance floor (below this → treated as fill / out of domain).
_VALID_MIN: float = -1.0


def _decode(dataset: h5py.Dataset) -> tuple[npt.NDArray[np.float64], float | None]:
    """Return ``(values, fill)`` with the CF ``scale_factor``/``add_offset`` applied.

    The native ``_FillValue`` is mapped to NaN *before* scaling so it never leaks into a
    scaled number; the (unscaled) fill is returned for the caller's masking.
    """
    raw = dataset[:]
    scale = dataset.attrs.get("scale_factor")
    offset = dataset.attrs.get("add_offset")
    fill = dataset.attrs.get("_FillValue")
    scale_f = float(scale[0]) if scale is not None else 1.0
    offset_f = float(offset[0]) if offset is not None else 0.0
    fill_v = float(fill[0]) if fill is not None else None

    values = raw.astype(np.float64)
    if fill_v is not None:
        values[raw == fill_v] = np.nan
    values = values * scale_f + offset_f
    return values, fill_v


class S3Adapter(LocalSourceAdapter):
    """Sentinel-3 OLCI ``[Oa17_radiance, Oa21_radiance]`` adapter (``med`` tier, swath-warp).

    Args:
        archive_root: The clipped S3 archive root (holds ``S3?_OL_1_EFR*.zip`` products).
    """

    bands_out = ["Oa17_radiance", "Oa21_radiance"]
    spatial_kind = "med"
    native_fill = None

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _products_for_day(self, day: datetime.date) -> list[Path]:
        """All S3 product zips whose acquisition start date is ``day``."""
        return sorted(self.archive_root.glob(f"S3?_OL_1_EFR____{day:%Y%m%d}T*.zip"))

    @staticmethod
    def _read_member(zf: zipfile.ZipFile, suffix: str, var: str) -> npt.NDArray[np.float64]:
        """Read+decode one variable from the ``.nc`` member ending in ``suffix``.

        SEN3 NetCDF can't be opened from a stream by h5netcdf (dimension-scale refs), so
        the member is materialized to an in-memory ``h5py.File``.
        """
        name = next(n for n in zf.namelist() if n.endswith(suffix))
        with h5py.File(io.BytesIO(zf.read(name)), "r") as f:
            values, _ = _decode(f[var])
        return values

    def _warp_overpass(
        self, zip_path: Path, cell: GridCell, target_lonlat: tuple[npt.NDArray, npt.NDArray]
    ) -> npt.NDArray[np.float64] | None:
        """Warp one overpass's two radiance bands onto the cell grid.

        Returns a ``(2, H, W)`` array (NaN where the swath does not cover a pixel), or
        ``None`` if the overpass does not cover the cell footprint at all.
        """
        tlon, tlat = target_lonlat
        with zipfile.ZipFile(zip_path) as zf:
            lat = self._read_member(zf, "geo_coordinates.nc", "latitude")
            lon = self._read_member(zf, "geo_coordinates.nc", "longitude")
            radiances = {
                band: self._read_member(zf, fname, band) for band, fname in _RADIANCE_FILES.items()
            }

        # Only points with valid geolocation AND valid radiance feed the interpolation.
        geo_ok = np.isfinite(lat) & np.isfinite(lon)
        if not geo_ok.any():
            return None
        # Reject overpasses that don't bracket the cell centre (cheap coverage gate).
        cx = float(np.mean(tlon))
        cy = float(np.mean(tlat))
        if not (lat[geo_ok].min() <= cy <= lat[geo_ok].max()):
            return None
        if not (lon[geo_ok].min() <= cx <= lon[geo_ok].max()):
            return None

        targets = np.column_stack([tlon, tlat])
        bands: list[npt.NDArray[np.float64]] = []
        for band in self.bands_out:
            rad = radiances[band]
            good = geo_ok & np.isfinite(rad) & (rad >= _VALID_MIN)
            if good.sum() < 4:
                bands.append(np.full(cell.shape, np.nan))
                continue
            warped = griddata(
                np.column_stack([lon[good], lat[good]]),
                rad[good],
                targets,
                method="nearest",  # ~300 m swath onto 10 m cell — nearest, no smoothing
            ).reshape(cell.shape)
            bands.append(warped)
        return np.stack(bands, axis=0)

    def _target_lonlat(self, cell: GridCell) -> tuple[npt.NDArray, npt.NDArray]:
        """Cell pixel-centre coordinates as ``(lon, lat)`` in degrees."""
        height, width = cell.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = xy(cell.transform, rows.ravel(), cols.ravel())
        transformer = Transformer.from_crs(cell.crs, _GEO_CRS, always_xy=True)
        lon, lat = transformer.transform(np.asarray(xs), np.asarray(ys))
        return np.asarray(lon), np.asarray(lat)

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the two OLCI radiance bands on the cell grid (``-9999`` where missing).

        If several overpasses cover the cell on ``day``, the one with the most valid
        Oa17 pixels over the cell footprint wins.
        """
        if day is None:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)
        products = self._products_for_day(day)
        if not products:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        target_lonlat = self._target_lonlat(cell)
        best: npt.NDArray[np.float64] | None = None
        best_valid = -1
        for product in products:
            warped = self._warp_overpass(product, cell, target_lonlat)
            if warped is None:
                continue
            n_valid = int(np.isfinite(warped[0]).sum())
            if n_valid > best_valid:
                best_valid, best = n_valid, warped

        if best is None or best_valid < 1:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        best[~np.isfinite(best)] = float(NO_DATA_VALUE)
        logger.info(
            "s3_fetch",
            cell_id=cell.cell_id,
            day=day.isoformat(),
            overpasses=len(products),
            valid_px=best_valid,
        )
        return best.astype(np.float32)
