"""ERA5-Land adapter — the five meteorological time bands (TASK-008).

Replaces the ERA5 placeholders with raw values on the cell grid, reading the
**already-daily** clipped archive (one slice per day — there is no hourly data on
disk to re-aggregate). Reproduces GEE ``ECMWF/ERA5_LAND/DAILY_AGGR``.

**Raw units.** Temps are emitted in **Kelvin**, winds in m/s, precip in metres.
The known temperature-shift (Kelvin→Celsius) lives in the downstream ``Normalizer``,
**not** here — the adapter must not replicate it (PLAN §4).

**Precip day-shift (the silent off-by-one).** ``total_precipitation_sum`` is an
accumulation (``GRIB_stepType=accum``) that ERA5-Land stamps at ``00:00`` of the day
*after* the one it sums. So precip for inference day ``d`` is read from the ``d+1``
``00:00`` slice — which can live in the *next month's* precip file (handled by the
across-file ``valid_time`` lookup). Instantaneous vars (``temperature_2m``,
``skin_temperature``, ``u/v_component_of_wind_10m``) carry **no** shift: read the
slice labelled ``d``.

**Resample = nearest.** The ~11 km 0.1° grid is far coarser than the 1 km cell, so
GEE upsamples it as a constant block per ERA5 cell. Nearest reproduces GEE to ~1e-4
(bilinear smears across cell boundaries by ~0.26 K — see PARITY_SPIKE_NOTES §7). We
route through :func:`~src.data.local_sources.base.reproject_to_cell` with
``categorical=True`` (its nearest path).

A missing ``(ERA5, day)`` returns the all-``-9999`` placeholder — the normal case,
not an error.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog
import xarray as xr
from affine import Affine

from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    create_placeholder,
    reproject_to_cell,
)

logger = structlog.get_logger(__name__)

#: ERA5 source CRS (the clipped NetCDF tiles are geographic).
_SOURCE_CRS: str = "EPSG:4326"

#: Canonical band order (matches ``src/data/earthengine/era5.py``).
_BANDS: list[str] = [
    "skin_temperature",
    "temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

#: Per-instantaneous-band: (filename within the monthly folder, NetCDF var name).
_INSTANT_FILES: dict[str, tuple[str, str]] = {
    "skin_temperature": ("skin_temperature_0_daily-mean.nc", "skt"),
    "temperature_2m": ("2m_temperature_0_daily-mean.nc", "t2m"),
    "u_component_of_wind_10m": ("10m_u_component_of_wind_0_daily-mean.nc", "u10"),
    "v_component_of_wind_10m": ("10m_v_component_of_wind_0_daily-mean.nc", "v10"),
}

#: The precip NetCDF var name (accumulation, day-shifted).
_PRECIP_VAR: str = "tp"


class Era5Adapter(LocalSourceAdapter):
    """ERA5-Land adapter emitting the five raw meteorological bands (``time`` tier).

    Static units, daily archive, precip ``i+1`` day-shift, nearest resample (see the
    module docstring for the full contract and rationale).

    Args:
        archive_root: The clipped ERA5 archive root (holds ``YYYYMM_ERA5LAND/`` folders
            and ``YYYYMM_ERA5LAND_totalprecip.nc`` files).
    """

    bands_out = _BANDS
    spatial_kind = "time"
    native_fill = None

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _monthly_folder(self, day: datetime.date) -> Path:
        return self.archive_root / f"{day:%Y%m}_ERA5LAND"

    def _precip_file(self, day: datetime.date) -> Path:
        return self.archive_root / f"{day:%Y%m}_ERA5LAND_totalprecip.nc"

    @staticmethod
    def _src_transform(lats: npt.NDArray, lons: npt.NDArray) -> Affine:
        """North-up affine for a regular lat/lon grid (lats descending)."""
        y_res = abs(float(lats[1] - lats[0]))
        x_res = abs(float(lons[1] - lons[0]))
        left = float(lons.min()) - x_res / 2.0
        top = float(lats.max()) + y_res / 2.0
        return Affine(x_res, 0.0, left, 0.0, -y_res, top)

    def _slice_for_day(
        self, path: Path, var: str, day: datetime.date
    ) -> tuple[npt.NDArray[np.float64], Affine] | None:
        """Read the ``var`` slice stamped exactly ``day`` as ``(array, transform)``.

        The array is oriented north-up (rows of decreasing latitude) to match the
        returned affine. Returns ``None`` if the file is missing or has no slice for
        ``day`` — the caller then treats the band as a placeholder.
        """
        if not path.exists():
            return None
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            time_name = "valid_time" if "valid_time" in ds.coords else "time"
            times = ds[time_name].values.astype("datetime64[D]")
            matches = np.where(times == np.datetime64(day))[0]
            if matches.size == 0:
                return None
            arr = ds[var].isel({time_name: int(matches[0])}).values.astype(np.float64)
            lats = np.asarray(ds["latitude"].values)
            lons = np.asarray(ds["longitude"].values)
        # Orient north-up (rows of decreasing latitude) to match the affine.
        if lats[0] < lats[-1]:
            arr = arr[::-1]
            lats = lats[::-1]
        return arr, self._src_transform(lats, lons)

    def _read_band(
        self, band: str, day: datetime.date
    ) -> tuple[npt.NDArray[np.float64], Affine] | None:
        """Read one band's native-grid slice for ``day``, applying the precip shift.

        Returns ``(array, src_transform)`` or ``None`` when the slice is absent.
        """
        if band == "total_precipitation_sum":
            # Accumulation closing day d is stamped at 00:00 of d+1.
            shifted = day + datetime.timedelta(days=1)
            return self._slice_for_day(self._precip_file(shifted), _PRECIP_VAR, shifted)
        fname, var = _INSTANT_FILES[band]
        return self._slice_for_day(self._monthly_folder(day) / fname, var, day)

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the five ERA5 bands for ``(cell, day)`` on the cell grid.

        Args:
            cell: Target :class:`GridCell` (UTM ``crs``/``transform``/``shape``).
            day: The inference day. ``None`` → all-``-9999`` (no day to resolve).

        Returns:
            ``(5, H, W)`` ``float32`` array in raw Kelvin/native units; any band whose
            slice is missing is filled with ``-9999`` for the whole cell.
        """
        if day is None:
            return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)

        bands: list[npt.NDArray[np.floating]] = []
        n_present = 0
        for name in self.bands_out:
            read = self._read_band(name, day)
            if read is None:
                bands.append(create_placeholder(n_bands=1, shape=cell.shape)[0])
                continue
            arr, src_transform = read
            # Nearest (categorical path) — coarse grid, GEE-parity (module docstring).
            reprojected = reproject_to_cell(
                source=arr[np.newaxis, :, :],
                src_transform=src_transform,
                src_crs=_SOURCE_CRS,
                cell=cell,
                categorical=True,
            )
            bands.append(reprojected[0])
            n_present += 1

        logger.info(
            "era5_fetch", cell_id=cell.cell_id, day=day.isoformat(), bands_present=n_present
        )
        return np.stack(bands, axis=0).astype(np.float32)
