"""List + parse Stage-2 outputs for the cube and daily-FSC viewer tabs.

These are **pipeline outputs** under ``processing_root`` (assembled per-cell cubes and
daily fractional-snow-cover COGs), NOT clip-manifest rows — so this module scans the
filesystem directly rather than reusing ``manifest.load_products``.

Filename contracts (verified on disk, see ``PLAN-V2-CUBE-FSC-TABS.md`` §3):

* Cube — ``PR_<YYYYMMDD>_<lat>_<lon>_SC<cc>.tif`` (308 bands, EPSG:32611, band
  descriptions ``<var>_t<idx>`` for the 38 dynamic vars × 8 timesteps, then 4
  un-suffixed statics ``DEM``/``slope``/``aspect``/``Map``).
* Daily FSC — ``fsc_<YYYYMMDD>.tif`` (single band, EPSG:32611, values ∈ [0, 1]).

The cube band catalogue is read from the cube's **own descriptions** (single source of
truth) — band selection downstream matches by description, never by a hardcoded offset, so
a future band-order change cannot silently mis-map a variable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
import structlog

from src.data.local_sources.layout import CUBE_FILENAME_PATTERN
from src.data.local_sources.viewer.settings import ViewerSettings

logger = structlog.get_logger(__name__)

#: ``fsc_<YYYYMMDD>.tif`` (daily FSC COG).
_FSC_FILENAME_PATTERN = re.compile(r"^fsc_(\d{8})\.tif$")

#: Trailing ``_t<idx>`` suffix on a dynamic cube band description.
_TIMESTEP_SUFFIX = re.compile(r"_t(\d+)$")


@dataclass(frozen=True)
class CubeRow:
    """One assembled per-cell cube on disk.

    Attributes:
        path: The ``PR_*.tif`` path.
        pred_date: Prediction day (the 8-day window end) parsed from the filename.
        lat: Cell-centre latitude (signed decimal degrees, EPSG:4326).
        lon: Cell-centre longitude (signed decimal degrees, EPSG:4326).
    """

    path: Path
    pred_date: date
    lat: float
    lon: float

    @property
    def cell_label(self) -> str:
        """Human-readable cell label for the picker (lat/lon to 4 dp)."""
        return f"{self.lat:.4f}, {self.lon:.4f}"


@dataclass(frozen=True)
class FscRow:
    """One daily fractional-snow-cover COG on disk.

    Attributes:
        path: The ``fsc_*.tif`` path.
        pred_date: The day the FSC was predicted for, parsed from the filename.
    """

    path: Path
    pred_date: date


def _parse_pred_date(token: str) -> date:
    """Parse a ``YYYYMMDD`` token to a :class:`date`."""
    return datetime.strptime(token, "%Y%m%d").date()


def list_cubes(settings: ViewerSettings | None = None) -> list[CubeRow]:
    """List assembled cubes under ``processing_root/cubes/``.

    Args:
        settings: Viewer settings; defaults to ``ViewerSettings()``.

    Returns:
        One :class:`CubeRow` per ``PR_*.tif``, sorted by ``(pred_date, lat, lon)``.
        Empty if the directory is absent or holds no matching files.
    """
    settings = settings or ViewerSettings()
    cubes_dir = settings.cubes_dir
    if not cubes_dir.exists():
        return []

    rows: list[CubeRow] = []
    for path in cubes_dir.glob("PR_*.tif"):
        if not CUBE_FILENAME_PATTERN.match(path.name):
            logger.warning("skipping_unparseable_cube", name=path.name)
            continue
        # PR_<YYYYMMDD>_<lat>_<lon>_SC<cc>.tif
        _, ymd, lat, lon, _sc = path.stem.split("_")
        rows.append(
            CubeRow(
                path=path,
                pred_date=_parse_pred_date(ymd),
                lat=float(lat),
                lon=float(lon),
            )
        )
    return sorted(rows, key=lambda r: (r.pred_date, r.lat, r.lon))


def list_fsc(settings: ViewerSettings | None = None) -> list[FscRow]:
    """List daily-FSC COGs under ``processing_root/daily_fsc/``.

    Args:
        settings: Viewer settings; defaults to ``ViewerSettings()``.

    Returns:
        One :class:`FscRow` per ``fsc_*.tif``, sorted by ``pred_date``. Empty if the
        directory is absent or holds no matching files.
    """
    settings = settings or ViewerSettings()
    fsc_dir = settings.daily_fsc_dir
    if not fsc_dir.exists():
        return []

    rows: list[FscRow] = []
    for path in fsc_dir.glob("fsc_*.tif"):
        match = _FSC_FILENAME_PATTERN.match(path.name)
        if match is None:
            logger.warning("skipping_unparseable_fsc", name=path.name)
            continue
        rows.append(FscRow(path=path, pred_date=_parse_pred_date(match.group(1))))
    return sorted(rows, key=lambda r: r.pred_date)


def dates_for_cubes(rows: list[CubeRow]) -> list[date]:
    """Distinct prediction dates across ``rows``, ascending."""
    return sorted({r.pred_date for r in rows})


def cubes_for_date(rows: list[CubeRow], pred_date: date) -> list[CubeRow]:
    """Cubes whose ``pred_date == pred_date`` (already lat/lon-sorted)."""
    return [r for r in rows if r.pred_date == pred_date]


@dataclass(frozen=True)
class CubeBands:
    """The cube's band catalogue, read from its descriptions.

    Attributes:
        dynamic: Ordered distinct dynamic variable names (``_t<idx>`` stripped).
        statics: The trailing un-suffixed static band names.
        n_timesteps: Number of timesteps present for the dynamic block.
    """

    dynamic: list[str]
    statics: list[str]
    n_timesteps: int


def cube_variables(path: Path) -> CubeBands:
    """Read the cube's band catalogue from its descriptions (single source of truth).

    Dynamic bands carry a ``_t<idx>`` suffix; the trailing un-suffixed descriptions are
    the statics. The dynamic variable order is the first-seen order at ``t0``.

    Args:
        path: A cube ``PR_*.tif``.

    Returns:
        A :class:`CubeBands` catalogue.

    Raises:
        ValueError: If the cube carries no band descriptions to parse.
    """
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)

    if not any(descriptions):
        raise ValueError(f"cube {path.name} has no band descriptions to catalogue")

    dynamic: list[str] = []
    statics: list[str] = []
    max_timestep = -1
    for desc in descriptions:
        if desc is None:
            continue
        match = _TIMESTEP_SUFFIX.search(desc)
        if match is None:
            statics.append(desc)
            continue
        var = desc[: match.start()]
        if var not in dynamic:
            dynamic.append(var)
        max_timestep = max(max_timestep, int(match.group(1)))

    return CubeBands(dynamic=dynamic, statics=statics, n_timesteps=max_timestep + 1)


def band_index(path: Path, *, var: str, timestep: int) -> int:
    """Resolve the 1-based rasterio band for ``(var, timestep)`` by description.

    Matches the description ``f"{var}_t{timestep}"`` for a dynamic variable, or exactly
    ``var`` for a static. Matching by description (never an arithmetic offset) means a
    future band-order change cannot silently mis-map a variable.

    Args:
        path: A cube ``PR_*.tif``.
        var: The variable name (dynamic root such as ``"VV"`` or a static like ``"DEM"``).
        timestep: The timestep index for a dynamic var; ignored for statics.

    Returns:
        The 1-based band index.

    Raises:
        KeyError: If neither ``f"{var}_t{timestep}"`` nor ``var`` is a band description.
    """
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)

    wanted_dynamic = f"{var}_t{timestep}"
    for offset, desc in enumerate(descriptions):
        if desc == wanted_dynamic or desc == var:
            return offset + 1
    raise KeyError(f"no band {wanted_dynamic!r} or {var!r} in {path.name}")


@dataclass(frozen=True)
class CubeAvailability:
    """Which ``(variable, timestep)`` pairs hold *real* (non-nodata) data.

    A cube band always *exists* for every dynamic ``(var, timestep)`` pair, but most are
    entirely nodata at most timesteps — e.g. Landsat reflectance is real at only 1–2 of
    the 8 timesteps, S1 may be all-nodata in a cube. "Available" here means the band is
    **not entirely nodata / non-finite**, i.e. it carries at least one real pixel.

    Statics (``DEM``/``slope``/``aspect``/``Map``) have no timestep and are treated as
    always available (the timestep axis does not apply to them).

    Attributes:
        dynamic_real: Dynamic variable name → the set of timesteps at which its band
            carries real data. Variables that are all-nodata at every timestep map to an
            empty set (still present as a key, in catalogue order via ``dynamic_order``).
        dynamic_order: Dynamic variable names in first-seen (catalogue) order, so callers
            can present a stable, ``t0``-ordered list.
        statics: The trailing un-suffixed static band names (always available).
        n_timesteps: Number of timesteps in the dynamic block.
    """

    dynamic_real: dict[str, set[int]]
    dynamic_order: list[str]
    statics: list[str]
    n_timesteps: int


def cube_availability(path: Path) -> CubeAvailability:
    """Compute which ``(var, timestep)`` cube bands carry real (non-nodata) data.

    Reads every band once and marks a dynamic ``<var>_t<idx>`` band as *available* when it
    is not entirely nodata (the cube's ``nodata``, default ``-9999``) or non-finite. Cubes
    are small (~100×100), so the full read is cheap (sub-second). Statics are collected by
    name and are always considered available.

    Args:
        path: A cube ``PR_*.tif``.

    Returns:
        A :class:`CubeAvailability`.

    Raises:
        ValueError: If the cube carries no band descriptions to parse.
    """
    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        if not any(descriptions):
            raise ValueError(f"cube {path.name} has no band descriptions to catalogue")
        nodata = src.nodata if src.nodata is not None else -9999.0

        dynamic_real: dict[str, set[int]] = {}
        dynamic_order: list[str] = []
        statics: list[str] = []
        max_timestep = -1
        for band_idx, desc in enumerate(descriptions, start=1):
            if desc is None:
                continue
            match = _TIMESTEP_SUFFIX.search(desc)
            if match is None:
                statics.append(desc)
                continue
            var = desc[: match.start()]
            timestep = int(match.group(1))
            if var not in dynamic_real:
                dynamic_real[var] = set()
                dynamic_order.append(var)
            max_timestep = max(max_timestep, timestep)

            band = src.read(band_idx)
            is_fill = (band == nodata) | ~np.isfinite(band)
            if not bool(np.all(is_fill)):
                dynamic_real[var].add(timestep)

    return CubeAvailability(
        dynamic_real=dynamic_real,
        dynamic_order=dynamic_order,
        statics=statics,
        n_timesteps=max_timestep + 1,
    )


def vars_at_timestep(avail: CubeAvailability, timestep: int) -> list[str]:
    """Selectable variables for a given timestep: real dynamic vars, then all statics.

    Dynamic variables appear only if their band carries real data at ``timestep``; statics
    are always appended (they have no timestep axis). Order is the cube catalogue order for
    dynamics, then the static order.

    Args:
        avail: The cube's availability.
        timestep: The timestep index.

    Returns:
        Variable names selectable at ``timestep`` (dynamics first, then statics).
    """
    dynamic = [v for v in avail.dynamic_order if timestep in avail.dynamic_real[v]]
    return [*dynamic, *avail.statics]


def timesteps_for_var(avail: CubeAvailability, var: str) -> list[int]:
    """Timesteps at which ``var`` carries real data (ascending).

    Returns an empty list for a static (the timestep axis does not apply) or for a dynamic
    variable that is all-nodata at every timestep.

    Args:
        avail: The cube's availability.
        var: The variable name.

    Returns:
        The sorted real timesteps for ``var``; ``[]`` for a static or an all-nodata var.
    """
    if var in avail.statics:
        return []
    return sorted(avail.dynamic_real.get(var, set()))
