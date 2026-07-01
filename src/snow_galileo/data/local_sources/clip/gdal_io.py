"""System-GDAL helpers for HDF4 (and other non-rasterio-readable) containers.

rasterio's bundled GDAL build lacks the HDF4 driver, so MODIS ``.hdf`` tiles
cannot be opened with ``rasterio.open``. These helpers shell out to the system
``gdalinfo`` / ``gdal_translate`` (GDAL 3.8+ with HDF4) to enumerate subdatasets
and extract them to GeoTIFF, after which rasterio can read the result.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SubdatasetInfo:
    """One enumerated subdataset of an HDF container.

    Attributes:
        name: The GDAL subdataset connection string (``HDF4_EOS:...:grid:band``).
        grid: The grid/group token parsed from the name (e.g.
            ``MODIS_Grid_500m_2D``), used to group bands by native resolution.
        band: The science/QA band name (e.g. ``sur_refl_b01``).
        width: Pixel width of the subdataset's grid.
        height: Pixel height of the subdataset's grid.
    """

    name: str
    grid: str
    band: str
    width: int
    height: int


def gdalinfo_json(path: Path | str) -> dict:
    """Run ``gdalinfo -json`` and return the parsed metadata.

    Args:
        path: A file path or a GDAL subdataset connection string.

    Returns:
        The parsed ``gdalinfo`` JSON document.

    Raises:
        subprocess.CalledProcessError: If ``gdalinfo`` exits non-zero.
    """
    result = subprocess.run(
        ["gdalinfo", "-json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def list_subdatasets(hdf_path: Path) -> list[SubdatasetInfo]:
    """Enumerate an HDF container's subdatasets, grouped by native grid.

    Args:
        hdf_path: Path to an HDF4 (``.hdf``) or HDF5 (``.h5``) container.

    Returns:
        One :class:`SubdatasetInfo` per subdataset. Empty if the container has
        no subdatasets.
    """
    meta = gdalinfo_json(hdf_path)
    subs = meta.get("metadata", {}).get("SUBDATASETS", {})
    names = [v for k, v in sorted(subs.items()) if k.endswith("_NAME")]

    infos: list[SubdatasetInfo] = []
    for name in names:
        sub_meta = gdalinfo_json(name)
        size = sub_meta.get("size", [0, 0])
        grid, band = _parse_grid_band(name)
        infos.append(
            SubdatasetInfo(
                name=name,
                grid=grid,
                band=band,
                width=int(size[0]),
                height=int(size[1]),
            )
        )
    return infos


def _parse_grid_band(name: str) -> tuple[str, str]:
    """Parse ``(grid, band)`` tokens from a GDAL subdataset connection string.

    Two descriptor dialects appear in this pipeline, and they delimit the grid
    and band differently:

    * **HDF4 (MODIS)** — ``HDF4_EOS:EOS_GRID:"path":GRID:BAND``. Grid and band are
      the last two ``:``-delimited fields.
    * **HDF5 (VIIRS)** — ``HDF5:"path"://HDFEOS/GRIDS/GRID/Data_Fields/BAND``. The
      grid and band live inside the HDF5 group path *after* ``://`` and are
      ``/``-delimited; a ``:``-split would wrongly capture the quoted file path
      and the slash-laden group path.

    Both dialects yield clean identifier tokens (no quotes or slashes), safe to
    use in output filenames.

    Args:
        name: The GDAL subdataset connection string.

    Returns:
        A ``(grid, band)`` tuple. Empty strings if the form is unrecognised.
    """
    # HDF5 group-path form: everything after the "://" is a "/"-delimited path.
    if "://" in name:
        group_path = name.split("://", 1)[1]
        segments = [s for s in group_path.split("/") if s]
        band = segments[-1] if segments else ""
        # The grid is the segment following the "GRIDS" container, when present.
        grid = ""
        if "GRIDS" in segments:
            idx = segments.index("GRIDS")
            if idx + 1 < len(segments):
                grid = segments[idx + 1]
        elif len(segments) >= 2:
            grid = segments[-2]
        return grid, band

    # HDF4 ":"-delimited form: GRID and BAND are the last two fields.
    parts = name.split(":")
    grid = parts[-2] if len(parts) >= 2 else ""
    band = parts[-1] if parts else ""
    return grid, band


def translate_subdataset(subdataset_name: str, out_tif: Path) -> None:
    """Extract a single subdataset to a GeoTIFF via ``gdal_translate``.

    The output preserves the subdataset's native CRS and geotransform, so the
    sinusoidal georeferencing survives the round-trip.

    Args:
        subdataset_name: GDAL subdataset connection string.
        out_tif: Destination GeoTIFF path (parent dirs created by the caller).

    Raises:
        subprocess.CalledProcessError: If ``gdal_translate`` exits non-zero.
    """
    subprocess.run(
        ["gdal_translate", "-of", "GTiff", subdataset_name, str(out_tif)],
        capture_output=True,
        text=True,
        check=True,
    )
