"""S1 GRD → COPERNICUS/S1_GRD value-domain parity logic (TASK-005).

Recreates the *value domain* GEE's ``COPERNICUS/S1_GRD`` produces (calibrated,
terrain-corrected σ⁰ in dB) for one IW GRD scene + cell, so VV/VH can be diffed
against the Phase-0 reference patch.

**Toolchain: ESA SNAP `gpt` — the engine GEE itself uses.** GEE's
``COPERNICUS/S1_GRD`` is the output of the SNAP Sentinel-1 Toolbox, so this runs
the SNAP graph (``s1_grd_snap_graph.xml``) via headless ``gpt``: Apply-Orbit
→ ThermalNoiseRemoval → Remove-GRD-Border-Noise → Calibration(σ⁰) →
Terrain-Correction(SRTM 1Sec, EPSG:32611, 10 m) → LinearToFromdB. This is the
**full** chain (no missing noise steps), so the parity verdict is unconditional.

**Why not `sarsen`/`xarray-sentinel`:** they cannot read Sentinel-1C SAFEs
(0.9.5: ``s1[ab]`` filename regex + a GCP-reader that returns a zero-size array
for S1C). The Bow Valley S1 archive is all S1C. See PARITY_SPIKE_NOTES.md §2 and
the ``xarray-sentinel-s1c-regex-bug`` memory note. The real S1 adapter (TASK-014)
must likewise drive SNAP (or an equivalent S1C-capable chain), not `xarray-sentinel`.

**Operational notes:**
- ``gpt`` is at ``$HOME/esa-snap/bin/gpt`` (NOT ``/usr/bin/snap`` = snapd).
- The graph subsets to the AOI in radar geometry before TC — terrain-correcting
  the full ~250 km swath at 10 m overflows SNAP's 4 GB classic-GeoTIFF writer and
  wastes compute.
- SNAP emits the bands **VH-then-VV** (not the graph's ``VV,VH`` order); this
  module assigns them by matching against the reference medians, not by index.

This is a **parity de-risk module**, not the production S1 adapter (TASK-014). The
command-line entrypoint is the thin wrapper at
``scripts/developer_scripts/bow_valley_inference_local/spikes/run_s1_parity.py``,
which supplies the SNAP graph path.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from rasterio.warp import transform_bounds
from shapely.geometry import box

from snow_galileo.data.local_sources.base import GridCell, reproject_to_cell

logger = structlog.get_logger(__name__)

#: SAR backscatter below this (dB) is treated as edge/no-data (GEE S1_GRD convention).
S1_EDGE_MASK_DB: float = -30.0

#: Margin (degrees) added around the patch when building the SNAP subset region.
_AOI_MARGIN_DEG: float = 0.06

#: Default ESA SNAP ``gpt`` location (override via kwarg / CLI).
DEFAULT_GPT = Path(f"{os.getenv('HOME')}/esa-snap/bin/gpt")


def _grid_from_patch(reference_patch: Path) -> GridCell:
    """Build a :class:`GridCell` mirroring the reference patch's exact grid."""
    with rasterio.open(reference_patch) as src:
        b = src.bounds
        return GridCell(
            cell_id=0,
            crs=src.crs.to_string(),
            transform=src.transform,
            shape=(src.height, src.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


def _aoi_wkt(reference_patch: Path) -> str:
    """Return a lon/lat WKT polygon for the patch bbox + margin (SNAP geoRegion)."""
    with rasterio.open(reference_patch) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    m = _AOI_MARGIN_DEG
    lo0, la0, lo1, la1 = lon0 - m, lat0 - m, lon1 + m, lat1 + m
    return f"POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))"


def _run_snap_chain(
    safe_dir: Path,
    aoi_wkt: str,
    out_tif: Path,
    gpt: Path,
    graph: Path,
) -> Path:
    """Run the SNAP S1_GRD graph via ``gpt``; return the dB GeoTIFF path.

    Args:
        safe_dir: Extracted ``.SAFE`` directory.
        aoi_wkt: lon/lat WKT polygon to subset to.
        out_tif: Output dB GeoTIFF path.
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the SNAP graph XML.

    Returns:
        ``out_tif``.

    Raises:
        FileNotFoundError: If ``gpt`` is not available.
        subprocess.CalledProcessError: If the SNAP chain fails.
    """
    if not gpt.exists():
        raise FileNotFoundError(f"ESA SNAP gpt not found at {gpt}; install SNAP and pass --gpt.")
    cmd = [
        str(gpt),
        str(graph),
        f"-Pinput={safe_dir / 'manifest.safe'}",
        f"-Pregion={aoi_wkt}",
        f"-Poutput={out_tif}",
        "-c",
        "4G",
        "-q",
        "4",
    ]
    logger.info("snap_chain_start", safe=safe_dir.name, out=str(out_tif))
    subprocess.run(cmd, check=True)
    return out_tif


def run_s1_spike(
    *,
    granule_zip: Path,
    reference_patch: Path,
    graph: Path,
    workdir: Path | None = None,
    gpt: Path = DEFAULT_GPT,
) -> dict[str, npt.NDArray[np.floating]]:
    """Run the S1 spike for one granule; return ``{VV, VH}`` dB on the patch grid.

    Extracts the SAFE (if needed), runs the SNAP S1_GRD chain over the patch AOI,
    masks ``< -30 dB``, reprojects onto the reference patch grid, and assigns the
    two output bands to VV/VH by best agreement with the reference medians (SNAP
    writes VH-then-VV).

    Args:
        granule_zip: The IW GRD SAFE ``.zip``.
        reference_patch: GEE reference patch defining the target grid.
        graph: Path to the SNAP S1_GRD graph XML.
        workdir: Scratch dir (a sibling of the zip is used if ``None``).
        gpt: Path to the ESA SNAP ``gpt`` executable.

    Returns:
        ``{"VV": arr, "VH": arr}`` — σ⁰ dB, ``< -30`` masked to NaN, on the patch grid.
    """
    work = workdir or (granule_zip.parent / "_s1_spike")
    work.mkdir(parents=True, exist_ok=True)

    safe_dirs = list(work.glob("*.SAFE"))
    if not safe_dirs:
        with zipfile.ZipFile(granule_zip) as zf:
            zf.extractall(work)
        safe_dirs = list(work.glob("*.SAFE"))
    safe_dir = safe_dirs[0]

    out_tif = work / "s1_grd_db.tif"
    if not out_tif.exists():
        _run_snap_chain(safe_dir, _aoi_wkt(reference_patch), out_tif, gpt, graph)

    cell = _grid_from_patch(reference_patch)
    with rasterio.open(reference_patch) as ref:
        ref_vv = ref.read(1).astype(np.float64)
        ref_vh = ref.read(2).astype(np.float64)

    reprojected: list[npt.NDArray[np.floating]] = []
    with rasterio.open(out_tif) as snap:
        for band_index in (1, 2):
            db = snap.read(band_index).astype(np.float64)
            masked = np.where(db < S1_EDGE_MASK_DB, np.nan, db)
            reprojected.append(
                reproject_to_cell(
                    source=masked[np.newaxis, :, :],
                    src_transform=snap.transform,
                    src_crs=snap.crs.to_string(),
                    cell=cell,
                    categorical=False,
                    restore_fill=np.nan,
                )[0]
            )

    def _median_abs(spike: npt.NDArray[np.floating], ref: np.ndarray) -> float:
        valid = (ref != -9999) & (ref != 0) & np.isfinite(spike)
        return float(np.median(np.abs(spike[valid] - ref[valid])))

    out: dict[str, npt.NDArray[np.floating]] = {}
    for ref_arr, name in ((ref_vv, "VV"), (ref_vh, "VH")):
        best = min(range(2), key=lambda i: _median_abs(reprojected[i], ref_arr))
        out[name] = reprojected[best]
        logger.info(
            "s1_spike_band",
            pol=name,
            median_abs_diff=round(_median_abs(reprojected[best], ref_arr), 2),
        )
    return out
