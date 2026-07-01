"""S3 OLCI ortho → COPERNICUS/S3/OLCI value-domain parity logic (TASK-011 follow-up).

Tests whether ESA SNAP's ``Reproject(orthorectify=true)`` closes the geolocation
residual the production swath-warp adapter (``src/data/local_sources/s3.py``) leaves.

**The residual under test.** The adapter warps the OLCI curvilinear swath onto the
cell grid with ``scipy.griddata`` and the full per-pixel ``geo_coordinates.nc``. The
radiance *scale* is bit-identical to GEE (Oa17 0.00493004); the ~18 % median / corr
~0.67 gap is OLCI's intrinsic **un-orthorectified geolocation** — GEE terrain-
orthorectifies OLCI in SNAP, which a plain warp cannot reproduce (PARITY_SPIKE_NOTES
§10). A rigid shift does not close it, so it is per-pixel terrain distortion.

**Toolchain.** SNAP ``Reproject`` with ``orthorectify=true`` + a DEM is the OLCI ortho
path: it uses the product's tie-point geocoding plus SRTM to terrain-correct the
optical swath. (The SAR ``Terrain-Correction`` / ``Ellipsoid-Correction`` ops are
radar-geometry only and reject an OLCI product.)

``run_s3_ortho_spike`` runs ``s3_olci_ortho_graph.xml`` over the patch AOI and
reprojects the ortho'd GeoTIFF onto the reference patch grid; the wrapper CLI then
reports median |Δ| + correlation against the GEE reference vs the adapter's own
numbers on the same cell. **Verdict (closed, [[s3-snap-ortho-rejected]]):** ortho did
**not** lift correlation past the swath-warp's ~0.67 — kept as evidence only.

**Caveat.** The patch is ~3 OLCI pixels wide (~300 m px over a ~1 km cell), so the
correlation is dominated by a handful of edge pixels.

This is a **parity de-risk module**, not the production adapter. The command-line
entrypoint is the thin wrapper at
``scripts/developer_scripts/bow_valley_inference_local/spikes/run_s3_parity.py``,
which supplies the SNAP graph path.
"""

from __future__ import annotations

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

#: Margin (degrees) around the patch for the SNAP subset region.
_AOI_MARGIN_DEG: float = 0.06

#: Default ESA SNAP ``gpt`` location (override via kwarg / CLI).
DEFAULT_GPT = Path("/home/dev/esa-snap/bin/gpt")

#: The radiance bands, in the graph's Subset / Write order.
S3_BANDS: list[str] = ["Oa17_radiance", "Oa21_radiance"]


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


def _run_snap_ortho(
    manifest: Path,
    aoi_wkt: str,
    out_tif: Path,
    gpt: Path,
    graph: Path,
) -> Path:
    """Run the SNAP OLCI ortho graph via ``gpt``; return the ortho'd GeoTIFF path.

    Args:
        manifest: The product's ``.SEN3/xfdumanifest.xml`` (SNAP's SEN3 entry point).
        aoi_wkt: lon/lat WKT polygon to subset to.
        out_tif: Output GeoTIFF path (Oa17, Oa21, orthorectified, EPSG:32611).
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
        f"-Pinput={manifest}",
        f"-Pregion={aoi_wkt}",
        f"-Poutput={out_tif}",
        "-c",
        "4G",
        "-q",
        "4",
    ]
    logger.info("snap_ortho_start", manifest=manifest.name, out=str(out_tif))
    subprocess.run(cmd, check=True)
    return out_tif


def run_s3_ortho_spike(
    *,
    product_zip: Path,
    reference_patch: Path,
    graph: Path,
    workdir: Path | None = None,
    gpt: Path = DEFAULT_GPT,
) -> dict[str, npt.NDArray[np.floating]]:
    """Run the S3 ortho spike for one product; return ``{band: array}`` on the patch grid.

    Extracts the SEN3 product (if needed), runs the SNAP ortho graph over the patch
    AOI, and reprojects each ortho'd radiance band onto the reference patch grid.

    Args:
        product_zip: The clipped ``S3?_OL_1_EFR____*.zip``.
        reference_patch: GEE reference patch defining the target grid.
        graph: Path to the SNAP OLCI ortho graph XML.
        workdir: Scratch dir (a sibling of the zip is used if ``None``).
        gpt: Path to the ESA SNAP ``gpt`` executable.

    Returns:
        ``{"Oa17_radiance": arr, "Oa21_radiance": arr}`` on the patch grid (NaN = no data).
    """
    work = workdir or (product_zip.parent / "_s3_ortho_spike")
    work.mkdir(parents=True, exist_ok=True)

    sen3_dirs = list(work.glob("*.SEN3"))
    if not sen3_dirs:
        with zipfile.ZipFile(product_zip) as zf:
            zf.extractall(work)
        sen3_dirs = list(work.glob("*.SEN3"))
    manifest = sen3_dirs[0] / "xfdumanifest.xml"

    out_tif = work / f"{product_zip.stem}_ortho.tif"
    if not out_tif.exists():
        _run_snap_ortho(manifest, _aoi_wkt(reference_patch), out_tif, gpt, graph)

    cell = _grid_from_patch(reference_patch)
    out: dict[str, npt.NDArray[np.floating]] = {}
    with rasterio.open(out_tif) as snap:
        descriptions = list(snap.descriptions)
        logger.info("snap_ortho_bands", descriptions=descriptions, count=snap.count)
        for band in S3_BANDS:
            # Map by band description, not index (SNAP Write order is not guaranteed).
            idx = next(
                (i + 1 for i, d in enumerate(descriptions) if d and band in d),
                S3_BANDS.index(band) + 1,
            )
            arr = snap.read(idx).astype(np.float64)
            arr[arr <= -1.0] = np.nan  # below the radiance valid floor → no-data
            out[band] = reproject_to_cell(
                source=arr[np.newaxis, :, :],
                src_transform=snap.transform,
                src_crs=snap.crs.to_string(),
                cell=cell,
                categorical=False,
                restore_fill=np.nan,
            )[0]
    return out


def stats(spike: npt.NDArray[np.floating], ref: np.ndarray) -> tuple[float, float, int]:
    """Return ``(median_abs_diff, correlation, n_valid)`` over co-valid pixels."""
    valid = (ref > -1.0) & np.isfinite(spike)
    n = int(valid.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    med = float(np.median(np.abs(spike[valid] - ref[valid])))
    corr = float(np.corrcoef(spike[valid], ref[valid])[0, 1])
    return med, corr, n
