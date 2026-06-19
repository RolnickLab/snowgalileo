"""S3 OLCI ortho → COPERNICUS/S3/OLCI parity spike (TASK-011 follow-up, throwaway).

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

The spike runs ``s3_olci_ortho_graph.xml`` over the patch AOI, reprojects the ortho'd
GeoTIFF onto the reference patch grid, and reports median |Δ| + correlation against
the GEE reference for Oa17/Oa21 — compared to the adapter's own numbers on the same
cell. **Go/no-go:** does ortho lift corr meaningfully past the swath-warp's ~0.67?

**Caveat (flagged before building).** The patch is ~3 OLCI pixels wide (~300 m px over
a ~1 km cell), so the correlation is dominated by a handful of edge pixels and may be
too coarse to confirm a real improvement. This spike *quantifies* that, it does not
assume it away.

Throwaway de-risk script, not the production adapter.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from rasterio.warp import transform_bounds
from shapely.geometry import box

from src.data.local_sources.base import GridCell, reproject_to_cell
from src.data.local_sources.s3 import S3Adapter

logger = structlog.get_logger(__name__)

#: Margin (degrees) around the patch for the SNAP subset region.
_AOI_MARGIN_DEG: float = 0.06

#: Default SNAP graph + gpt location (override via CLI).
_DEFAULT_GPT = Path("/home/dev/esa-snap/bin/gpt")
_GRAPH = Path("scripts/spikes/s3_olci_ortho_graph.xml")

#: The radiance bands, in the graph's Subset / Write order.
_BANDS = ["Oa17_radiance", "Oa21_radiance"]

#: 38-band dynamic block: Oa17 at offset 15, Oa21 at 16 (matches test_s3_adapter).
_DYNAMIC_PER_TS = 38
_OFF = {"Oa17_radiance": 15, "Oa21_radiance": 16}


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
    workdir: Path | None = None,
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _GRAPH,
) -> dict[str, npt.NDArray[np.floating]]:
    """Run the S3 ortho spike for one product; return ``{band: array}`` on the patch grid.

    Extracts the SEN3 product (if needed), runs the SNAP ortho graph over the patch
    AOI, and reprojects each ortho'd radiance band onto the reference patch grid.

    Args:
        product_zip: The clipped ``S3?_OL_1_EFR____*.zip``.
        reference_patch: GEE reference patch defining the target grid.
        workdir: Scratch dir (a sibling of the zip is used if ``None``).
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the SNAP graph XML.

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
        for band in _BANDS:
            # Map by band description, not index (SNAP Write order is not guaranteed).
            idx = next(
                (i + 1 for i, d in enumerate(descriptions) if d and band in d),
                _BANDS.index(band) + 1,
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


def _stats(spike: npt.NDArray[np.floating], ref: np.ndarray) -> tuple[float, float, int]:
    """Return ``(median_abs_diff, correlation, n_valid)`` over co-valid pixels."""
    valid = (ref > -1.0) & np.isfinite(spike)
    n = int(valid.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    med = float(np.median(np.abs(spike[valid] - ref[valid])))
    corr = float(np.corrcoef(spike[valid], ref[valid])[0, 1])
    return med, corr, n


def _main() -> None:
    parser = argparse.ArgumentParser(description="S3 OLCI ortho parity spike (TASK-011, SNAP).")
    parser.add_argument(
        "--product",
        type=Path,
        default=Path(
            "data/clipped_bow_valley_selection_raw/sentinel3/"
            "S3A_OL_1_EFR____20250401T183122_20250401T183422_20250402T192445"
            "_0179_124_184_1980_PS1_O_NT_004.zip"
        ),
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path(
            "tests/fixtures/gee_reference_patches/"
            "PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif"
        ),
    )
    parser.add_argument("--day", type=str, default="2025-04-01")
    parser.add_argument("--timestep", type=int, default=2)
    parser.add_argument("--gpt", type=Path, default=_DEFAULT_GPT)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()

    day = datetime.date.fromisoformat(args.day)

    # Ortho spike on the patch grid.
    ortho = run_s3_ortho_spike(
        product_zip=args.product,
        reference_patch=args.ref,
        workdir=args.workdir,
        gpt=args.gpt,
    )

    # Baseline: the current swath-warp adapter on the same cell/day.
    cell = _grid_from_patch(args.ref)
    adapter = S3Adapter(archive_root=args.product.parent)
    warp = adapter.fetch(cell, day=day)
    warp_by_band = {b: warp[i].astype(np.float64) for i, b in enumerate(_BANDS)}

    with rasterio.open(args.ref) as ds:
        for band in _BANDS:
            ref = ds.read(_DYNAMIC_PER_TS * args.timestep + _OFF[band] + 1).astype(np.float64)
            o_med, o_corr, o_n = _stats(ortho[band], ref)
            w = warp_by_band[band]
            w[w == -9999] = np.nan
            w_med, w_corr, w_n = _stats(w, ref)
            logger.info(
                "s3_parity_compare",
                band=band,
                ortho_median_abs=round(o_med, 2),
                ortho_corr=round(o_corr, 3),
                ortho_n=o_n,
                warp_median_abs=round(w_med, 2),
                warp_corr=round(w_corr, 3),
                warp_n=w_n,
                corr_delta=round(o_corr - w_corr, 3),
            )

    logger.info("s3_ortho_spike_done", product=args.product.name, ref=args.ref.name)


if __name__ == "__main__":
    _main()
