"""S2 L1C → COPERNICUS/S2_HARMONIZED parity spike (TASK-005, throwaway).

Recreates the *value domain* GEE's ``COPERNICUS/S2_HARMONIZED`` produces for an
L1C granule, for one cell, and diffs it against the Phase-0 reference patch.

Recipe (baseline ≥ N0400, all archive granules N0511): read the L1C JP2 bands,
subtract the **−1000 DN** harmonization offset, reproject onto the reference
patch's exact ``(crs, transform, shape)`` cell grid via the shared
:func:`src.data.local_sources.base.reproject_to_cell` (bilinear; upsamples the
20 m B11/B12 to the 10 m grid). No atmospheric correction — L1C is TOA, and
``S2_HARMONIZED`` is too.

This is a **throwaway de-risk script**, not the production S2 adapter (TASK-013).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from shapely.geometry import box

from src.data.local_sources.base import GridCell, reproject_to_cell

logger = structlog.get_logger(__name__)

#: The harmonization offset GEE subtracts from baseline ≥ N0400 L1C DN.
S2_HARMONIZE_OFFSET_DN: int = 1000

#: S2 bands the cube carries, in canonical order (matches layout DYNAMIC_BANDS[3:9]).
S2_BANDS: list[str] = ["B2", "B3", "B4", "B8", "B11", "B12"]

#: JP2 filename suffix per cube band (S2 JP2s use zero-padded numbers).
_JP2_SUFFIX: dict[str, str] = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B8": "B08",
    "B11": "B11",
    "B12": "B12",
}


def _grid_from_patch(reference_patch: Path) -> GridCell:
    """Build a :class:`GridCell` mirroring the reference patch's exact grid.

    The patch's CRS/transform/shape *are* the parity target; the spike reprojects
    onto them so the diff is a direct per-pixel comparison.
    """
    with rasterio.open(reference_patch) as src:
        bounds = src.bounds
        height, width = src.height, src.width
        return GridCell(
            cell_id=0,
            crs=src.crs.to_string(),
            transform=src.transform,
            shape=(height, width),
            polygon=box(bounds.left, bounds.bottom, bounds.right, bounds.top),
        )


def _jp2_path(granule_zip: Path, band_suffix: str) -> str:
    """Return the ``/vsizip`` GDAL path to one band's JP2 inside the SAFE zip."""
    with zipfile.ZipFile(granule_zip) as zf:
        matches = [n for n in zf.namelist() if n.endswith(f"_{band_suffix}.jp2")]
    if not matches:
        raise FileNotFoundError(f"No _{band_suffix}.jp2 in {granule_zip}")
    return f"/vsizip/{granule_zip}/{matches[0]}"


def run_s2_spike(
    *,
    granule_zip: Path,
    reference_patch: Path,
) -> dict[str, npt.NDArray[np.floating]]:
    """Run the S2 spike for one granule and return ``{band: array}`` on the patch grid.

    Args:
        granule_zip: The L1C SAFE ``.zip``.
        reference_patch: The GEE reference patch whose grid defines the target.

    Returns:
        A dict mapping each S2 band name to its harmonized (post −1000 DN),
        reprojected ``(H, W)`` array on the reference patch's grid.
    """
    cell = _grid_from_patch(reference_patch)
    out: dict[str, npt.NDArray[np.floating]] = {}

    for band in S2_BANDS:
        vsipath = _jp2_path(granule_zip, _JP2_SUFFIX[band])
        with rasterio.open(vsipath) as src:
            dn = src.read(1).astype(np.float64)
            src_transform = src.transform
            src_crs = src.crs.to_string()

        # Harmonize: subtract the N0400+ offset. DN==0 is the L1C no-data/saturate
        # sentinel; keep it out of the valid domain by leaving it (the diff masks
        # reference zeros anyway, and reproject treats it as a value, not fill).
        harmonized = dn - S2_HARMONIZE_OFFSET_DN

        reprojected = reproject_to_cell(
            source=harmonized[np.newaxis, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            cell=cell,
            categorical=False,
        )[0]
        out[band] = reprojected

        finite = reprojected[np.isfinite(reprojected)]
        logger.info(
            "s2_spike_band",
            band=band,
            granule=granule_zip.name,
            min=float(finite.min()) if finite.size else None,
            max=float(finite.max()) if finite.size else None,
            mean=round(float(finite.mean()), 1) if finite.size else None,
        )
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="S2 L1C parity spike (TASK-005).")
    parser.add_argument(
        "--granule",
        type=Path,
        default=Path(
            "data/bow_valley_selection_raw/sentinel2/"
            "S2B_MSIL1C_20250403T184919_N0511_R113_T11UNS_20250403T222302.zip"
        ),
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("tests/fixtures/gee_reference_patches"),
        help="Reference-patch dir or a single patch tif.",
    )
    args = parser.parse_args()

    ref = args.ref
    if ref.is_dir():
        ref = ref / ("PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif")

    run_s2_spike(granule_zip=args.granule, reference_patch=ref)
    logger.info("s2_spike_done", reference=str(ref))


if __name__ == "__main__":
    _main()
