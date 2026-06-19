r"""Build slim S2 SAFE zips for the parity tests — only the read bands, windowed.

The S2 parity tests (``test_s2_adapter`` B4/QA60, ``test_s2_parity`` spike) each
read **one** clipped L1C SAFE zip per case, but only the six cube bands
(B2,B3,B4,B8,B11,B12) + ``MSK_CLASSI`` over a single ~1 km patch window. A full
clipped SAFE is ~400 MB; this repackages each into a slim SAFE zip (~tens of MB)
holding just those bands cropped to the patch window — preserving each band's CRS
and geotransform so the adapter's windowed read (``cell_window`` →
``reproject_to_cell``) sees byte-identical pixels and parity stays **bit-exact**.

Crops are written **lossless** (``REVERSIBLE=YES``, ``QUALITY=100``): lossy JP2
recompression corrupts both reflectance and the categorical MSK_CLASSI (the
``s2-clip-lossy-jp2-bug`` regression), which the bit-exact parity would catch.

Run from the repo root with the real ``data/`` clipped S2 archive present::

    uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/build_slim_s2_safe.py --include-raw

Output is the committed ``tests/fixtures/clipped/sentinel2`` (and, with
``--include-raw``, ``clipped/sentinel2_raw``) — small + bit-exact, so the S2 parity
tests run from committed data on any full-suite run (CI itself excludes ``slow``).
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import rasterio
import structlog
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from src.data.local_sources.base import GridCell

logger = structlog.get_logger(__name__)

_REF_DIR = Path("tests/fixtures/gee_reference_patches")
#: Slim S2 zips are small + bit-exact → committed (their tests run in CI).
_OUT_ROOT = Path("tests/fixtures/clipped/sentinel2")

#: Candidate source roots for the full SAFE zips, in preference order.
_SRC_ROOTS = [
    Path("data/clipped_bow_valley_selection_raw/sentinel2"),
]

#: The band JP2 suffixes the adapter reads (cube bands), plus the cloud mask.
_BAND_SUFFIXES = ["B02", "B03", "B04", "B08", "B11", "B12"]

#: Extra pixels around the cell window so ``cell_window``'s own margin always fits.
_CROP_MARGIN_PX = 64

#: parity_case patch_key → the exact clipped SAFE zip the case's cell falls in
#: (one tile per case; the sibling tiles of the same date are never read).
_CASE_ZIP = {
    "PR_20250406": "S2B_MSIL1C_20250403T184919_N0511_R113_T11UNS_20250403T222302.zip",
    "PR_20250414": "S2C_MSIL1C_20250408T184941_N0511_R113_T11UNT_20250408T223716.SAFE.zip",
    "PR_20250423": "S2B_MSIL1C_20250423T184919_N0511_R113_T11UNS_20250423T205050.zip",
    "PR_20250510": "S2B_MSIL1C_20250503T184919_N0511_R113_T11UPT_20250503T222250.zip",
}


def _find_src(zip_name: str) -> Path | None:
    for root in _SRC_ROOTS:
        candidate = root / zip_name
        if candidate.exists():
            return candidate
    return None


def _cell_for(patch_key: str) -> GridCell:
    patch = sorted(_REF_DIR.glob(f"{patch_key}_*.tif"))[0]
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=0,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


def _crop_jp2_member(zf: zipfile.ZipFile, member: str, cell: GridCell, tmp: Path) -> Path:
    """Crop one JP2 zip member to the cell window (+margin), lossless; return the temp path."""
    with rasterio.open(f"/vsizip/{zf.filename}/{member}") as src:
        lo, la, hi, ha = transform_bounds(cell.crs, src.crs, *cell.polygon.bounds)
        win = from_bounds(lo, la, hi, ha, transform=src.transform)
        m = _CROP_MARGIN_PX
        row0 = max(0, int(win.row_off) - m)
        col0 = max(0, int(win.col_off) - m)
        row1 = min(src.height, int(win.row_off + win.height) + m)
        col1 = min(src.width, int(win.col_off + win.width) + m)
        window = Window(col0, row0, col1 - col0, row1 - row0)
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            height=int(window.height),
            width=int(window.width),
            transform=src.window_transform(window),
            driver="JP2OpenJPEG",
            QUALITY=100,
            REVERSIBLE=True,
        )
    out = tmp / Path(member).name
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data)
    return out


def build_slim(patch_key: str, zip_name: str) -> None:
    src = _find_src(zip_name)
    if src is None:
        logger.warning("slim_s2_source_missing", zip=zip_name)
        return
    cell = _cell_for(patch_key)
    out_zip = _OUT_ROOT / zip_name
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(src) as zin:
        tmp = Path(td)
        names = zin.namelist()
        mtd = next(n for n in names if n.endswith("MTD_MSIL1C.xml"))
        band_members = {
            suf: next((n for n in names if n.endswith(f"_{suf}.jp2") and "/IMG_DATA/" in n), None)
            for suf in _BAND_SUFFIXES
        }
        msk = next((n for n in names if "MSK_CLASSI" in n and n.endswith(".jp2")), None)

        # Write to a temp file then atomically replace, so a full zip is never left half-built.
        tmp_zip = out_zip.with_suffix(f"{out_zip.suffix}.partial")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr(mtd, zin.read(mtd))  # baseline XML, verbatim
            kept = 0
            for suf, member in band_members.items():
                if member is None:
                    logger.warning("slim_s2_band_missing", zip=zip_name, band=suf)
                    continue
                cropped = _crop_jp2_member(zin, member, cell, tmp)
                zout.write(cropped, arcname=member)
                kept += 1
            if msk is not None:
                cropped = _crop_jp2_member(zin, msk, cell, tmp)
                zout.write(cropped, arcname=msk)
                kept += 1
    tmp_zip.replace(out_zip)
    logger.info(
        "slim_s2_written",
        zip=zip_name,
        bands=kept,
        mb=round(out_zip.stat().st_size / 1e6, 1),
    )


#: The raw value-domain granule (test_s2_parity), its patch cell, and the source
#: + output roots (the spike reads no MSK_CLASSI — bands only).
_RAW_GRANULE = "S2B_MSIL1C_20250403T184919_N0511_R113_T11UNS_20250403T222302.zip"
_RAW_PATCH_KEY = "PR_20250406"
_RAW_SRC_ROOTS = [
    Path("data/bow_valley_selection_raw/sentinel2"),
]
_RAW_OUT_ROOT = Path("tests/fixtures/clipped/sentinel2_raw")


def build_slim_raw() -> None:
    """Slim the raw value-domain granule the spike harmonizes (6 bands, windowed)."""
    src = next((r / _RAW_GRANULE for r in _RAW_SRC_ROOTS if (r / _RAW_GRANULE).exists()), None)
    if src is None:
        logger.warning("slim_s2_raw_source_missing", zip=_RAW_GRANULE)
        return
    cell = _cell_for(_RAW_PATCH_KEY)
    _RAW_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_zip = _RAW_OUT_ROOT / _RAW_GRANULE

    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(src) as zin:
        tmp = Path(td)
        names = zin.namelist()
        mtd = next(n for n in names if n.endswith("MTD_MSIL1C.xml"))
        tmp_zip = out_zip.with_suffix(f"{out_zip.suffix}.partial")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr(mtd, zin.read(mtd))
            kept = 0
            for suf in _BAND_SUFFIXES:
                member = next(
                    (n for n in names if n.endswith(f"_{suf}.jp2") and "/IMG_DATA/" in n), None
                )
                if member is None:
                    logger.warning("slim_s2_raw_band_missing", band=suf)
                    continue
                zout.write(_crop_jp2_member(zin, member, cell, tmp), arcname=member)
                kept += 1
    tmp_zip.replace(out_zip)
    logger.info("slim_s2_raw_written", bands=kept, mb=round(out_zip.stat().st_size / 1e6, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs="+",
        choices=sorted(_CASE_ZIP),
        default=sorted(_CASE_ZIP),
        help="Which parity cases' clipped SAFE zips to slim.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also slim the raw value-domain granule (test_s2_parity).",
    )
    args = parser.parse_args()
    for patch_key in args.case:
        build_slim(patch_key, _CASE_ZIP[patch_key])
    if args.include_raw:
        build_slim_raw()


if __name__ == "__main__":
    main()
