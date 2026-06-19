r"""Build the slim adapter-test fixtures for DEM / MODIS / VIIRS / S3 from the real archive.

Crops only what the tests read — the specific acquisition dates and the band files
the adapters open — windowed to the reference-patch footprints. The windowed crop
is **bit-exact** vs the full tile for the adapters' reads, so the crops satisfy
parity tests too. Output tier per source (see ``_OUT_ROOT``):

- **dem / modis / viirs** → committed ``tests/fixtures/clipped/`` (small enough to
  commit; their tests then run in CI).
- **sentinel3** → gitignored ``tests/fixtures/archive/`` (kept full — the swath warp
  reads ``geo_coordinates``, cropping is risky — so it is download-only).

(S2 is handled by ``build_slim_s2_safe.py``, which repackages slim lossless SAFE zips.)

Run from the repo root with the real clipped archive present locally::

    uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/populate_test_archive.py --source dem modis viirs sentinel3
"""

from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path

import rasterio
import structlog
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

logger = structlog.get_logger(__name__)

_CLIPPED = Path("data/clipped_bow_valley_selection_raw")
#: Committed fixture tier (in git) — slim crops small enough to commit run in CI.
_COMMITTED = Path("tests/fixtures/clipped")
#: Gitignored fixture tier — only sources too large to commit (S3) land here.
_ARCHIVE = Path("tests/fixtures/archive")
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

#: Output tier per source. Slim, bit-exact crops are committed (run in CI); S3 is
#: kept full (swath warp reads geo_coordinates, cropping is risky) so it stays
#: gitignored and download-only.
_OUT_ROOT = {
    "dem": _COMMITTED,
    "modis": _COMMITTED,
    "viirs": _COMMITTED,
    "sentinel3": _ARCHIVE,
}

#: Pixel halo around each patch window when cropping (covers the reproject stencil;
#: DEM's bilinear warp needs the larger value or parity skews by metres).
_CROP_MARGIN_PX = 8
#: DEM is bilinear-warped onto the cell; the stencil reaches well past the cell, so
#: it needs a much wider halo than the nearest-sampled sources (measured: ~0.06° ≈
#: 230 px of 1″ DEM closes the parity gap; below it edge patches skew by metres).
_DEM_CROP_MARGIN_PX = 256


def _write_crop(src: rasterio.io.DatasetReader, window: Window, out_tif: Path) -> None:
    data = src.read(window=window)
    profile = src.profile.copy()
    profile.update(
        height=int(window.height),
        width=int(window.width),
        transform=src.window_transform(window),
        compress="deflate",
    )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data)


def _union_window(src: rasterio.io.DatasetReader, margin_px: int) -> Window | None:
    """Clamped window over the union of all reference-patch footprints in ``src``'s CRS."""
    rows: list[float] = []
    cols: list[float] = []
    for patch in sorted(_REF_DIR.glob("PR_*.tif")):
        with rasterio.open(patch) as pds:
            lo, la, hi, ha = transform_bounds(pds.crs, src.crs, *pds.bounds)
        win = from_bounds(lo, la, hi, ha, transform=src.transform)
        rows += [win.row_off, win.row_off + win.height]
        cols += [win.col_off, win.col_off + win.width]
    row0 = max(0, int(min(rows)) - margin_px)
    col0 = max(0, int(min(cols)) - margin_px)
    row1 = min(src.height, int(max(rows)) + margin_px)
    col1 = min(src.width, int(max(cols)) + margin_px)
    if row1 <= row0 or col1 <= col0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def _crop_union(src_tif: Path, out_tif: Path, *, margin_px: int) -> None:
    """Crop to the union window, flat (for adapters that glob band files in a granule dir)."""
    with rasterio.open(src_tif) as src:
        window = _union_window(src, margin_px)
        if window is not None:
            _write_crop(src, window, out_tif)


def _crop_per_patch(src_tif: Path, out_dir: Path, *, margin_px: int) -> int:
    """Write one crop of ``src_tif`` per intersecting patch (``p{i}/`` subdir), preserving CRS.

    For adapters that ``rglob`` and mosaic (DEM): one window per patch avoids bridging
    scattered patches into one huge bounding box. Returns the number of crops written.
    """
    written = 0
    with rasterio.open(src_tif) as src:
        tb = src.bounds
        for i, patch in enumerate(sorted(_REF_DIR.glob("PR_*.tif"))):
            with rasterio.open(patch) as pds:
                lo, la, hi, ha = transform_bounds(pds.crs, src.crs, *pds.bounds)
            if hi < tb.left or lo > tb.right or ha < tb.bottom or la > tb.top:
                continue
            win = from_bounds(lo, la, hi, ha, transform=src.transform)
            row0 = max(0, int(win.row_off) - margin_px)
            col0 = max(0, int(win.col_off) - margin_px)
            row1 = min(src.height, int(win.row_off + win.height) + margin_px)
            col1 = min(src.width, int(win.col_off + win.width) + margin_px)
            if row1 <= row0 or col1 <= col0:
                continue
            _write_crop(
                src, Window(col0, row0, col1 - col0, row1 - row0), out_dir / f"p{i}" / src_tif.name
            )
            written += 1
    return written


#: Acquisition dates every date-windowed test references: ``2025-04-04`` plus the
#: 8-day window ``2025-04-06 − (7 − ts)`` for ``ts in range(8)`` → Mar 30 … Apr 06.
_TEST_DAYS: list[datetime.date] = sorted(
    {datetime.date(2025, 4, 4)}
    | {datetime.date(2025, 4, 6) - datetime.timedelta(days=7 - ts) for ts in range(8)}
)

#: MOD09GA band files the MODIS adapters open (7 sur_refl + state_1km); the other
#: ~14 per-granule layers are never read, so they are skipped.
_MODIS_KEEP = {f"MODIS_Grid_500m_2D__sur_refl_b0{i}_1.tif" for i in range(1, 8)} | {
    "MODIS_Grid_1km_2D__state_1km_1.tif"
}

#: VNP09GA band files the VIIRS adapters open (fine I1/I3 @500m + coarse M5/M7/M10/M11 @1km).
_VIIRS_KEEP = {f"VIIRS_Grid_500m_2D__SurfReflect_I{i}_1.tif" for i in (1, 3)} | {
    f"VIIRS_Grid_1km_2D__SurfReflect_M{i}_1.tif" for i in (5, 7, 10, 11)
}


def _acq_tag(prefix: str, day: datetime.date) -> str:
    return f"{prefix}.A{day.year:04d}{day.timetuple().tm_yday:03d}."


def _copy_granule_dirs(source: str, prefix: str, keep: set[str]) -> None:
    """Crop only the test dates' granule dirs to the patch windows, only ``keep`` bands.

    Sinusoidal nearest reprojection samples only the cell footprint, so the windowed
    crop is **bit-exact** vs the full tile (verified for MODIS/VIIRS parity).
    """
    src_root = _CLIPPED / source
    out_root = _OUT_ROOT[source] / source
    tags = {_acq_tag(prefix, d) for d in _TEST_DAYS}
    copied_dirs = 0
    copied_files = 0
    for granule in sorted(src_root.glob(f"{prefix}.A*")):
        if not granule.is_dir() or not any(granule.name.startswith(t) for t in tags):
            continue
        dest = out_root / granule.name
        for name in keep:
            src_f = granule / name
            if src_f.exists():
                _crop_union(src_f, dest / name, margin_px=_CROP_MARGIN_PX)
                copied_files += 1
        copied_dirs += 1
    size_mb = sum(p.stat().st_size for p in out_root.rglob("*")) / 1e6 if out_root.exists() else 0
    logger.info(
        f"{source}_fixture_built",
        out=str(out_root),
        dirs=copied_dirs,
        files=copied_files,
        mb=round(size_mb, 1),
    )


def populate_modis() -> None:
    _copy_granule_dirs("modis", "MOD09GA", _MODIS_KEEP)


def populate_viirs() -> None:
    _copy_granule_dirs("viirs", "VNP09GA", _VIIRS_KEEP)


#: The S3 acquisition date the OLCI tests cover (test_s3_adapter ``_COVERED_DAY``).
_S3_TEST_DATE = "20250401"


def populate_sentinel3() -> None:
    """Copy only the S3 OLCI products for the single covered test date (kept full → archive)."""
    src_root = _CLIPPED / "sentinel3"
    out_root = _OUT_ROOT["sentinel3"] / "sentinel3"
    out_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for zip_path in sorted(src_root.glob(f"S3?_OL_1_EFR____{_S3_TEST_DATE}T*.zip")):
        shutil.copy2(zip_path, out_root / zip_path.name)
        n += 1
    size_mb = sum(p.stat().st_size for p in out_root.rglob("*")) / 1e6 if out_root.exists() else 0
    logger.info("sentinel3_archive_populated", out=str(out_root), products=n, mb=round(size_mb, 1))


def populate_dem() -> None:
    """Crop DEM tiles to the patch windows with a wide halo (bilinear warp needs it)."""
    src_root = _CLIPPED / "dem"
    out_root = _OUT_ROOT["dem"] / "dem"
    n = 0
    for tile in sorted(src_root.rglob("*_DEM.tif")):
        n += _crop_per_patch(tile, out_root, margin_px=_DEM_CROP_MARGIN_PX)
    size_mb = sum(p.stat().st_size for p in out_root.rglob("*")) / 1e6 if out_root.exists() else 0
    logger.info("dem_fixture_built", out=str(out_root), crops=n, mb=round(size_mb, 1))


_POPULATORS = {
    "dem": populate_dem,
    "modis": populate_modis,
    "viirs": populate_viirs,
    "sentinel3": populate_sentinel3,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="+",
        choices=sorted(_POPULATORS),
        default=sorted(_POPULATORS),
        help="Which archive-tier sources to populate.",
    )
    args = parser.parse_args()
    for source in args.source:
        _POPULATORS[source]()


if __name__ == "__main__":
    main()
