r"""Build the simple committed test fixtures (EPSG:4326 WorldCover crop + ERA5 copy).

Writes tiny committed excerpts under ``tests/fixtures/clipped/<source>/`` that the
adapters discover exactly like the full archive, so their tests run in CI. This
script covers only the sources with simple 4326 / copy logic:

- **worldcover** — windowed 4326 crop per reference patch.
- **era5** — copied whole (already AOI-native, ~1 MB).

DEM, MODIS, VIIRS need per-source windowing (``populate_test_archive.py``); S2 needs
slim SAFE repackaging (``build_slim_s2_safe.py``); S3 stays in the gitignored archive
tier. See ``tests/fixtures/clipped/README.md``.

Run from the repo root (needs the real clipped archive present locally)::

    uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/build_test_fixtures.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import rasterio
import structlog
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

logger = structlog.get_logger(__name__)

#: Real clipped archive root (this script's input).
_ARCHIVE = Path("data/clipped_bow_valley_selection_raw")

#: Committed slim-fixture root (this script's output).
_FIXTURE = Path("tests/fixtures/clipped")

#: Committed GEE reference patches whose footprints define the windows to crop.
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

#: Degrees of margin added around each patch footprint (EPSG:4326 sources).
_MARGIN_DEG: float = 0.01


def _patch_bboxes_4326() -> list[tuple[float, float, float, float]]:
    """Return each reference patch's footprint as a 4326 ``(lon0, lat0, lon1, lat1)``."""
    boxes: list[tuple[float, float, float, float]] = []
    for patch in sorted(_REF_DIR.glob("PR_*.tif")):
        with rasterio.open(patch) as src:
            lo0, la0, lo1, la1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        boxes.append((lo0 - _MARGIN_DEG, la0 - _MARGIN_DEG, lo1 + _MARGIN_DEG, la1 + _MARGIN_DEG))
    return boxes


def _crop_tile_4326(
    src_tif: Path,
    out_dir: Path,
    boxes: list[tuple[float, float, float, float]],
) -> int:
    """Write one tiny crop of ``src_tif`` (EPSG:4326) per intersecting patch box.

    Each patch footprint becomes its own small fixture tile (``<stem>__p{i}.tif``)
    so scattered patches never get bridged into one huge bounding window. The
    adapter rglobs all of them and mosaics, exactly as with the full archive.
    Returns the number of crops written.
    """
    written = 0
    with rasterio.open(src_tif) as src:
        tb = src.bounds
        profile_base = src.profile.copy()
        for i, (lo0, la0, lo1, la1) in enumerate(boxes):
            # Skip boxes that do not intersect this tile.
            if lo1 < tb.left or lo0 > tb.right or la1 < tb.bottom or la0 > tb.top:
                continue
            win = from_bounds(lo0, la0, lo1, la1, transform=src.transform)
            row0 = max(0, int(win.row_off) - 1)
            col0 = max(0, int(win.col_off) - 1)
            row1 = min(src.height, int(win.row_off + win.height) + 1)
            col1 = min(src.width, int(win.col_off + win.width) + 1)
            if row1 <= row0 or col1 <= col0:
                continue
            window = Window(col0, row0, col1 - col0, row1 - row0)

            data = src.read(window=window)
            profile = profile_base.copy()
            profile.update(
                height=window.height,
                width=window.width,
                transform=src.window_transform(window),
                driver="GTiff",
                compress="deflate",
            )

            # Keep the original filename (adapters glob on its suffix, e.g.
            # ``*_DEM.tif`` / ``*_Map.tif``); disambiguate per-patch via a subdir.
            out_tif = out_dir / f"p{i}" / src_tif.name
            out_tif.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(data)
            logger.info(
                "fixture_tile_written",
                out=str(out_tif.relative_to(_FIXTURE)),
                shape=(int(window.height), int(window.width)),
                kb=round(out_tif.stat().st_size / 1024, 1),
            )
            written += 1
    return written


def build_worldcover() -> None:
    """Crop the ESA WorldCover tiles (EPSG:4326) to the patch footprints."""
    boxes = _patch_bboxes_4326()
    src_root = _ARCHIVE / "worldcover"
    written = sum(
        _crop_tile_4326(tile, _FIXTURE / "worldcover", boxes)
        for tile in sorted(src_root.rglob("*_Map.tif"))
    )
    logger.info("worldcover_fixtures_done", tiles_written=written)


def build_era5() -> None:
    """Copy the ERA5 NetCDFs whole — already AOI-native (~1 MB) and parity-passing."""
    src_root = _ARCHIVE / "era5"
    out_root = _FIXTURE / "era5"
    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(src_root, out_root)
    n = len(list(out_root.rglob("*.nc")))
    size_mb = sum(p.stat().st_size for p in out_root.rglob("*")) / 1e6
    logger.info("era5_fixtures_done", files=n, mb=round(size_mb, 1))


# DEM and the science-band / S2 sources are built by ``populate_test_archive.py`` and
# ``build_slim_s2_safe.py`` — they need windowing logic this 4326-crop script lacks.
_BUILDERS = {
    "worldcover": build_worldcover,
    "era5": build_era5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs="+",
        choices=sorted(_BUILDERS),
        default=sorted(_BUILDERS),
        help="Which source fixtures to (re)build.",
    )
    args = parser.parse_args()
    for source in args.source:
        _BUILDERS[source]()


if __name__ == "__main__":
    main()
