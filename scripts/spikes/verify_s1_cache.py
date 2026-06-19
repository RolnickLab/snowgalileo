r"""Verify every S1 SNAP cache output: extent, valid pixels, and per-granule coverage.

Run after a full ``process-s1`` build to confirm no granule produced a truncated sliver
(the failure that motivated the extent guard). For each cache tif: reports size, extent in
km, decimated valid-VV count, and the output/expected-overlap area ratio (the same metric
``_output_extent_is_plausible`` gates on). Flags any tif below the ratio floor or with zero
valid pixels, and lists raw granules that have NO cache tif at all.

    uv run python scripts/spikes/verify_s1_cache.py
"""

from __future__ import annotations

import numpy as np
import rasterio
import structlog
from shapely import wkt as shapely_wkt

from src.data.local_sources.clip.footprints import sentinel_safe_footprint
from src.data.local_sources.clip.settings import load_aoi_polygon
from src.data.local_sources.paths import LocalPaths
from src.data.local_sources.s1_snap import (
    _MIN_EXTENT_RATIO,
    _aoi_region_wkt,
    _utm_area_m2,
    cache_tif_name,
)

logger = structlog.get_logger(__name__)


def main() -> None:
    paths = LocalPaths()
    raw_dir = paths.raw_root / "sentinel1"
    cache_dir = paths.clipped_root / "sentinel1_snap"
    aoi = load_aoi_polygon(paths.aoi_path)
    region = shapely_wkt.loads(_aoi_region_wkt(aoi))

    raw = sorted(raw_dir.glob("S1*_IW_GRDH_*.zip"))
    print(f"=== S1 cache verification ({len(raw)} raw granules) ===")
    print(
        f"{'granule':50s} {'size':>13s} {'extent_km':>14s} {'validVV':>9s} {'ratio':>7s}  status"
    )

    problems: list[str] = []
    missing: list[str] = []
    for zip_path in raw:
        tif = cache_dir / cache_tif_name(zip_path.stem)
        short = zip_path.stem.split("_")[-1]  # uid
        if not tif.exists():
            missing.append(zip_path.stem)
            print(f"{short:50s} {'--':>13s} {'--':>14s} {'--':>9s} {'--':>7s}  MISSING")
            continue

        footprint = sentinel_safe_footprint(zip_path, "manifest.safe")
        expected = region.intersection(footprint) if footprint is not None else region
        expected_m2 = _utm_area_m2(expected)

        with rasterio.open(tif) as ds:
            b = ds.bounds
            w_km = (b.right - b.left) / 1000
            h_km = (b.top - b.bottom) / 1000
            out_m2 = abs((b.right - b.left) * (b.top - b.bottom))
            arr = ds.read(1, out_shape=(1, min(512, ds.height), min(512, ds.width)))
            valid = int(np.sum(arr > 0))
            size_mb = tif.stat().st_size / 1e6

        ratio = out_m2 / expected_m2 if expected_m2 > 0 else float("nan")
        bad = (ratio == ratio and ratio < _MIN_EXTENT_RATIO) or valid == 0
        status = "TRUNCATED" if bad else "ok"
        if bad:
            problems.append(zip_path.stem)
        print(
            f"{short:50s} {size_mb:10.0f}MB {w_km:6.0f}x{h_km:<5.0f}km "
            f"{valid:9d} {ratio:7.2f}  {status}"
        )

    print()
    print(
        f"verified: {len(raw) - len(missing)} present, {len(missing)} missing, "
        f"{len(problems)} truncated"
    )
    if missing:
        print("MISSING:", *(f"\n  {m}" for m in missing))
    if problems:
        print("TRUNCATED:", *(f"\n  {p}" for p in problems))
    if not missing and not problems:
        print("ALL GOOD — every granule has a full-extent, non-empty cache tif.")


if __name__ == "__main__":
    main()
