"""Sentinel-1 SNAP preprocessing cache builder (TASK-014, offline step).

This stage processes Sentinel-1 from the **raw** granules: SNAP needs the full-swath
orbit/calibration/noise metadata and a dense scene. It is the **only** S1 processing —
S1 is no longer clipped at all; this per-granule SNAP cache is the single S1 product
**both** the cube ``S1Adapter`` and the viewer's S1 quicklook read (the raw-DN clip was
removed — see ``PLAN-S1-PERGRANULE-SNAP.md``). Producing the GEE ``COPERNICUS/S1_GRD``
value domain needs the full ESA SNAP Sentinel-1 Toolbox chain (the same engine GEE
uses), which is heavy (orbit + SRTM download + terrain correction).

To keep the :class:`~src.data.local_sources.s1.S1Adapter` ``fetch`` a **pure raster**
operation (no SNAP dependency, unit-testable, fast), SNAP runs here **once per raw
granule** over the AOI, producing one 3-band dB+angle GeoTIFF cached on disk:

    raw SAFE .zip ──(gpt, this module)──▶ s1_grd_<granule_stem>.tif
    (full swath)     production graph     (Sigma0_VV, Sigma0_VH linear→dB-in-adapter,
                                            incidenceAngleFromEllipsoid in degrees;
                                            EPSG:32611, 10 m, AOI bbox)

The adapter then reads, coalesces, mosaics, and **windows** that AOI-wide tif to each
cell (``reproject_to_cell``) like every other scene source. The SNAP ``Subset`` to the
AOI bbox is applied **after** Terrain-Correction (map geometry — a clean raster crop,
no radar-geometry back-projection, no "Empty region!" NPE; S1 GRD best practice). It
bounds compute: terrain-correcting + writing the full ~250 km IW swath at 10 m
overflows SNAP's classic-GeoTIFF writer.

This is an **idempotent, offline** step: run it once before exporting cubes;
re-running skips granules whose cache tif already exists. One AOI-wide tif per granule
(not per cell) — the adapter windows it to any cell, so mode A and mode B share it.

    uv run python -m src.data.local_sources.s1_snap

Sentinel-1C note: the archive is all ``S1C_*`` (the satellite launched Dec 2024).
SNAP reads S1C natively; ``xarray-sentinel``/``sarsen`` do **not** (the ``s1[ab]``
regex bug), which is why this chain uses SNAP. See the
``xarray-sentinel-s1c-regex-bug`` memory note.
"""

from __future__ import annotations

import datetime
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import structlog
from shapely.geometry import Polygon, box

from src.data.local_sources.clip.footprints import sentinel_safe_footprint

logger = structlog.get_logger(__name__)

#: Default ESA SNAP ``gpt`` location (NOT ``/usr/bin/snap``, which is snapd).
_DEFAULT_GPT = Path("/home/dev/esa-snap/bin/gpt")

#: The production SNAP graph (beside this module).
_DEFAULT_GRAPH = Path(__file__).with_name("s1_grd_graph.xml")

#: Cache-tif filename prefix.
_CACHE_PREFIX = "s1_grd_"

#: Margin (degrees) added around the AOI bbox so the post-TC crop keeps context at the
#: AOI edges (no edge artefacts when the adapter later windows per cell).
_AOI_MARGIN_DEG: float = 0.02

#: Acquisition ``YYYYMMDD`` in a raw S1 granule stem (``S1C_IW_GRDH_1SDV_<acq>T...``).
_GRANULE_ACQ_PATTERN = re.compile(r"_(\d{8})T\d{6}_")


def _granule_acq_date(granule_zip: Path) -> datetime.date | None:
    """Parse the acquisition date from a raw S1 granule ``.zip`` stem.

    Returns ``None`` if the stem carries no parseable ``_<YYYYMMDD>T<HHMMSS>_`` token.
    """
    match = _GRANULE_ACQ_PATTERN.search(granule_zip.stem)
    if match is None:
        return None
    return datetime.datetime.strptime(match.group(1), "%Y%m%d").date()


def cache_tif_name(granule_stem: str) -> str:
    """Return the cache-tif filename for one **raw** granule (AOI-wide, per-granule).

    The cache is keyed by **granule** (not granule×cell): SNAP runs once per raw
    granule over the whole AOI bbox, with the ``geoRegion`` Subset applied **after**
    Terrain-Correction (map geometry — a clean raster crop, no radar-geometry
    back-projection, no "Empty region!" NPE). The :class:`S1Adapter` then windows
    that one AOI-wide tif to each cell via ``reproject_to_cell``. This replaces the
    old per-cell keying (thousands of fragile pre-TC radar-geometry subsets).

    Args:
        granule_stem: The granule ``.zip`` stem (``S1C_IW_GRDH_...``).

    Returns:
        ``s1_grd_<granule_stem>.tif``.
    """
    return f"{_CACHE_PREFIX}{granule_stem}.tif"


def _aoi_region_wkt(aoi_4326: Polygon, *, margin_deg: float = _AOI_MARGIN_DEG) -> str:
    """Return the SNAP ``geoRegion`` WKT for the AOI: its bbox (4326) + a small margin.

    The Subset runs **after** Terrain-Correction, so this geoRegion crops the projected
    EPSG:32611 raster to the AOI bbox in map geometry. A whole-AOI bbox is dense in the
    swath (no empty regions), and the crop bounds compute (the full ~250 km swath at 10 m
    overflows SNAP's writer). The :class:`S1Adapter` later windows the AOI-wide tif per
    cell, so per-cell bounding here is unnecessary.

    Args:
        aoi_4326: The AOI polygon in EPSG:4326.
        margin_deg: Degrees of padding added around the AOI bbox.

    Returns:
        A ``POLYGON((...))`` WKT string in lon/lat.
    """
    lon0, lat0, lon1, lat1 = aoi_4326.bounds
    m = margin_deg
    lo0, la0 = lon0 - m, lat0 - m
    lo1, la1 = lon1 + m, lat1 + m
    return f"POLYGON(({lo0} {la0},{lo1} {la0},{lo1} {la1},{lo0} {la1},{lo0} {la0}))"


def _aoi_intersects_footprint(aoi_4326: Polygon, footprint_4326: Polygon | None) -> bool:
    """True if the AOI bbox overlaps the granule footprint (or footprint unknown).

    Gates out granules whose swath does not touch the AOI at all (no point running
    SNAP). ``None``/invalid footprint → attempt the subset anyway (SNAP skips if truly
    empty). Compares bbox-to-footprint, matching the clip stage's footprint gate.
    """
    if footprint_4326 is None or not footprint_4326.is_valid:
        return True  # can't tell — attempt the subset (SNAP will skip if truly empty)
    lon0, lat0, lon1, lat1 = aoi_4326.bounds
    return box(lon0, lat0, lon1, lat1).intersects(footprint_4326)


def _run_snap_chain(
    *,
    safe_manifest: Path,
    region_wkt: str,
    out_tif: Path,
    gpt: Path,
    graph: Path,
) -> Path:
    """Run the S1_GRD graph via ``gpt`` for one (granule, region); return the tif.

    Args:
        safe_manifest: The extracted ``.SAFE/manifest.safe`` to read.
        region_wkt: lon/lat WKT polygon to subset to (the ``${region}`` param).
        out_tif: Output GeoTIFF path (the ``${output}`` param).
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.

    Returns:
        ``out_tif``.

    Raises:
        FileNotFoundError: If ``gpt`` is not available.
        subprocess.CalledProcessError: If the SNAP chain fails.
    """
    if not gpt.exists():
        raise FileNotFoundError(
            f"ESA SNAP gpt not found at {gpt}; install SNAP or pass gpt=."
        )
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(gpt),
        str(graph),
        f"-Pinput={safe_manifest}",
        f"-Pregion={region_wkt}",
        f"-Poutput={out_tif}",
        "-c",
        "2G",
        "-q",
        "4",
    ]
    logger.info("s1_snap_chain_start", granule=safe_manifest.parent.name, out=out_tif.name)
    subprocess.run(cmd, check=True)
    return out_tif


def build_granule_cache(
    *,
    granule_zip: Path,
    aoi_4326: Polygon,
    cache_dir: Path,
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _DEFAULT_GRAPH,
    overwrite: bool = False,
) -> list[Path]:
    """Produce one AOI-wide dB+angle cache tif for one **raw** S1 granule ``.zip``.

    Runs the SNAP graph **once** over the whole AOI bbox (the ``geoRegion`` Subset is
    applied after Terrain-Correction, in map geometry), writing
    ``{cache_dir}/s1_grd_<stem>.tif``. Idempotent per granule. The
    :class:`~src.data.local_sources.s1.S1Adapter` later windows this AOI-wide tif to
    each cell, so no per-cell SNAP run is needed.

    Args:
        granule_zip: The **raw** IW GRD SAFE ``.zip`` (full swath — SNAP needs the
            un-clipped orbit/calibration/noise metadata and the dense full scene).
        aoi_4326: The AOI polygon (EPSG:4326); the post-TC Subset crops to its bbox.
        cache_dir: Directory the cache tif is written to.
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.
        overwrite: Re-run SNAP even where the cache tif already exists.

    Returns:
        ``[cache tif]`` for this granule, or ``[]`` if its swath does not overlap the
        AOI (footprint-gated out) or SNAP produced no output (empty crop).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    footprint = sentinel_safe_footprint(granule_zip, "manifest.safe")
    if not _aoi_intersects_footprint(aoi_4326, footprint):
        logger.info("s1_snap_no_overlap", granule=granule_zip.stem)
        return []

    out_tif = cache_dir / cache_tif_name(granule_zip.stem)
    if out_tif.exists() and not overwrite:
        logger.info("s1_snap_cache_hit", granule=granule_zip.stem)
        return [out_tif]

    # SNAP writes to a temp path INSIDE cache_dir, then we atomically rename into place
    # only on success. A crash / SIGKILL / disk-full mid-write therefore leaves a stray
    # ``.partial`` (not matched by the s1_grd_*.tif cache glob), never a truncated final
    # tif that the next run would mistake for a valid cache hit — the build stays safely
    # idempotent. (The temp must be on the same filesystem as out_tif for an atomic
    # rename, hence cache_dir, not the system tmp.)
    partial_tif = out_tif.with_suffix(out_tif.suffix + ".partial")
    partial_tif.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(granule_zip) as zf:  # read-mode: raw archive never mutated
            zf.extractall(tmp_dir)
        safe_dirs = list(tmp_dir.glob("*.SAFE"))
        if not safe_dirs:
            raise FileNotFoundError(f"No .SAFE directory inside {granule_zip}.")
        manifest = safe_dirs[0] / "manifest.safe"
        try:
            _run_snap_chain(
                safe_manifest=manifest,
                region_wkt=_aoi_region_wkt(aoi_4326),
                out_tif=partial_tif,
                gpt=gpt,
                graph=graph,
            )
        except subprocess.CalledProcessError as exc:
            # The AOI bbox overlaps the granule footprint, but the projected scene has
            # no pixels in the crop — SNAP's post-TC Subset raises "Empty region!".
            # That means S1 genuinely cannot be produced for this granule over the AOI;
            # skip it (clean up the partial) rather than aborting the whole offline
            # build. A systemic no-gpt failure raises in _run_snap_chain.
            partial_tif.unlink(missing_ok=True)
            logger.warning(
                "s1_snap_chain_failed", granule=granule_zip.stem, returncode=exc.returncode
            )
            return []
    # Success — publish atomically (overwrites a prior tif under --overwrite).
    partial_tif.replace(out_tif)
    return [out_tif]


def build_s1_cache(
    *,
    archive_root: Path,
    aoi_4326: Polygon,
    cache_dir: Path,
    gpt: Path = _DEFAULT_GPT,
    graph: Path = _DEFAULT_GRAPH,
    overwrite: bool = False,
) -> list[Path]:
    """Build the per-granule AOI-wide SNAP cache for every **raw** S1 granule.

    The once/offline step analogous to the clip stage. Idempotent — already-cached
    per-granule tifs are skipped (unless ``overwrite``). One AOI-wide tif per granule
    (the :class:`S1Adapter` windows it per cell), not per (granule, cell).

    Args:
        archive_root: The **raw** S1 archive (holds ``S1*_IW_GRDH_*.zip`` full swaths).
        aoi_4326: The AOI polygon (EPSG:4326); each granule's post-TC Subset crops to it.
        cache_dir: Output directory for the ``s1_grd_*.tif`` cache.
        gpt: Path to the ESA SNAP ``gpt`` executable.
        graph: Path to the production SNAP graph XML.
        overwrite: Re-run SNAP even where a cache tif already exists.

    Returns:
        All cache tif paths produced, in granule-name order.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    granules = sorted(archive_root.glob("S1*_IW_GRDH_*.zip"))
    logger.info(
        "s1_snap_build_start",
        n_granules=len(granules), cache_dir=str(cache_dir),
    )
    cached: list[Path] = []
    for granule_zip in granules:
        cached.extend(
            build_granule_cache(
                granule_zip=granule_zip,
                aoi_4326=aoi_4326,
                cache_dir=cache_dir,
                gpt=gpt,
                graph=graph,
                overwrite=overwrite,
            )
        )
    logger.info("s1_snap_build_done", n_cached=len(cached))
    return cached


class S1CacheUnavailableError(RuntimeError):
    """A needed per-granule S1 cache tif is missing (the offline build has not been run).

    Raised by :func:`ensure_s1_cache` so cube export fails loudly instead of silently
    assembling an all-``-9999`` S1 block (the historical silent-dropout bug). The fix is
    to run the offline build (``build_bow_valley_s1_cache.py``), not to build inline.
    """


def ensure_s1_cache(
    *,
    raw_archive_root: Path,
    aoi_4326: Polygon,
    cache_dir: Path,
    window_days: list[datetime.date],
) -> None:
    """Verify the per-granule S1 SNAP cache covers the export window — fail loud if not.

    A **verification-only** pre-flight (it does NOT run SNAP): building the cache is the
    offline driver's job (raw → SNAP, once — see :func:`build_s1_cache` and
    ``build_bow_valley_s1_cache.py``). Keeping the heavyweight SNAP/orbit/SRTM step out
    of the cube-export path is the whole point of the per-granule offline stage; the
    exporter only checks that the offline build has been run.

    Computes the **needed** per-granule cache tifs — raw granules whose acquisition day
    is in ``window_days`` and whose footprint intersects the AOI — and:

    * returns if every needed tif already exists;
    * **raises** :class:`S1CacheUnavailableError` listing the missing granules otherwise,
      so the cube is never silently assembled with an all-``-9999`` S1 block.

    The check is footprint-aware: a window-day granule whose swath does not overlap the
    AOI is **not** "missing" (it genuinely has no data here). A window with no covering
    granule at all needs nothing and does not trip the guard — genuinely-absent S1 is
    acceptable; only a *built-offline-but-not-yet-built* cache trips it.

    Args:
        raw_archive_root: The **raw** S1 archive (holds ``S1*_IW_GRDH_*.zip`` full swaths)
            — the same archive the offline build reads, so the needed-granule set matches.
        aoi_4326: The AOI polygon (EPSG:4326) the cache tifs cover.
        cache_dir: The SNAP cache directory the :class:`S1Adapter` reads.
        window_days: The export window's days (the adapter reads one timestep per day).

    Raises:
        S1CacheUnavailableError: If a needed per-granule cache tif is missing.
    """
    window = set(window_days)
    granules = sorted(raw_archive_root.glob("S1*_IW_GRDH_*.zip"))
    in_window = [g for g in granules if _granule_acq_date(g) in window]

    missing: list[str] = []
    for granule_zip in in_window:
        footprint = sentinel_safe_footprint(granule_zip, "manifest.safe")
        if not _aoi_intersects_footprint(aoi_4326, footprint):
            continue
        if not (cache_dir / cache_tif_name(granule_zip.stem)).exists():
            missing.append(granule_zip.stem)

    if not missing:
        return

    raise S1CacheUnavailableError(
        f"{len(missing)} S1 per-granule cache tif(s) are missing for the export window. "
        f"Build the cache first (offline, once): "
        f"`uv run python scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py`. "
        f"Missing granules: {missing}."
    )


def _main() -> None:
    """CLI: build the per-granule S1 SNAP cache (the offline preprocessing step).

    Runs SNAP **once per raw granule** over the AOI bbox (the geoRegion Subset is applied
    after Terrain-Correction). For the operator-facing entry point with preflight checks,
    use ``scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py``.
    """
    import argparse

    from src.data.local_sources.clip.settings import load_aoi_polygon
    from src.data.local_sources.paths import LocalPaths

    paths = LocalPaths()
    parser = argparse.ArgumentParser(description="Build the Sentinel-1 per-granule SNAP cache.")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=paths.raw_root / "sentinel1",
        help="Raw S1 archive (holds full-swath S1*_IW_GRDH_*.zip).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=paths.clipped_root / "sentinel1_snap",
        help="Output directory for the s1_grd_*.tif cache.",
    )
    parser.add_argument(
        "--aoi", type=Path, default=paths.aoi_path, help="AOI polygon (EPSG:4326 GeoJSON)."
    )
    parser.add_argument("--gpt", type=Path, default=_DEFAULT_GPT)
    parser.add_argument("--graph", type=Path, default=_DEFAULT_GRAPH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cached = build_s1_cache(
        archive_root=args.archive_root,
        aoi_4326=load_aoi_polygon(args.aoi),
        cache_dir=args.cache_dir,
        gpt=args.gpt,
        graph=args.graph,
        overwrite=args.overwrite,
    )
    for path in cached:
        print(path)


if __name__ == "__main__":
    _main()
