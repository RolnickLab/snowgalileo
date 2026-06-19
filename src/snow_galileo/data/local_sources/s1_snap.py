"""Sentinel-1 SNAP preprocessing cache builder (TASK-014, offline step).

This stage processes Sentinel-1 from the **raw** granules: SNAP needs the full-swath
orbit/calibration/noise metadata and a dense scene. It is the **only** S1 processing —
S1 is no longer clipped at all; this per-granule SNAP cache is the single S1 product
**both** the cube ``S1Adapter`` and the viewer's S1 quicklook read (the raw-DN clip was
removed — see ``PLAN-S1-PERGRANULE-SNAP.md``). Producing the GEE ``COPERNICUS/S1_GRD``
value domain needs the full ESA SNAP Sentinel-1 Toolbox chain (the same engine GEE
uses), which is heavy (orbit + SRTM download + terrain correction).

To keep the :class:`~snow_galileo.data.local_sources.s1.S1Adapter` ``fetch`` a **pure raster**
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

    uv run python -m snow_galileo.data.local_sources.s1_snap

Sentinel-1C note: the archive is all ``S1C_*`` (the satellite launched Dec 2024).
SNAP reads S1C natively; ``xarray-sentinel``/``sarsen`` do **not** (the ``s1[ab]``
regex bug), which is why this chain uses SNAP. See the
``xarray-sentinel-s1c-regex-bug`` memory note.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import rasterio
import structlog
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import transform as shapely_transform

from snow_galileo.data.local_sources.clip.footprints import sentinel_safe_footprint

logger = structlog.get_logger(__name__)

#: Default ESA SNAP ``gpt`` location (NOT ``/usr/bin/snap``, which is snapd).
_HOME = os.getenv("HOME")
_DEFAULT_GPT = Path(f"{_HOME}/esa-snap/bin/gpt")

#: The production SNAP graph (beside this module).
_DEFAULT_GRAPH = Path(__file__).with_name("s1_grd_graph.xml")

#: Cache-tif filename prefix.
_CACHE_PREFIX = "s1_grd_"

#: Minimum acceptable ratio of a SNAP output's georeferenced footprint to the expected
#: AOI∩granule-footprint area. An interrupted/failed SNAP run can exit 0 yet write a tiny
#: truncated raster (observed: 3 S1C segments published as ~4×5.5 km / ~0.4% slivers while
#: the cache-hit logic then refused to overwrite them). This guard rejects an output whose
#: extent is implausibly small for the AOI overlap, so the truncated tif is never published
#: as a valid cache entry (PLAN-S1-PERGRANULE-SNAP). 0.25 is deliberately generous — a
#: legitimately partial swath still clears it; only a gross truncation trips it.
_MIN_EXTENT_RATIO: float = 0.25

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
        raise FileNotFoundError(f"ESA SNAP gpt not found at {gpt}; install SNAP or pass gpt=.")
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


def _utm_area_m2(geom_4326: Polygon, *, utm_epsg: int = 32611) -> float:
    """Area (m²) of a lon/lat polygon reprojected to the cube's UTM zone."""
    if geom_4326.is_empty:
        return 0.0
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True).transform
    return shapely_transform(to_utm, geom_4326).area


def _output_extent_is_plausible(
    *,
    out_tif: Path,
    aoi_4326: Polygon,
    footprint_4326: Polygon | None,
    region_wkt: str,
    min_ratio: float = _MIN_EXTENT_RATIO,
) -> bool:
    """True if the SNAP output's georeferenced extent is large enough to be trusted.

    Guards against the **silent-truncation** failure: an interrupted SNAP run can exit 0
    yet write a tiny raster (observed: ~0.4% slivers published as valid cache hits). The
    expected covered area is the AOI-bbox-with-margin (the ``geoRegion``) intersected with
    the granule footprint, in the cube's UTM metres. The output's own georeferenced bbox
    area is compared to it; below ``min_ratio`` → reject (truncated).

    A footprint we cannot read (``None``) falls back to the AOI region area alone — still
    catches a gross sliver. The check is cheap (raster bounds only, no pixel read).

    Args:
        out_tif: The SNAP output GeoTIFF to validate.
        aoi_4326: AOI polygon (EPSG:4326).
        footprint_4326: Granule footprint (EPSG:4326), or ``None`` if unreadable.
        region_wkt: The ``geoRegion`` WKT the Subset used (AOI bbox + margin).
        min_ratio: Reject below this output/expected area ratio.

    Returns:
        ``True`` if plausible (publish), ``False`` if implausibly small (reject).
    """
    from shapely import wkt as shapely_wkt

    region = shapely_wkt.loads(region_wkt)
    expected_geom = region.intersection(footprint_4326) if footprint_4326 is not None else region
    expected_m2 = _utm_area_m2(expected_geom)
    if expected_m2 <= 0.0:
        # Nothing expected (no overlap) — don't second-guess SNAP here; the no-overlap
        # gate already ran upstream. Treat as plausible.
        return True

    try:
        with rasterio.open(out_tif) as ds:
            b = ds.bounds
            out_m2 = abs((b.right - b.left) * (b.top - b.bottom))  # output already in UTM m
    except rasterio.errors.RasterioIOError:
        # An unreadable output is certainly not a valid full-extent raster — reject it
        # (same disposition as a truncated sliver). Defends against a 0-byte / corrupt
        # write that exited 0.
        logger.warning("s1_snap_output_unreadable", out=out_tif.name)
        return False

    ratio = out_m2 / expected_m2
    if ratio < min_ratio:
        logger.warning(
            "s1_snap_output_truncated",
            out=out_tif.name,
            output_km2=round(out_m2 / 1e6, 1),
            expected_km2=round(expected_m2 / 1e6, 1),
            ratio=round(ratio, 4),
            min_ratio=min_ratio,
        )
        return False
    return True


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
    :class:`~snow_galileo.data.local_sources.s1.S1Adapter` later windows this AOI-wide tif to
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

    # SNAP writes to a temp path, then we atomically rename it into place only on success.
    # A crash / SIGKILL / disk-full mid-write therefore leaves a stray temp file the cache
    # glob never sees, never a truncated final tif that the next run would mistake for a
    # valid cache hit — the build stays idempotent.
    #
    # The temp lives in a ``.partial/`` SUBDIRECTORY of cache_dir (not a sibling with a
    # ``.partial`` suffix): SNAP's BigGeoTIFF writer FORCES a ``.tif`` extension on the
    # output path, so a ``…tif.partial`` sibling gets silently rewritten to
    # ``…tif.partial.tif`` and the rename source vanishes. Keeping the temp name itself a
    # plain ``.tif`` avoids the mangling; putting it one directory down keeps it out of the
    # non-recursive ``s1_grd_*.tif`` cache glob (s1.py / viewer manifest.py). The subdir is
    # on the same filesystem as out_tif, so the rename is still atomic.
    partial_dir = cache_dir / ".partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_tif = partial_dir / out_tif.name
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

    # Extent sanity gate (BEFORE publishing): a SNAP run can exit 0 yet emit a truncated
    # sliver (an earlier interrupted build published 3 such ~0.4% tifs, which the cache-hit
    # logic then refused to overwrite). Reject an output too small for the AOI overlap so it
    # is never published as a valid cache entry — leave it as a discarded partial; the next
    # build retries it (the cache stays "missing" rather than "wrongly satisfied").
    if not _output_extent_is_plausible(
        out_tif=partial_tif,
        aoi_4326=aoi_4326,
        footprint_4326=footprint,
        region_wkt=_aoi_region_wkt(aoi_4326),
    ):
        partial_tif.unlink(missing_ok=True)
        logger.warning("s1_snap_output_rejected_truncated", granule=granule_zip.stem)
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
        n_granules=len(granules),
        cache_dir=str(cache_dir),
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


def cli() -> None:
    """Single-command Typer CLI for the offline S1 SNAP cache build.

    Runs SNAP **once per raw granule** over the AOI bbox (the geoRegion Subset is applied
    after Terrain-Correction). For the operator-facing entry point with preflight checks,
    use ``scripts/developer_scripts/bow_valley_inference_local/build_bow_valley_s1_cache.py``.

    ``typer`` is imported here, not at module scope: this module is read on the S1-adapter
    hot path (``s1.py``), and the CLI only runs under ``python -m`` — so the import stays
    out of every library import.
    """
    import typer

    from snow_galileo.data.local_sources.clip.settings import load_aoi_polygon
    from snow_galileo.data.local_sources.paths import LocalPaths

    paths = LocalPaths()
    app = typer.Typer(add_completion=False, help="Build the Sentinel-1 per-granule SNAP cache.")

    @app.command()
    def build(
        archive_root: Path = typer.Option(
            paths.raw_root / "sentinel1",
            "--archive-root",
            help="Raw S1 archive (holds full-swath S1*_IW_GRDH_*.zip).",
        ),
        cache_dir: Path = typer.Option(
            paths.clipped_root / "sentinel1_snap",
            "--cache-dir",
            help="Output directory for the s1_grd_*.tif cache.",
        ),
        aoi: Path = typer.Option(paths.aoi_path, "--aoi", help="AOI polygon (EPSG:4326 GeoJSON)."),
        gpt: Path = typer.Option(_DEFAULT_GPT, "--gpt"),
        graph: Path = typer.Option(_DEFAULT_GRAPH, "--graph"),
        overwrite: bool = typer.Option(False, "--overwrite"),
    ) -> None:
        """Build the per-granule AOI-wide SNAP dB+angle cache from raw."""
        cached = build_s1_cache(
            archive_root=archive_root,
            aoi_4326=load_aoi_polygon(aoi),
            cache_dir=cache_dir,
            gpt=gpt,
            graph=graph,
            overwrite=overwrite,
        )
        for path in cached:
            typer.echo(str(path))

    app()


if __name__ == "__main__":
    cli()
