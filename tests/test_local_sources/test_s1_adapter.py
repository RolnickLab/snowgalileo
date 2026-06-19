"""Sentinel-1 GRD adapter tests (TASK-014, AC-12/13/14).

The S1 adapter replaces the ``[VV, VH, angle]`` placeholders (head of the HIGH
group) with GEE ``COPERNICUS/S1_GRD`` values: calibrated, terrain-corrected σ⁰ in
**dB** for VV/VH + the ellipsoid **incidence angle** (degrees), on the cell grid,
with the project edge mask (pixels ``< -30.0`` dB → ``-9999``).

Unlike S2/Landsat, the clip stage does **not** preprocess S1 — the heavy ESA SNAP
chain runs once per granule into a cached 3-band dB+angle GeoTIFF
(:mod:`src.data.local_sources.s1_snap`). The adapter ``fetch`` is then a pure
raster read of that cache, so these unit tests build synthetic **post-SNAP**
GeoTIFFs (no SNAP, no archive — they never skip).

Two test layers:

- **Synthetic** (no SNAP/archive): tiny in-memory dB+angle cache tifs exercise the
  edge mask (VV/VH only; angle never masked), the same-date coalesce (valid-pixel
  union, latest winner), band order, and the missing-day placeholder.
- **Real-archive parity** (skips unless the SNAP cache is built): for each covered
  date, VV/VH median ``|Δ|`` ≤ 1.0 dB (TASK-005 tolerance) and angle ≈ the GEE
  reference, on the matching reference patch.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell
from src.data.local_sources.s1 import (
    S1Adapter,
    _parse_granule,
)
from src.data.local_sources.s1_snap import cache_tif_name

_S1_CACHE = Path("data/clipped_bow_valley_selection_raw/sentinel1_snap")
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

_DYNAMIC_PER_TS = 38
#: S1 band offsets inside the 38-band dynamic block: VV=0, VH=1, angle=2.
_OFF_VV, _OFF_VH, _OFF_ANGLE = 0, 1, 2


def _db_to_linear(db: float) -> float:
    """Inverse of the adapter's 10·log10: a dB value as a linear σ⁰ for the cache."""
    return float(10.0 ** (db / 10.0))


# --------------------------------------------------------------------------- #
# Synthetic post-SNAP cache builder
# --------------------------------------------------------------------------- #
def _write_cache_tif(
    path: Path,
    *,
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    angle: np.ndarray,
    transform: Affine,
    crs: str,
) -> None:
    """Write a 3-band cache tif in the real SNAP format.

    The real cache stores **linear σ⁰** (not dB) in the SNAP band order
    **VH, VV, angle**, with no band descriptions (BigTIFF drops them). The inputs
    here are given in dB for readability and converted to linear; a non-finite /
    masked dB input is written as ``0`` (SNAP's no-data / σ⁰≤0, which the adapter
    treats as invalid). The adapter re-derives dB via ``10·log10``.
    """

    def _lin(db: np.ndarray) -> np.ndarray:
        out = np.where(np.isfinite(db), 10.0 ** (db / 10.0), 0.0)
        # A masked/below-floor sentinel (we use -inf) → 0 linear (invalid for the adapter).
        return out.astype(np.float32)

    # Band order VH, VV, angle (SNAP's order); linear σ⁰ for the backscatter bands.
    bands = np.stack([_lin(vh_db), _lin(vv_db), angle.astype(np.float32)], axis=0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=3,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(bands)


_CELL_PX = 20


@pytest.fixture()
def synthetic_cell() -> GridCell:
    return GridCell.from_utm_bounds(
        cell_id=1,
        min_x=563000.0,
        min_y=5653000.0,
        max_x=563200.0,
        max_y=5653200.0,
        px=_CELL_PX,
    )


def _src_transform(cell: GridCell) -> Affine:
    """A 24×24 source grid (10 m px) fully covering the cell, same CRS."""
    return from_origin(cell.transform.c - 20, cell.transform.f + 20, 10.0, 10.0)


def _granule_stem(acq: str, *, end: str = "013749", uid: str = "88AD") -> str:
    """A realistic S1C cache-granule stem for ``acq`` (YYYYMMDD)."""
    return f"S1C_IW_GRDH_1SDV_{acq}T013724_{acq}T{end}_001664_002BB2_{uid}"


def _build_cache_granule(
    cache_root: Path,
    *,
    acq: str,
    cell: GridCell,
    vv: float = -8.0,
    vh: float = -14.0,
    angle: float = 43.6,
    end: str = "013749",
    uid: str = "88AD",
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Path:
    """Write one synthetic cache tif under ``cache_root`` for acquisition ``acq``.

    ``vv``/``vh`` and the ``arrays`` VV/VH planes are in **dB** (converted to the
    cache's linear σ⁰ internally); ``angle`` is in degrees.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    h = w = 24
    if arrays is None:
        arrays = (
            np.full((h, w), vv, np.float32),
            np.full((h, w), vh, np.float32),
            np.full((h, w), angle, np.float32),
        )
    stem = _granule_stem(acq, end=end, uid=uid)
    path = cache_root / cache_tif_name(stem)
    _write_cache_tif(
        path,
        vv_db=arrays[0],
        vh_db=arrays[1],
        angle=arrays[2],
        transform=_src_transform(cell),
        crs=cell.crs,
    )
    return path


# --------------------------------------------------------------------------- #
# Contract / smoke
# --------------------------------------------------------------------------- #
def test_bands_out_and_kind() -> None:
    """``bands_out`` is [VV,VH,angle]; high tier; no native fill (AC-12)."""
    adapter = S1Adapter(cache_root=_S1_CACHE)
    assert adapter.bands_out == ["VV", "VH", "angle"]
    assert adapter.spatial_kind == "high"
    assert adapter.native_fill is None


def test_granule_name_parses_s1c() -> None:
    """The granule regex parses the per-granule S1C cache name (archive is S1C, not S1A/B)."""
    stem = _granule_stem("20250406")
    info = _parse_granule(Path(cache_tif_name(stem)))
    assert info is not None
    assert info.acq == datetime.date(2025, 4, 6)
    assert info.uid == "88AD"


def test_missing_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """A day with no granule → all-``-9999`` (3, H, W) (AC-13, the common S1 case)."""
    (tmp_path / "s1").mkdir()
    adapter = S1Adapter(cache_root=tmp_path / "s1")
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 7))
    assert out.shape == (3, *synthetic_cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


def test_none_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """``day=None`` → placeholder."""
    adapter = S1Adapter(cache_root=tmp_path)
    out = adapter.fetch(synthetic_cell, day=None)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


# --------------------------------------------------------------------------- #
# AC-12: band order + domain
# --------------------------------------------------------------------------- #
def test_band_order_and_domain(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Output is [VV,VH,angle] in order, on the cell grid; dB VV/VH + degree angle."""
    cache = tmp_path / "s1"
    _build_cache_granule(cache, acq="20250406", cell=synthetic_cell, vv=-8.0, vh=-14.0, angle=43.6)
    adapter = S1Adapter(cache_root=cache)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 6))
    assert out.shape == (3, *synthetic_cell.shape)
    # atol=1e-2: VV/VH round-trip dB→linear(float32 cache)→dB; angle is stored direct.
    np.testing.assert_allclose(np.median(out[_OFF_VV]), -8.0, atol=1e-2)
    np.testing.assert_allclose(np.median(out[_OFF_VH]), -14.0, atol=1e-2)
    np.testing.assert_allclose(np.median(out[_OFF_ANGLE]), 43.6, atol=1e-3)


# --------------------------------------------------------------------------- #
# AC-14: edge mask < -30 dB on VV/VH only; angle never masked
# --------------------------------------------------------------------------- #
def test_edge_mask_below_minus30(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Pixels ``< -30`` dB in VV/VH become ``-9999``; the angle band is untouched."""
    cache = tmp_path / "s1"
    h = w = 24
    vv = np.full((h, w), -8.0, np.float32)
    vv[:, : w // 2] = -45.0  # below the edge floor → must be masked
    vh = np.full((h, w), -14.0, np.float32)
    angle = np.full((h, w), 43.6, np.float32)
    _build_cache_granule(cache, acq="20250406", cell=synthetic_cell, arrays=(vv, vh, angle))
    out = S1Adapter(cache_root=cache).fetch(synthetic_cell, day=datetime.date(2025, 4, 6))

    half = synthetic_cell.shape[1] // 2
    left_vv = out[_OFF_VV][:, :half]
    right_vv = out[_OFF_VV][:, half:]
    assert (left_vv == NO_DATA_VALUE).mean() > 0.9, "below-floor VV not masked"
    np.testing.assert_allclose(np.median(right_vv), -8.0, atol=1e-2)
    # Angle is geometry, NOT a backscatter band — never edge-masked.
    assert not (out[_OFF_ANGLE] == NO_DATA_VALUE).any(), "angle was wrongly edge-masked"
    np.testing.assert_allclose(np.median(out[_OFF_ANGLE]), 43.6, atol=1e-3)


def test_edge_mask_keeps_just_above_floor(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """A value just above the -30 dB floor survives (mask is ``< -30``, not aggressive)."""
    cache = tmp_path / "s1"
    h = w = 24
    vv = np.full((h, w), -29.5, np.float32)  # above the floor → must survive
    _build_cache_granule(
        cache,
        acq="20250406",
        cell=synthetic_cell,
        arrays=(vv, np.full((h, w), -14.0, np.float32), np.full((h, w), 43.6, np.float32)),
    )
    out = S1Adapter(cache_root=cache).fetch(synthetic_cell, day=datetime.date(2025, 4, 6))
    assert not (out[_OFF_VV] == NO_DATA_VALUE).any(), "-29.5 dB should survive the < -30 mask"
    np.testing.assert_allclose(np.median(out[_OFF_VV]), -29.5, atol=1e-2)


# --------------------------------------------------------------------------- #
# AC-15b-style: same-date coalesce across granules (valid-pixel union)
# --------------------------------------------------------------------------- #
def test_coalesce_complementary_masks(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Two same-date granules with complementary edge masks → no holes; union of valids."""
    cache = tmp_path / "s1"
    h = w = 24
    # granule A (later uid): left valid, right below-floor
    a_vv = np.full((h, w), -6.0, np.float32)
    a_vv[:, w // 2 :] = -45.0
    # granule B (earlier uid): right valid, left below-floor
    b_vv = np.full((h, w), -10.0, np.float32)
    b_vv[:, : w // 2] = -45.0
    angle = np.full((h, w), 43.6, np.float32)
    vh = np.full((h, w), -14.0, np.float32)
    # uid sorts: "FFFF" (later) vs "0000" (earlier) — coalesce orders latest-first.
    _build_cache_granule(
        cache, acq="20250406", cell=synthetic_cell, uid="FFFF", arrays=(a_vv, vh, angle)
    )
    _build_cache_granule(
        cache,
        acq="20250406",
        cell=synthetic_cell,
        uid="0000",
        end="013814",
        arrays=(b_vv, vh, angle),
    )
    out = S1Adapter(cache_root=cache).fetch(synthetic_cell, day=datetime.date(2025, 4, 6))

    vv = out[_OFF_VV]
    assert not (vv == NO_DATA_VALUE).any(), "coalesce left holes where one granule was valid"
    half = synthetic_cell.shape[1] // 2
    np.testing.assert_allclose(np.median(vv[:, :half]), -6.0, atol=1e-3)  # A wins left
    np.testing.assert_allclose(np.median(vv[:, half:]), -10.0, atol=1e-3)  # B fills right


def test_same_date_granules_with_different_footprints(
    synthetic_cell: GridCell, tmp_path: Path
) -> None:
    """Same-date granules with **different extents** mosaic without a shape error.

    Regression for the real-archive case (verified on 2025-04-06): one pass emits two
    per-granule SNAP outputs that are the same zone (EPSG:32611) + 10 m res but cover
    **different footprints** (adjacent sub-swath segments — e.g. a 393×550 segment
    alongside a 14466×9637 one). They are the *mosaic* case, not the coalesce case; the
    prior code forced both through ``coalesce_tile`` (one-shared-grid assumption), which
    raised ``IndexError: boolean index did not match indexed array`` on the shape
    mismatch and aborted the whole sweep. One granule covers the cell; the other is a
    far-away segment that does not. The fetch must succeed and return the covering
    granule's values on the cell.
    """
    cache = tmp_path / "s1"
    cache.mkdir(parents=True)
    h = w = 24
    angle = np.full((h, w), 43.6, np.float32)
    vh = np.full((h, w), -14.0, np.float32)

    # Granule A (latest uid) covers the cell, on the 24×24 grid centred on it.
    _write_cache_tif(
        cache / cache_tif_name(_granule_stem("20250406", uid="FFFF")),
        vv_db=np.full((h, w), -7.0, np.float32),
        vh_db=vh,
        angle=angle,
        transform=_src_transform(synthetic_cell),
        crs=synthetic_cell.crs,
    )
    # Granule B (earlier uid) is a far-away swath segment with a *different* size and
    # origin (a few hundred km north) — disjoint from the cell. This is what crashed.
    _write_cache_tif(
        cache / cache_tif_name(_granule_stem("20250406", end="013814", uid="0000")),
        vv_db=np.full((550, 393), -3.0, np.float32),
        vh_db=np.full((550, 393), -14.0, np.float32),
        angle=np.full((550, 393), 43.6, np.float32),
        transform=from_origin(528488.0, 5800300.0, 10.0, 10.0),
        crs=synthetic_cell.crs,
    )

    out = S1Adapter(cache_root=cache).fetch(synthetic_cell, day=datetime.date(2025, 4, 6))
    assert out.shape == (3, *synthetic_cell.shape)
    # Only granule A covers the cell, so the covering granule's value fills it.
    assert not (out[_OFF_VV] == NO_DATA_VALUE).any(), "covering granule should fill the cell"
    np.testing.assert_allclose(np.median(out[_OFF_VV]), -7.0, atol=1e-2)


# --------------------------------------------------------------------------- #
# Real-archive parity — skips unless the SNAP cache is built
# --------------------------------------------------------------------------- #
_S1_ARCHIVE = Path("data/bow_valley_selection_raw/sentinel1")

#: patch-key → (timestep with S1, acquisition date, a stable parity cell_id). Each
#: patch carries S1 on its window-end timestep (ts7 = the patch/prediction date), all
#: present in the raw archive. Each parity case has a distinct acquisition date → distinct
#: granules → distinct per-granule cache tifs, so the patch caches never collide.
_PARITY_CASES = {
    "PR_20250406": (7, datetime.date(2025, 4, 6), 90406),
    "PR_20250423": (7, datetime.date(2025, 4, 23), 90423),
    "PR_20250519": (7, datetime.date(2025, 5, 19), 90519),
}

#: Patches the parity gate proves the pipeline on. ALL THREE now reproduce GEE
#: ``COPERNICUS/S1_GRD`` within tolerance under the per-granule (raw, post-TC-Subset)
#: pipeline — verified 2026-06-11 on a SNAP-capable run:
#:   PR_20250519: VV 0.401 / VH 0.421 dB, angle 0.240°  (the long-proven case)
#:   PR_20250423: VV 0.427 / VH 0.468 dB, angle 0.343°  (was "Empty region!" under the
#:                old pre-TC radar-geometry Subset — the post-TC ordering fixed it)
#:   PR_20250406: VV 0.579 / VH 0.509 dB, angle 0.784°  (was a ~10 dB "single-scene
#:                anomaly" under the old CLIPPED + pre-TC path; the anomaly was OUR
#:                processing, not GEE's data — processing the RAW full swath with the
#:                post-TC Subset reproduces GEE faithfully).
#: All comfortably under the 1.0 dB / 1.0° tolerances, so they assert as real passes
#: (no xfail). A regression in the SNAP chain or adapter now fails the gate loudly.
_PARITY_PROVEN = {"PR_20250406", "PR_20250423", "PR_20250519"}

#: No known non-adapter parity failures remain — the per-granule refactor resolved both
#: prior xfails (see _PARITY_PROVEN). Kept as an (empty) extension point.
_PARITY_XFAIL: dict[str, str] = {}

#: TASK-005 tolerance: median |Δ| ≤ 1.0 dB for VV/VH (SAR is speckly → median).
_S1_DRIFT_TOLERANCE_DB = 1.0
#: Ellipsoid incidence angle varies only with range; allow a small absolute drift.
_ANGLE_TOLERANCE_DEG = 1.0


def _cell_from_patch(patch: Path, cell_id: int) -> GridCell:
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=cell_id,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


def _ensure_patch_cache(cell: GridCell, acq: datetime.date) -> S1Adapter:
    """Build (idempotently) the per-granule SNAP cache for ``acq``'s granules; skip w/o SNAP.

    Drives the real pipeline end-to-end: SNAP-process each **raw** granule acquired on
    ``acq`` over the patch-cell's 4326 bbox (the post-TC Subset region), into ``_S1_CACHE``.
    The adapter then windows the AOI-wide tif to the cell. Skips cleanly if the archive or
    ESA SNAP is absent (CI), so the parity check only runs where it can.
    """
    import datetime as _dt

    from pyproj import Transformer
    from shapely.geometry import box as _box

    from src.data.local_sources.s1_snap import _DEFAULT_GPT, build_granule_cache

    if not _DEFAULT_GPT.exists():
        pytest.skip(f"ESA SNAP gpt not found at {_DEFAULT_GPT}; cannot build S1 cache.")
    if not _S1_ARCHIVE.exists():
        pytest.skip("No raw S1 archive.")

    granules = [
        z
        for z in sorted(_S1_ARCHIVE.glob("S1*_IW_GRDH_*.zip"))
        if _dt.datetime.strptime(z.stem.split("_")[4][:8], "%Y%m%d").date() == acq
    ]
    if not granules:
        pytest.skip(f"No S1 granule in the archive for {acq}.")

    # The patch cell's 4326 bbox is the post-TC Subset region for this parity build.
    tr = Transformer.from_crs(cell.crs, "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = cell.polygon.bounds
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)
    cell_bbox_4326 = _box(min(lon0, lon1), min(lat0, lat1), max(lon0, lon1), max(lat0, lat1))

    _S1_CACHE.mkdir(parents=True, exist_ok=True)
    for granule_zip in granules:
        build_granule_cache(granule_zip=granule_zip, aoi_4326=cell_bbox_4326, cache_dir=_S1_CACHE)
    return S1Adapter(cache_root=_S1_CACHE)


def _parity_params() -> list[object]:
    """One param per patch. Any keys in ``_PARITY_XFAIL`` carry an ``xfail`` mark; it is
    currently empty (all three patches pass under the per-granule pipeline).
    """
    params: list[object] = []
    for key in _PARITY_CASES:
        marks: tuple[pytest.MarkDecorator, ...] = ()
        if key in _PARITY_XFAIL:
            marks = (pytest.mark.xfail(reason=_PARITY_XFAIL[key], strict=False),)
        params.append(pytest.param(key, marks=marks, id=key))
    return params


@pytest.mark.parametrize("patch_key", _parity_params())
def test_parity_vv_vh_angle_against_gee(patch_key: str) -> None:
    """VV/VH within 1.0 dB and angle within 1° of the GEE reference patch (AC-14).

    All three patches pass under the per-granule (raw, post-TC-Subset) pipeline — see
    ``_PARITY_PROVEN`` for the measured per-patch margins. The post-TC ordering fixed the
    two cases that previously xfailed (PR_20250423 "Empty region!" and PR_20250406, whose
    ~10 dB "anomaly" was our old clipped+pre-TC processing, not GEE's data).
    """
    ts, acq, cell_id = _PARITY_CASES[patch_key]
    patches = sorted(_REF_DIR.glob(f"{patch_key}_*.tif"))
    if not patches:
        pytest.skip(f"No reference patch for {patch_key}")
    patch = patches[0]
    cell = _cell_from_patch(patch, cell_id)

    adapter = _ensure_patch_cache(cell, acq)
    out = adapter.fetch(cell, day=acq)
    with rasterio.open(patch) as ds:
        ref_vv = ds.read(_DYNAMIC_PER_TS * ts + _OFF_VV + 1).astype(np.float64)
        ref_vh = ds.read(_DYNAMIC_PER_TS * ts + _OFF_VH + 1).astype(np.float64)
        ref_ang = ds.read(_DYNAMIC_PER_TS * ts + _OFF_ANGLE + 1).astype(np.float64)

    def _median_abs(got: np.ndarray, ref: np.ndarray) -> float:
        valid = (ref != NO_DATA_VALUE) & (ref != 0) & (got != NO_DATA_VALUE)
        assert valid.sum() > 50, f"{patch_key}: too few valid pixels ({valid.sum()})"
        return float(np.median(np.abs(got[valid] - ref[valid])))

    assert _median_abs(out[_OFF_VV], ref_vv) <= _S1_DRIFT_TOLERANCE_DB
    assert _median_abs(out[_OFF_VH], ref_vh) <= _S1_DRIFT_TOLERANCE_DB
    assert _median_abs(out[_OFF_ANGLE], ref_ang) <= _ANGLE_TOLERANCE_DEG
