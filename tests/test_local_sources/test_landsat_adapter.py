"""Landsat 8/9 TOA adapter tests (TASK-012, AC-12/13/15b/16).

The Landsat adapter replaces the ``B2_landsat..B7_landsat`` placeholders with
top-of-atmosphere reflectance on the cell grid, reading clipped L1TP ``.tar`` scenes
(raw DN + ``_MTL.json``). A companion :class:`LandsatCloudAdapter` emits ``QA_PIXEL``.

Two test layers:

- **Synthetic unit tests** (no archive): build tiny in-memory scene tars to exercise the
  L9→L8 fallback, the same-(tile, date) coalesce (valid-pixel union, latest-proc wins),
  and zone-agnostic reprojection (a 32611 same-zone AND a 32612 cross-zone source onto
  the 32611 cell grid). These never skip.
- **Real-archive parity** (skips if archive/fixtures absent): the three reference patches
  TASK-012b unblocked — ``PR_20250406`` (t3), ``PR_20250414`` (t2), ``PR_20250510`` (t1)
  — must match the GEE ``B4_landsat`` band **bit-exactly** (GEE upsamples the 30 m source
  to the 10 m cell as constant blocks → nearest, same as MODIS; bilinear would smear).
"""

from __future__ import annotations

import datetime
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell
from src.data.local_sources.landsat import LandsatAdapter, LandsatCloudAdapter

# --------------------------------------------------------------------------- #
# Real-archive parity wiring
# --------------------------------------------------------------------------- #
_L9_ROOT = Path("data/clipped_bow_valley_selection_raw/landsat9")
_L8_ROOT = Path("data/clipped_bow_valley_selection_raw/landsat8")
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

_DYNAMIC_PER_TS = 38
_OFF_B4_LANDSAT = 11  # B4_landsat offset inside the 38-band dynamic block

#: patch-date → (filename glob, timestep with Landsat, acquisition date).
_PARITY_CASES = {
    "PR_20250406": ("PR_20250406_*.tif", 3, datetime.date(2025, 4, 2)),
    "PR_20250414": ("PR_20250414_*.tif", 2, datetime.date(2025, 4, 9)),
    "PR_20250510": ("PR_20250510_*.tif", 1, datetime.date(2025, 5, 4)),
}


def _cell_from_patch(patch: Path) -> GridCell:
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=0,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


# --------------------------------------------------------------------------- #
# Synthetic in-memory scene builders
# --------------------------------------------------------------------------- #
def _write_band_tif(path: Path, dn: np.ndarray, transform: Affine, crs: str) -> None:
    with rasterio.open(
        path, "w", driver="GTiff", height=dn.shape[0], width=dn.shape[1],
        count=1, dtype="uint16", crs=crs, transform=transform,
    ) as ds:
        ds.write(dn.astype(np.uint16), 1)


def _make_scene_tar(
    *,
    tar_path: Path,
    stem: str,
    dn_by_band: dict[int, np.ndarray],
    transform: Affine,
    crs: str,
    mult: float = 2.0e-5,
    add: float = -0.1,
    sun_elevation: float = 41.0,
    tmp: Path,
) -> None:
    """Build a minimal clipped-scene tar: ``_MTL.json`` + ``_B{n}.TIF`` members."""
    contents = {f"FILE_NAME_BAND_{n}": f"{stem}_B{n}.TIF" for n in dn_by_band}
    mtl = {
        "LANDSAT_METADATA_FILE": {
            "PRODUCT_CONTENTS": contents,
            "IMAGE_ATTRIBUTES": {"SUN_ELEVATION": sun_elevation},
            "LEVEL1_RADIOMETRIC_RESCALING": {
                k: v
                for n in dn_by_band
                for k, v in {
                    f"REFLECTANCE_MULT_BAND_{n}": mult,
                    f"REFLECTANCE_ADD_BAND_{n}": add,
                }.items()
            },
        }
    }
    mtl_path = tmp / f"{stem}_MTL.json"
    mtl_path.write_text(json.dumps(mtl))
    with tarfile.open(tar_path, "w") as tar:
        tar.add(mtl_path, arcname=f"{stem}_MTL.json")
        for n, dn in dn_by_band.items():
            bp = tmp / f"{stem}_B{n}.TIF"
            _write_band_tif(bp, dn, transform, crs)
            tar.add(bp, arcname=f"{stem}_B{n}.TIF")


def _toa(dn: float, mult: float = 2.0e-5, add: float = -0.1, sun: float = 41.0) -> float:
    return (mult * dn + add) / np.sin(np.deg2rad(sun))


_CELL_PX = 20  # 20 px × 10 m = 200 m cell


@pytest.fixture()
def synthetic_cell() -> GridCell:
    """A 20×20 px cell (200 m, 10 m px) at a Bow Valley UTM-11N origin."""
    return GridCell.from_utm_bounds(
        cell_id=1, min_x=563000.0, min_y=5653000.0, max_x=563200.0, max_y=5653200.0,
        px=_CELL_PX,
    )


def _scene_transform_crs(cell: GridCell, crs: str) -> tuple[Affine, str]:
    """A 24×24 source grid (10 m px) that fully covers the cell in ``crs``."""
    if crs == cell.crs:
        return from_origin(cell.transform.c - 20, cell.transform.f + 20, 10.0, 10.0), crs
    # Cross-zone: place the source in 32612 over the same ground area.
    from pyproj import Transformer

    t = Transformer.from_crs(cell.crs, crs, always_xy=True)
    x0, y0 = t.transform(cell.transform.c, cell.transform.f)
    return from_origin(x0 - 20, y0 + 20, 10.0, 10.0), crs


# --------------------------------------------------------------------------- #
# Contract / smoke
# --------------------------------------------------------------------------- #
def test_bands_out_and_kind() -> None:
    """``bands_out`` is B2_landsat..B7_landsat in order; high tier; no native fill (AC-12)."""
    adapter = LandsatAdapter(landsat9_root=_L9_ROOT, landsat8_root=_L8_ROOT)
    assert adapter.bands_out == [f"B{n}_landsat" for n in range(2, 8)]
    assert adapter.spatial_kind == "high"
    assert adapter.native_fill is None


def test_missing_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """A day with no scene → all-``-9999`` of declared shape (AC-13)."""
    adapter = LandsatAdapter(landsat9_root=tmp_path / "l9", landsat8_root=tmp_path / "l8")
    (tmp_path / "l9").mkdir()
    (tmp_path / "l8").mkdir()
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    assert out.shape == (6, *synthetic_cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


def test_none_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """``day=None`` → placeholder (the adapter is date-driven)."""
    adapter = LandsatAdapter(landsat9_root=tmp_path / "l9", landsat8_root=tmp_path / "l8")
    out = adapter.fetch(synthetic_cell, day=None)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


# --------------------------------------------------------------------------- #
# AC-16: L9→L8 fallback
# --------------------------------------------------------------------------- #
def _build_single_scene(
    root: Path, sat: str, pathrow: str, acq: str, proc: str, dn: int, cell: GridCell,
    tmp: Path, crs: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    crs = crs or cell.crs
    transform, scrs = _scene_transform_crs(cell, crs)
    h = w = 24
    stem = f"{sat}_L1TP_{pathrow}_{acq}_{proc}_02_T1"
    _make_scene_tar(
        tar_path=root / f"{stem}.tar", stem=stem,
        dn_by_band={n: np.full((h, w), dn, dtype=np.uint16) for n in range(2, 8)},
        transform=transform, crs=scrs, tmp=tmp,
    )


def test_l9_preferred_over_l8(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """When both L9 and L8 cover the day, L9 wins (AC-16)."""
    l9, l8 = tmp_path / "l9", tmp_path / "l8"
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    _build_single_scene(l9, "LC09", "043024", "20250402", "20250402", dn=20000, cell=synthetic_cell, tmp=tmp)
    _build_single_scene(l8, "LC08", "043024", "20250402", "20250410", dn=8000, cell=synthetic_cell, tmp=tmp)
    adapter = LandsatAdapter(landsat9_root=l9, landsat8_root=l8)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    # B4_landsat (index 2) should equal the L9 DN's TOA, not L8's.
    valid = out[2][out[2] != NO_DATA_VALUE]
    assert valid.size > 0
    np.testing.assert_allclose(np.median(valid), _toa(20000), atol=1e-4)


def test_l8_fallback_when_l9_absent(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """When L9 has no scene for the day, L8 is used (AC-16)."""
    l9, l8 = tmp_path / "l9", tmp_path / "l8"
    l9.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    _build_single_scene(l8, "LC08", "043024", "20250402", "20250410", dn=8000, cell=synthetic_cell, tmp=tmp)
    adapter = LandsatAdapter(landsat9_root=l9, landsat8_root=l8)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    valid = out[2][out[2] != NO_DATA_VALUE]
    assert valid.size > 0
    np.testing.assert_allclose(np.median(valid), _toa(8000), atol=1e-4)


# --------------------------------------------------------------------------- #
# AC-15b: same-(tile, date) coalesce
# --------------------------------------------------------------------------- #
def test_coalesce_complementary_masks(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Two same-(tile,date) products with complementary nodata → zero ``-9999`` where
    either is valid; surviving value = latest-processing winner (AC-15b).
    """
    l9 = tmp_path / "l9"
    l9.mkdir()
    l8 = tmp_path / "l8"
    l8.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    h = w = 24
    transform, scrs = _scene_transform_crs(synthetic_cell, synthetic_cell.crs)

    # Product A (later proc): left half valid (DN 20000), right half DN 0 (fill).
    a = np.full((h, w), 20000, dtype=np.uint16)
    a[:, w // 2:] = 0
    # Product B (earlier proc): right half valid (DN 8000), left half DN 0 (fill).
    b = np.full((h, w), 8000, dtype=np.uint16)
    b[:, : w // 2] = 0
    for stem, arr, proc in (
        ("LC09_L1TP_043024_20250402_20250410_02_T1", a, "20250410"),
        ("LC09_L1TP_043024_20250402_20250404_02_T1", b, "20250404"),
    ):
        s = f"LC09_L1TP_043024_20250402_{proc}_02_T1"
        _make_scene_tar(
            tar_path=l9 / f"{s}.tar", stem=s,
            dn_by_band={n: arr for n in range(2, 8)}, transform=transform, crs=scrs, tmp=tmp,
        )

    adapter = LandsatAdapter(landsat9_root=l9, landsat8_root=l8)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    b4 = out[2]
    # Coalesced: no -9999 anywhere (every pixel valid in exactly one product).
    assert not (b4 == NO_DATA_VALUE).any(), "coalesce left holes where one product was valid"
    # Left half = product A's TOA (it is valid there); right half = product B's TOA.
    left = b4[:, : synthetic_cell.shape[1] // 2]
    right = b4[:, synthetic_cell.shape[1] // 2:]
    np.testing.assert_allclose(np.median(left), _toa(20000), atol=1e-4)
    np.testing.assert_allclose(np.median(right), _toa(8000), atol=1e-4)


def test_coalesce_latest_proc_wins_on_overlap(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Where both products are valid, the latest-processing-time value wins (AC-15b)."""
    l9 = tmp_path / "l9"
    l9.mkdir()
    l8 = tmp_path / "l8"
    l8.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    h = w = 24
    transform, scrs = _scene_transform_crs(synthetic_cell, synthetic_cell.crs)
    later = np.full((h, w), 20000, dtype=np.uint16)
    earlier = np.full((h, w), 8000, dtype=np.uint16)
    for arr, proc in ((later, "20250410"), (earlier, "20250404")):
        s = f"LC09_L1TP_043024_20250402_{proc}_02_T1"
        _make_scene_tar(
            tar_path=l9 / f"{s}.tar", stem=s,
            dn_by_band={n: arr for n in range(2, 8)}, transform=transform, crs=scrs, tmp=tmp,
        )
    adapter = LandsatAdapter(landsat9_root=l9, landsat8_root=l8)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    valid = out[2][out[2] != NO_DATA_VALUE]
    np.testing.assert_allclose(np.median(valid), _toa(20000), atol=1e-4)


# --------------------------------------------------------------------------- #
# Zone-agnostic reproject (mixed UTM): same-zone 32611 AND cross-zone 32612
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("crs", ["EPSG:32611", "EPSG:32612"])
def test_reproject_zone_agnostic(synthetic_cell: GridCell, tmp_path: Path, crs: str) -> None:
    """A constant-DN source in 32611 (same zone) OR 32612 (cross-zone) reprojects to the
    32611 cell grid with the same TOA value (AC-16 — no hardcoded zone).
    """
    l9 = tmp_path / "l9"
    l8 = tmp_path / "l8"
    l8.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    _build_single_scene(
        l9, "LC09", "043024", "20250402", "20250402", dn=20000, cell=synthetic_cell, tmp=tmp, crs=crs
    )
    adapter = LandsatAdapter(landsat9_root=l9, landsat8_root=l8)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    assert out.shape == (6, *synthetic_cell.shape)
    valid = out[2][out[2] != NO_DATA_VALUE]
    assert valid.size > 0, f"no valid pixels reprojecting from {crs}"
    np.testing.assert_allclose(np.median(valid), _toa(20000), atol=1e-4)


# --------------------------------------------------------------------------- #
# Real-archive parity against the GEE reference patches (TASK-012b unblocked)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def real_adapter() -> LandsatAdapter:
    if not (any(_L9_ROOT.glob("*.tar")) or any(_L8_ROOT.glob("*.tar"))):
        pytest.skip("No clipped Landsat archive")
    return LandsatAdapter(landsat9_root=_L9_ROOT, landsat8_root=_L8_ROOT)


@pytest.mark.parametrize("patch_key", list(_PARITY_CASES))
def test_parity_b4_landsat_against_gee(real_adapter: LandsatAdapter, patch_key: str) -> None:
    """B4_landsat matches the GEE reference patch at its Landsat timestep (AC-12)."""
    glob, ts, acq = _PARITY_CASES[patch_key]
    patches = sorted(_REF_DIR.glob(glob))
    if not patches:
        pytest.skip(f"No reference patch for {patch_key}")
    patch = patches[0]
    cell = _cell_from_patch(patch)

    out = real_adapter.fetch(cell, day=acq)
    with rasterio.open(patch) as ds:
        ref = ds.read(_DYNAMIC_PER_TS * ts + _OFF_B4_LANDSAT + 1)

    valid = (ref != NO_DATA_VALUE) & (ref > 1e-7) & (out[2] != NO_DATA_VALUE)
    assert valid.sum() > 0.5 * ref.size, f"{patch_key}: <50% overlapping valid B4 pixels"
    # Bit-exact under nearest (GEE upsamples 30 m → 10 m as constant blocks). The DN→TOA
    # formula (M·DN+A)/sin(elev) reproduces GEE's value domain; nearest matches its
    # resampling. Bilinear would smear ~0.003-0.012 over cloud/snow edges (PARITY check).
    med = float(np.median(np.abs(out[2][valid] - ref[valid])))
    assert med == 0.0, f"{patch_key}: B4_landsat not bit-exact vs GEE (median {med:.6f})"


# --------------------------------------------------------------------------- #
# QA_PIXEL cloud adapter (subtask 3)
# --------------------------------------------------------------------------- #
def test_cloud_adapter_qa_pixel(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """The cloud adapter emits a single categorical ``QA_PIXEL`` band, NN-resampled,
    with the L9→L8 fallback (AC-12).
    """
    l9 = tmp_path / "l9"
    l9.mkdir()
    l8 = tmp_path / "l8"
    l8.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    transform, scrs = _scene_transform_crs(synthetic_cell, synthetic_cell.crs)
    h = w = 24
    stem = "LC09_L1TP_043024_20250402_20250402_02_T1"
    # A plausible QA_PIXEL bit-flag field (clear=21824, cloud=22280) as a checkerboard.
    qa = np.full((h, w), 21824, dtype=np.uint16)
    qa[::2, ::2] = 22280
    mtl_path = tmp / f"{stem}_MTL.json"
    mtl_path.write_text(json.dumps({"LANDSAT_METADATA_FILE": {"PRODUCT_CONTENTS": {}}}))
    qa_path = tmp / f"{stem}_QA_PIXEL.TIF"
    _write_band_tif(qa_path, qa, transform, scrs)
    with tarfile.open(l9 / f"{stem}.tar", "w") as tar:
        tar.add(mtl_path, arcname=f"{stem}_MTL.json")
        tar.add(qa_path, arcname=f"{stem}_QA_PIXEL.TIF")

    cloud = LandsatCloudAdapter(landsat9_root=l9, landsat8_root=l8)
    assert cloud.bands_out == ["QA_PIXEL"]
    assert cloud.spatial_kind == "time"
    out = cloud.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    assert out.shape == (1, *synthetic_cell.shape)
    # Nearest keeps the discrete bit-flag codes — no interpolated intermediate values.
    assert set(np.unique(out[0])).issubset({21824.0, 22280.0, float(NO_DATA_VALUE)})


def test_cloud_adapter_missing_day(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """No scene → single-band all-``-9999`` placeholder (AC-13)."""
    l9 = tmp_path / "l9"
    l9.mkdir()
    l8 = tmp_path / "l8"
    l8.mkdir()
    cloud = LandsatCloudAdapter(landsat9_root=l9, landsat8_root=l8)
    out = cloud.fetch(synthetic_cell, day=datetime.date(2025, 4, 2))
    assert out.shape == (1, *synthetic_cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))
