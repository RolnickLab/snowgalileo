"""Sentinel-2 L1C adapter tests (TASK-013, AC-12/13/15/15b).

The S2 adapter replaces the ``[B2,B3,B4,B8,B11,B12]`` placeholders with harmonized
reflectance (GEE ``S2_HARMONIZED`` domain: raw L1C DN − 1000 for baseline ≥ N0400) on the
cell grid, reading clipped L1C SAFE ``.zip`` granules.

Three test layers:

- **Synthetic** (no archive): tiny in-memory SAFE zips exercise the −1000 DN offset, the
  same-(tile,date) coalesce (valid-pixel union, latest-proc winner), and the missing-day
  placeholder. Never skip.
- **Coverage validation** (the TASK-012b lesson, user-requested): assert each reference
  patch has ≥1 covered S2 acquisition date in the clipped archive; the test reports the
  missing dates explicitly (TASK-013b follow-up) instead of passing blindly.
- **Real-patch parity** (skips if archive/fixtures absent): for each patch's covered date,
  ``B4`` matches the GEE reference **bit-exactly** (signed median 0 under nearest +
  −1000 DN; GEE upsamples the 30/20/10 m source to the 10 m cell as constant blocks).
"""

from __future__ import annotations

import datetime
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

from src.data.config import NO_DATA_VALUE
from src.data.local_sources.base import GridCell
from src.data.local_sources.s2 import S2Adapter

_S2_ROOT = Path("data/clipped_bow_valley_selection_raw/sentinel2")
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

_DYNAMIC_PER_TS = 38

#: B4 offset inside the 38-band dynamic block (VV,VH,angle,B2,B3,B4 → 5).
_OFF_B4 = 5

#: Valid floor after harmonization (matches the adapter's ``_VALID_MIN``).
_VALID_MIN = -1.0

#: Each reference patch → the S2 acquisition dates its timesteps need (derived from the
#: patch's per-timestep dates that carry S2 data). Covered subset validated against archive.
_NEEDED_DATES = {
    "PR_20250406": ["2025-03-31", "2025-04-03", "2025-04-05"],
    "PR_20250414": ["2025-04-08", "2025-04-13"],
    "PR_20250423": ["2025-04-17", "2025-04-18", "2025-04-20", "2025-04-23"],
    "PR_20250502": ["2025-04-25", "2025-04-28", "2025-04-30"],
    "PR_20250510": ["2025-05-03", "2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
    "PR_20250519": ["2025-05-13", "2025-05-15", "2025-05-18"],
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


def _archive_acq_dates() -> set[datetime.date]:
    """Acquisition dates present in the clipped S2 archive (by granule name)."""
    dates: set[datetime.date] = set()
    for z in _S2_ROOT.glob("*.zip"):
        m = re.match(r"S2[AB]_MSIL1C_(\d{8})T", z.name)
        if m:
            dates.add(datetime.datetime.strptime(m.group(1), "%Y%m%d").date())
    return dates


# --------------------------------------------------------------------------- #
# Synthetic SAFE builder
# --------------------------------------------------------------------------- #
def _write_jp2(path: Path, dn: np.ndarray, transform: Affine, crs: str) -> None:
    # REVERSIBLE=YES → lossless (5/3 wavelet), matching real S2 L1C JP2s. Default OpenJPEG
    # is lossy and would corrupt the exact DN the harmonization/coalesce asserts on.
    with rasterio.open(
        path, "w", driver="JP2OpenJPEG", height=dn.shape[0], width=dn.shape[1],
        count=1, dtype="uint16", crs=crs, transform=transform,
        QUALITY=100, REVERSIBLE=True,
    ) as ds:
        ds.write(dn.astype(np.uint16), 1)


_MTD = (
    "<n1:Level-1C_User_Product xmlns:n1='x'><General_Info><Product_Info>"
    "<PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>"
    "</Product_Info></General_Info></n1:Level-1C_User_Product>"
)


def _make_granule_zip(
    *,
    zip_path: Path,
    stem: str,
    tile: str,
    dn_by_suffix: dict[str, np.ndarray],
    transform: Affine,
    crs: str,
    baseline: str = "05.11",
    tmp: Path,
) -> None:
    """Build a minimal SAFE zip: ``MTD_MSIL1C.xml`` + ``IMG_DATA`` JP2 bands."""
    safe = f"{stem}.SAFE"
    img_dir = f"{safe}/GRANULE/L1C_{tile}_A000000_20250101T000000/IMG_DATA"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{safe}/MTD_MSIL1C.xml", _MTD.format(baseline=baseline))
        for suffix, dn in dn_by_suffix.items():
            jp2 = tmp / f"{tile}_{suffix}.jp2"
            _write_jp2(jp2, dn, transform, crs)
            zf.write(jp2, arcname=f"{img_dir}/{tile}_20250101T000000_{suffix}.jp2")


def _harm(dn: float, baseline_ge_400: bool = True) -> float:
    return dn - 1000 if baseline_ge_400 else dn


_CELL_PX = 20


@pytest.fixture()
def synthetic_cell() -> GridCell:
    return GridCell.from_utm_bounds(
        cell_id=1, min_x=563000.0, min_y=5653000.0, max_x=563200.0, max_y=5653200.0,
        px=_CELL_PX,
    )


def _src_transform(cell: GridCell) -> Affine:
    """A 24×24 source grid (10 m px) fully covering the cell, same CRS."""
    return from_origin(cell.transform.c - 20, cell.transform.f + 20, 10.0, 10.0)


def _build_granule(
    root: Path, sat: str, tile: str, acq: str, proc: str, dn: int, cell: GridCell, tmp: Path,
    baseline: str = "05.11",
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    h = w = 24
    stem = f"{sat}_MSIL1C_{acq}T185831_N0511_R113_{tile}_{proc}T235929"
    _make_granule_zip(
        zip_path=root / f"{stem}.zip", stem=stem, tile=tile,
        dn_by_suffix={s: np.full((h, w), dn, dtype=np.uint16) for s in _S2_BANDS_SUFFIX},
        transform=_src_transform(cell), crs=cell.crs, baseline=baseline, tmp=tmp,
    )


_S2_BANDS_SUFFIX = ["B02", "B03", "B04", "B08", "B11", "B12"]


# --------------------------------------------------------------------------- #
# Contract / smoke
# --------------------------------------------------------------------------- #
def test_bands_out_and_kind() -> None:
    """``bands_out`` is [B2,B3,B4,B8,B11,B12]; high tier; no native fill (AC-12)."""
    adapter = S2Adapter(archive_root=_S2_ROOT)
    assert adapter.bands_out == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert adapter.spatial_kind == "high"
    assert adapter.native_fill is None


def test_missing_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """A day with no granule → all-``-9999`` of declared shape (AC-13)."""
    (tmp_path / "s2").mkdir()
    adapter = S2Adapter(archive_root=tmp_path / "s2")
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 3))
    assert out.shape == (6, *synthetic_cell.shape)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


def test_none_day_is_all_nodata(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """``day=None`` → placeholder."""
    adapter = S2Adapter(archive_root=tmp_path)
    out = adapter.fetch(synthetic_cell, day=None)
    np.testing.assert_array_equal(out, np.full_like(out, NO_DATA_VALUE))


# --------------------------------------------------------------------------- #
# AC-15: −1000 DN harmonization
# --------------------------------------------------------------------------- #
def test_harmonization_offset_applied(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """N0511 granule → raw DN − 1000 (AC-15)."""
    s2 = tmp_path / "s2"
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    _build_granule(s2, "S2A", "T11UNS", "20250403", "20250403", dn=5000, cell=synthetic_cell, tmp=tmp)
    adapter = S2Adapter(archive_root=s2)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 3))
    valid = out[2][out[2] != NO_DATA_VALUE]  # B4
    assert valid.size > 0
    np.testing.assert_allclose(np.median(valid), _harm(5000), atol=1e-4)


# --------------------------------------------------------------------------- #
# AC-15b: same-(tile, date) coalesce
# --------------------------------------------------------------------------- #
def test_coalesce_complementary_masks(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Two same-(tile,date) products with complementary nodata → zero ``-9999`` where
    either is valid; surviving value = latest-processing winner (AC-15b).
    """
    s2 = tmp_path / "s2"
    s2.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    h = w = 24
    transform = _src_transform(synthetic_cell)
    a = np.full((h, w), 5000, dtype=np.uint16)
    a[:, w // 2:] = 0  # later proc: left valid
    b = np.full((h, w), 3000, dtype=np.uint16)
    b[:, : w // 2] = 0  # earlier proc: right valid
    for arr, proc in ((a, "20250404"), (b, "20250403")):
        stem = f"S2A_MSIL1C_20250403T185831_N0511_R113_T11UNS_{proc}T235929"
        _make_granule_zip(
            zip_path=s2 / f"{stem}.zip", stem=stem, tile="T11UNS",
            dn_by_suffix={s: arr for s in _S2_BANDS_SUFFIX},
            transform=transform, crs=synthetic_cell.crs, tmp=tmp,
        )
    adapter = S2Adapter(archive_root=s2)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 3))
    b4 = out[2]
    assert not (b4 == NO_DATA_VALUE).any(), "coalesce left holes where one product was valid"
    left = b4[:, : synthetic_cell.shape[1] // 2]
    right = b4[:, synthetic_cell.shape[1] // 2:]
    np.testing.assert_allclose(np.median(left), _harm(5000), atol=1e-4)
    np.testing.assert_allclose(np.median(right), _harm(3000), atol=1e-4)


def test_coalesce_latest_proc_wins_on_overlap(synthetic_cell: GridCell, tmp_path: Path) -> None:
    """Where both products are valid, the latest-processing value wins (AC-15b)."""
    s2 = tmp_path / "s2"
    s2.mkdir()
    tmp = tmp_path / "scratch"
    tmp.mkdir()
    h = w = 24
    transform = _src_transform(synthetic_cell)
    for dn, proc in ((5000, "20250404"), (3000, "20250403")):
        arr = np.full((h, w), dn, dtype=np.uint16)
        stem = f"S2A_MSIL1C_20250403T185831_N0511_R113_T11UNS_{proc}T235929"
        _make_granule_zip(
            zip_path=s2 / f"{stem}.zip", stem=stem, tile="T11UNS",
            dn_by_suffix={s: arr for s in _S2_BANDS_SUFFIX},
            transform=transform, crs=synthetic_cell.crs, tmp=tmp,
        )
    adapter = S2Adapter(archive_root=s2)
    out = adapter.fetch(synthetic_cell, day=datetime.date(2025, 4, 3))
    valid = out[2][out[2] != NO_DATA_VALUE]
    np.testing.assert_allclose(np.median(valid), _harm(5000), atol=1e-4)


# --------------------------------------------------------------------------- #
# Coverage validation (TASK-012b lesson — user-requested)
# --------------------------------------------------------------------------- #
def test_every_patch_has_a_covered_s2_date() -> None:
    """Each reference patch must have ≥1 S2 acquisition date in the clipped archive.

    Reports missing dates explicitly (TASK-013b download list) rather than passing blind.
    """
    if not any(_S2_ROOT.glob("*.zip")):
        pytest.skip("No clipped S2 archive")
    archive = _archive_acq_dates()
    uncovered: dict[str, list[str]] = {}
    no_coverage: list[str] = []
    for patch, dates in _NEEDED_DATES.items():
        missing = [d for d in dates if datetime.date.fromisoformat(d) not in archive]
        covered = [d for d in dates if datetime.date.fromisoformat(d) in archive]
        if missing:
            uncovered[patch] = missing
        if not covered:
            no_coverage.append(patch)
    # Hard requirement: every patch is validatable now (≥1 covered date).
    assert not no_coverage, f"patches with NO covered S2 date: {no_coverage}"
    # Soft signal: surface the TASK-013b download backlog without failing.
    if uncovered:
        pytest.xfail(f"TASK-013b backlog — missing S2 dates per patch: {uncovered}")


# --------------------------------------------------------------------------- #
# Real-archive parity (covered dates) — bit-exact B4 under nearest + −1000 DN
# --------------------------------------------------------------------------- #
@pytest.fixture()
def real_adapter() -> S2Adapter:
    if not any(_S2_ROOT.glob("*.zip")):
        pytest.skip("No clipped S2 archive")
    return S2Adapter(archive_root=_S2_ROOT)


#: One covered (patch, timestep, acquisition date) per patch for the parity check.
_PARITY_CASES = {
    "PR_20250406": (4, datetime.date(2025, 4, 3)),
    "PR_20250423": (7, datetime.date(2025, 4, 23)),
    "PR_20250510": (0, datetime.date(2025, 5, 3)),
}


@pytest.mark.parametrize("patch_key", list(_PARITY_CASES))
def test_parity_b4_against_gee(real_adapter: S2Adapter, patch_key: str) -> None:
    """B4 matches the GEE reference bit-exactly at a covered S2 timestep (AC-12/AC-15)."""
    ts, acq = _PARITY_CASES[patch_key]
    patches = sorted(_REF_DIR.glob(f"{patch_key}_*.tif"))
    if not patches:
        pytest.skip(f"No reference patch for {patch_key}")
    patch = patches[0]
    cell = _cell_from_patch(patch)

    out = real_adapter.fetch(cell, day=acq)
    with rasterio.open(patch) as ds:
        ref = ds.read(_DYNAMIC_PER_TS * ts + _OFF_B4 + 1)

    valid = (ref != NO_DATA_VALUE) & (ref > _VALID_MIN) & (out[2] != NO_DATA_VALUE)
    assert valid.sum() > 0.5 * ref.size, f"{patch_key}: <50% overlapping valid B4 pixels"
    # Signed median 0 == bit-exact for ≥half the pixels (nearest + −1000 reproduces GEE).
    med = float(np.median(out[2][valid] - ref[valid]))
    assert med == 0.0, f"{patch_key}: B4 not bit-exact vs GEE (signed median {med:.4f})"
