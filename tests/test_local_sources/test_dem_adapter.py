"""Copernicus DEM adapter tests (TASK-007, AC-12 / AC-21).

The DEM adapter replaces the ``[DEM, slope, aspect]`` static placeholders with
the real Copernicus GLO-30 derivatives on the cell grid. The contract under test
mirrors GEE ``ee.Terrain`` + ``create_ee_image`` export (TASK-007 §2/§5):

- ``bands_out == ["DEM", "slope", "aspect"]``; output shape ``(3, *cell.shape)``;
  ``spatial_kind == "space"``; ``native_fill is None``; ``day`` ignored (static).
- **Terrain is computed in the DEM's native EPSG:4326 frame** with latitude-correct
  metric pixel spacing (Horn, degrees), *then* the three bands are resampled onto
  the cell's **EPSG:32611** grid — never computed in a projected grid.
- **Parity = GEE.** Validated against the Phase-0 GEE reference patches (which carry
  DEM/slope/aspect as bands 305/306/307). The native-frame + **nearest** resample
  recipe was measured across all six patches: DEM median ≤1.0 m, slope median
  ≤1.5°, aspect median (circular) ≤12°. Bilinear resampling roughly doubles the
  slope error — nearest replicates how GEE upsampled native ~30 m terrain to 10 m
  (decision recorded 2026-06-04, PARITY_SPIKE_NOTES.md).
- **Degenerate guard:** slopes must NOT all be ≈90° — the classic pixel-spacing bug
  (Horn on a degree grid with unit spacing → gradients ×111 000) is caught here.

Uses the real clipped DEM archive + the real reference patches; skips cleanly if
either is absent.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from src.data.local_sources.base import CELL_TARGET_CRS, GridCell

#: Clipped Copernicus DEM archive root (the adapter's input).
_DEM_ROOT = Path("data/clipped_bow_valley_selection_raw/dem")

#: Phase-0 GEE reference patches (308-band cubes; DEM/slope/aspect = bands 305/306/307).
_REF_DIR = Path("tests/fixtures/gee_reference_patches")

#: GEE static-band positions inside the 308-band reference cube (1-based).
_DEM_BAND, _SLOPE_BAND, _ASPECT_BAND = 305, 306, 307

#: Empirically-measured parity tolerances vs GEE (median over the interior).
_DEM_MED_TOL_M = 1.0
_SLOPE_MED_TOL_DEG = 1.5
_ASPECT_MED_TOL_DEG = 12.0

#: Interior margin (px) dropped before diffing — edge pixels carry warp artefacts.
_EDGE_PX = 5


def _ref_patches() -> list[Path]:
    return sorted(_REF_DIR.glob("PR_*.tif"))


def _cell_from_patch(patch: Path) -> GridCell:
    """Build a :class:`GridCell` matching a reference patch's exact grid.

    The patches are EPSG:32611 at 10 m but are not the default 100×100 square, so
    we construct the cell directly from the patch ``(transform, shape, bounds)``
    rather than via :meth:`GridCell.from_utm_bounds`.
    """
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=0,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


def _ref_static_bands(patch: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(patch) as ds:
        return (
            ds.read(_DEM_BAND).astype(np.float64),
            ds.read(_SLOPE_BAND).astype(np.float64),
            ds.read(_ASPECT_BAND).astype(np.float64),
        )


@pytest.fixture()
def adapter():
    """The real DEM adapter; skip if the clipped archive is missing."""
    if not any(_DEM_ROOT.rglob("*.tif")):
        pytest.skip(f"No clipped DEM tiles under {_DEM_ROOT}")
    from src.data.local_sources.dem import DemAdapter

    return DemAdapter(archive_root=_DEM_ROOT)


@pytest.fixture()
def patch() -> Path:
    patches = _ref_patches()
    if not patches:
        pytest.skip(f"No GEE reference patches under {_REF_DIR}")
    return patches[0]


def test_bands_out_and_kind(adapter) -> None:
    """``bands_out`` is the ordered terrain triple; static-space tier (AC-1)."""
    assert adapter.bands_out == ["DEM", "slope", "aspect"]
    assert adapter.spatial_kind == "space"
    assert adapter.native_fill is None


def test_fetch_shape_and_grid(adapter, patch: Path) -> None:
    """Output is ``(3, H, W)`` on the patch's UTM grid (AC-1 golden-grid triple)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=None)
    assert out.shape == (3, *cell.shape)
    assert cell.crs == CELL_TARGET_CRS


def test_static_independent_of_day(adapter, patch: Path) -> None:
    """The terrain triple is identical for any ``day`` and ``None`` (static, AC-2)."""
    cell = _cell_from_patch(patch)
    a = adapter.fetch(cell, day=None)
    b = adapter.fetch(cell, day=datetime.date(2025, 4, 6))
    np.testing.assert_array_equal(a, b)


def test_slopes_not_all_degenerate(adapter, patch: Path) -> None:
    """Degenerate guard: the pixel-spacing bug would push every slope to ≈90° (AC-2)."""
    cell = _cell_from_patch(patch)
    slope = adapter.fetch(cell, day=None)[1]
    valid = (slope >= 0) & (slope < 90)
    assert valid.any(), "no valid slope pixels"
    # A real mountain crop spans a range; if spacing were wrong all slopes pin at ~90°.
    assert np.median(slope[valid]) < 60.0, "slopes near-uniformly ≈90° — pixel-spacing bug"


@pytest.mark.parametrize("patch", _ref_patches(), ids=lambda p: p.name[:22])
def test_parity_against_gee(adapter, patch: Path) -> None:
    """DEM/slope/aspect match the GEE reference within measured tolerances (AC-2)."""
    cell = _cell_from_patch(patch)
    out = adapter.fetch(cell, day=None)
    dem_ref, slope_ref, aspect_ref = _ref_static_bands(patch)

    interior = np.zeros(cell.shape, dtype=bool)
    interior[_EDGE_PX:-_EDGE_PX, _EDGE_PX:-_EDGE_PX] = True
    valid = interior & (slope_ref >= 0) & (slope_ref < 90) & (dem_ref > 0.0000001)
    assert valid.sum() > 100, "too few valid reference pixels to assert parity"

    dem_med = np.median(np.abs(out[0][valid] - dem_ref[valid]))
    slope_med = np.median(np.abs(out[1][valid] - slope_ref[valid]))
    # Aspect is circular (0/360 wrap) — diff on the unit circle.
    aspect_diff = np.angle(
        np.exp(1j * np.deg2rad(out[2][valid] - aspect_ref[valid])), deg=True
    )
    aspect_med = np.median(np.abs(aspect_diff))

    assert dem_med <= _DEM_MED_TOL_M, f"DEM median {dem_med:.3f} m > {_DEM_MED_TOL_M}"
    assert slope_med <= _SLOPE_MED_TOL_DEG, f"slope median {slope_med:.3f}° > {_SLOPE_MED_TOL_DEG}"
    assert aspect_med <= _ASPECT_MED_TOL_DEG, (
        f"aspect median {aspect_med:.3f}° > {_ASPECT_MED_TOL_DEG}"
    )
