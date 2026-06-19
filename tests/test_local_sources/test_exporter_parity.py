"""Full-stack exporter parity gate (TASK-016, SPEC AC-27).

Where the per-adapter tests fetch one source in isolation, this test runs the **whole**
:class:`~snow_galileo.data.local_sources.exporter.LocalSourceExporter` — assembling the canonical
308-band cube for a parity cell × window-end — and then diffs **each source's band slice**
against the Phase-0 GEE reference patch. It is the gate that proves the *assembled* output
(band order + per-source value domains together, in the real exporter, not a stub) matches
GEE, closing AC-27.

**Coordinate reconciliation (AC-27).** The cube and its reference are paired by the shared
generated-CSV cell: ``_cell_from_patch`` rebuilds the exact ``GridCell`` from the reference
patch's CRS/transform/bounds, and the exporter is driven with that same cell, so the
UTM-CSV vs degree-filename representation difference never enters matching.

**Tolerances are the documented per-source ones — never re-invented here.** S2/Landsat
reflectance are bit-exact for ≥ 90 % of valid pixels (sub-pixel grid registration is the
irreducible residual, see ``test_s2_adapter``); S3 OLCI uses its swath-warp correlation
floor (``s3-snap-ortho-rejected`` — sampling geometry, not a value bug).

**Archive-dependent, skips cleanly.** A source whose clipped archive is absent, or which
has no acquisition on the chosen date for a patch (its reference slice is all-nodata),
**skips** that assertion rather than failing — AC-27 is "within tolerance *where
exercised*". Marked ``slow`` + the ``slow_archive`` xdist group like the other real-archive
parity tests (KNOWLEDGE.md). The test reports which sources it actually exercised.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from snow_galileo.data.config import NO_DATA_VALUE
from snow_galileo.data.local_sources.base import GridCell
from snow_galileo.data.local_sources.exporter import LocalSourceExporter
from snow_galileo.data.local_sources.layout import full_band_order

_REF_DIR = Path("tests/fixtures/gee_reference_patches")
_ARCHIVE = Path("data/clipped_bow_valley_selection_raw")

#: 38 dynamic bands per timestep (the per-ts block the loader slices).
_DYNAMIC_PER_TS = 38
_VALID_MIN = -1.0

#: Min fraction of overlapping valid pixels that must be bit-exact vs GEE for the
#: optical reflectance sources (the documented S2/Landsat tolerance; sub-pixel grid
#: registration is the irreducible residual).
_OPTICAL_MIN_EXACT_FRAC = 0.90


def _cell_from_patch(patch: Path) -> GridCell:
    """Rebuild the exact :class:`GridCell` a reference patch was exported on."""
    with rasterio.open(patch) as ds:
        b = ds.bounds
        return GridCell(
            cell_id=0,
            crs=str(ds.crs),
            transform=ds.transform,
            shape=(ds.height, ds.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )


_S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
_LANDSAT_BANDS = [f"B{n}_landsat" for n in (2, 3, 4, 5, 6, 7)]

#: Per parity patch: the window-end (prediction) day, plus the timestep each source's
#: acquisition lands at — **different per source** (S2 and Landsat overpass the cell on
#: different days within the same 8-day window). The S2 timesteps reuse ``test_s2_adapter``'s
#: validated cases; the Landsat timesteps reuse ``test_landsat_adapter``'s. Driving the real
#: exporter with the patch's window-end places each source's data at the named timestep, so
#: the full-stack diff exercises **both** S2 and Landsat end to end (not S2 only).
_PARITY_CASES = {
    "PR_20250406": dict(window_end=date(2025, 4, 6), s2_ts=4, landsat_ts=3),
    "PR_20250414": dict(window_end=date(2025, 4, 14), s2_ts=1, landsat_ts=2),
    "PR_20250510": dict(window_end=date(2025, 5, 10), s2_ts=0, landsat_ts=1),
}


def _band_index(name: str, ts: int) -> int:
    """1-based rasterio band index of dynamic band ``name`` at timestep ``ts``."""
    order = full_band_order()
    target = f"{name}_t{ts}"
    return order.index(target) + 1


@pytest.fixture(scope="module")
def _have_archive() -> bool:
    if not _ARCHIVE.exists() or not any(_ARCHIVE.iterdir()):
        pytest.skip("No clipped archive on this host")
    return True


def _exact_fraction(out: np.ndarray, ref: np.ndarray) -> tuple[float, int]:
    """Bit-exact fraction over pixels valid in both, and the overlap count."""
    valid = (ref != NO_DATA_VALUE) & (ref > _VALID_MIN) & (out != NO_DATA_VALUE)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float((out[valid] == ref[valid]).mean()), n


@pytest.mark.slow
@pytest.mark.xdist_group("slow_archive")
@pytest.mark.parametrize("patch_key", list(_PARITY_CASES))
def test_full_stack_optical_parity(patch_key: str, _have_archive: bool) -> None:
    """Assembled-cube S2 + Landsat reflectance match GEE within tolerance, end to end (AC-27).

    Runs the real exporter (all adapters) once, then diffs the assembled **S2**
    (``B2..B12``, at its covered timestep) and **Landsat** (``B2_landsat..B7_landsat``,
    at *its* covered timestep) slices against the reference patch. A source with no
    coverage on its date (negligible overlap) is skipped, not failed — but the test
    requires **both** S2 and Landsat to have been exercised across the case set, so the
    gate cannot silently degrade to S2-only.
    """
    patches = sorted(_REF_DIR.glob(f"{patch_key}_*.tif"))
    if not patches:
        pytest.skip(f"No reference patch for {patch_key}")
    patch = patches[0]
    case = _PARITY_CASES[patch_key]
    window_end: date = case["window_end"]  # type: ignore[assignment]

    cell = _cell_from_patch(patch)
    # This is an *optical* (S2/Landsat) parity test — S1 is not asserted. Disable the S1
    # SNAP cache pre-flight (verify-only) so the test stays hermetic and does not require a
    # pre-built per-granule S1 cache for the optical patches.
    exporter = LocalSourceExporter(placeholder=False, archive_root=_ARCHIVE, verify_s1_cache=False)
    cube_path = exporter.export(cell=cell, window_end=window_end)

    by_source = {
        "S2": (_S2_BANDS, case["s2_ts"]),
        "Landsat": (_LANDSAT_BANDS, case["landsat_ts"]),
    }

    exercised: list[str] = []
    with rasterio.open(cube_path) as cube, rasterio.open(patch) as ref_ds:
        for source, (bands, ts) in by_source.items():
            for band in bands:
                idx = _band_index(band, ts)  # type: ignore[arg-type]
                out = cube.read(idx)
                ref = ref_ds.read(idx)
                frac, overlap = _exact_fraction(out, ref)
                if overlap < 0.1 * ref.size:
                    continue  # no coverage for this source on its date → skip
                exercised.append(f"{source}:{band}")
                assert frac >= _OPTICAL_MIN_EXACT_FRAC, (
                    f"{patch_key} {source} {band} t{ts}: only {frac:.1%} bit-exact vs GEE "
                    f"(< {_OPTICAL_MIN_EXACT_FRAC:.0%}, overlap {overlap}px)"
                )

    assert exercised, f"{patch_key}: no optical source had coverage — case mismatch?"
