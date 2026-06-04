"""S2 L1C parity spike test (TASK-005, AC-15 spike form) — go/no-go de-risk.

Quantifies value-domain drift between a minimal **throwaway** S2 spike
(`scripts/spikes/s2_parity_spike.py`) and the Phase-0 GEE reference patch, for
one validated ``(patch, timestep, granule)`` triple where the raw L1C tile
genuinely covers the patch footprint.

**Recipe under test (what `COPERNICUS/S2_HARMONIZED` is):** for processing
baseline ≥ N0400 (all archive granules are **N0511**), the harmonized product is
L1C DN with a **−1000 DN** offset applied; reflectance = ``DN / 10000`` downstream
(no atmospheric correction — L1C TOA). The spike reads the JP2 bands, subtracts
1000, reprojects onto the reference patch's exact cell grid, and the test diffs
per band against the patch.

**Chosen tolerance (documented, PARITY_SPIKE_NOTES.md §4):** median absolute
per-band difference ≤ **50 DN** (post −1000). The harmonized DN domain is ~0–10000
(reflectance ×10000); 50 DN = 0.005 reflectance, well under the model's
normalization sensitivity. The diff is taken over valid (non-``-9999``, non-zero)
reference pixels only.

**Validated cell (see PARITY_SPIKE_NOTES.md):**
``PR_20250406…562863.8…5653083.8`` timestep 4 (date 2025-04-03), tile ``T11UNS``.
The matching granule was chosen by *footprint coverage*, not the first tile of
that date — the first-listed ``T11UNT`` tile does **not** cover this southern
patch, a trap a naive pick would have hit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

#: Median-absolute-difference tolerance, in harmonized DN (post −1000 offset).
S2_DRIFT_TOLERANCE_DN: float = 50.0

#: The validated parity cell (PARITY_SPIKE_NOTES.md §1/§4).
_REF_PATCH = (
    "tests/fixtures/gee_reference_patches/"
    "PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif"
)
_TIMESTEP = 4
_GRANULE = (
    "data/bow_valley_selection_raw/sentinel2/"
    "S2B_MSIL1C_20250403T184919_N0511_R113_T11UNS_20250403T222302.zip"
)
_S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
#: 1-based band indices of the S2 block at the test timestep (38 dynamic/ts; S2 at offset 4..9).
_S2_BAND_INDICES = [38 * _TIMESTEP + 4 + i for i in range(len(_S2_BANDS))]


@pytest.fixture(scope="module")
def spike_output() -> dict[str, np.ndarray]:
    """Run the S2 spike for the validated cell; return ``{band: array}`` on the patch grid."""
    pytest.importorskip("rasterio")
    from scripts.spikes.s2_parity_spike import run_s2_spike

    return run_s2_spike(
        granule_zip=Path(_GRANULE),
        reference_patch=Path(_REF_PATCH),
    )


def _reference_band(index: int) -> np.ndarray:
    """Read one 1-based band from the reference patch."""
    with rasterio.open(_REF_PATCH) as src:
        return src.read(index).astype(np.float64)


def test_s2_drift_within_tolerance(spike_output: dict[str, np.ndarray]) -> None:
    """Each S2 band's median |spike − reference| is within the documented tolerance."""
    drift: dict[str, float] = {}
    for band, index in zip(_S2_BANDS, _S2_BAND_INDICES):
        ref = _reference_band(index)
        spike = spike_output[band].astype(np.float64)
        assert spike.shape == ref.shape, f"{band}: shape {spike.shape} != ref {ref.shape}"

        # Diff over pixels the reference actually populated (GEE writes 0 / -9999
        # where no acquisition fell in the slot).
        valid = (ref != -9999) & (ref != 0)
        assert valid.sum() > 50, f"{band}: too few valid reference pixels ({valid.sum()})"

        median_abs = float(np.median(np.abs(spike[valid] - ref[valid])))
        drift[band] = median_abs

    worst = max(drift.values())
    assert worst <= S2_DRIFT_TOLERANCE_DN, (
        f"S2 per-band drift exceeds {S2_DRIFT_TOLERANCE_DN} DN: {drift}"
    )


def test_s2_domain_is_harmonized(spike_output: dict[str, np.ndarray]) -> None:
    """The spike's −1000-offset output sits in the harmonized DN domain (~0–10000+)."""
    for band in _S2_BANDS:
        arr = spike_output[band]
        finite = arr[np.isfinite(arr) & (arr != -9999)]
        assert finite.size > 0, f"{band}: no finite pixels"
        # Post-offset L1C TOA: non-negative after offset, comfortably below saturation.
        assert finite.min() >= -100, f"{band}: min {finite.min()} below harmonized floor"
        assert finite.max() <= 20000, f"{band}: max {finite.max()} above plausible TOA ceiling"
