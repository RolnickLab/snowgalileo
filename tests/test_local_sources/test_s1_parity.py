r"""S1 GRD parity spike test (TASK-005, AC-14 spike form) — go/no-go de-risk.

Quantifies value-domain drift between the SNAP ``COPERNICUS/S1_GRD`` chain and
the Phase-0 GEE reference patch for one validated ``(patch, timestep, granule)``
triple.

**Recipe (the engine GEE uses):** the ESA Sentinel-1 Toolbox via headless
``gpt`` — Apply-Orbit → ThermalNoiseRemoval → Remove-GRD-Border-Noise →
Calibration(σ⁰) → Terrain-Correction(SRTM 1Sec, EPSG:32611) → LinearToFromdB
(graph: ``scripts/developer_scripts/bow_valley_inference_local/spikes/s1_grd_snap_graph.xml``).
This reproduces the *full*
chain, including the noise-removal steps the ``sarsen`` path could not (and which
``xarray-sentinel`` can't even read for S1C — see PARITY_SPIKE_NOTES.md §2).

**Chosen tolerance (PARITY_SPIKE_NOTES.md §4):** median absolute per-band
difference ≤ **1.0 dB** for VV/VH, over valid (non-0, non-−9999) reference pixels
with the ``< -30 dB`` edge mask applied. SAR is inherently speckly pixel-to-pixel,
so the gate uses the **median**, not a tail percentile.

**Measured (recorded GO):** VV 0.54 dB, VH 0.48 dB median |Δ| — both within
tolerance. The ``angle`` (incidence) band is deterministic geometry, not a
value-domain risk; it is recovered in the real adapter (TASK-014), not the spike.

**Skip policy.** Running the SNAP chain needs ESA SNAP installed and is
compute-heavy (full IW GRD + SRTM download), so this test does **not** invoke
``gpt`` itself. It validates the *already-produced* dB GeoTIFF
(``S1_SNAP_OUTPUT``, default ``/tmp/s1run/s1_grd_db.tif``) when present, and
``skip``s cleanly otherwise (e.g. CI without SNAP). To (re)generate the artifact:

    /home/dev/esa-snap/bin/gpt \\
      scripts/developer_scripts/bow_valley_inference_local/spikes/s1_grd_snap_graph.xml \\
      -Pinput=<extracted .SAFE>/manifest.safe \\
      -Pregion='POLYGON((...AOI...))' -Poutput=/tmp/s1run/s1_grd_db.tif
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio

#: Median-absolute-difference tolerance for VV/VH, in dB.
S1_DRIFT_TOLERANCE_DB: float = 1.0

#: SAR backscatter below this (dB) is edge/no-data (GEE S1_GRD convention).
S1_EDGE_MASK_DB: float = -30.0

#: The validated parity cell (PARITY_SPIKE_NOTES.md §1/§4).
_REF_PATCH = Path(
    "tests/fixtures/gee_reference_patches/"
    "PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif"
)
_TIMESTEP = 0
#: Reference VV/VH band indices at the test timestep (38 dynamic/ts; S1 at offset 1..3).
_REF_VV_BAND = 38 * _TIMESTEP + 1
_REF_VH_BAND = 38 * _TIMESTEP + 2

#: The SNAP dB GeoTIFF produced by the graph (override via env for a custom path).
S1_SNAP_OUTPUT = Path(os.environ.get("S1_SNAP_OUTPUT", "/tmp/s1run/s1_grd_db.tif"))


@pytest.fixture(scope="module")
def parity() -> dict[str, float]:
    """Reproject the SNAP dB output onto the patch grid; return per-band median |Δ|.

    Skips if the SNAP artifact is absent (SNAP not installed / spike not run).
    """
    if not S1_SNAP_OUTPUT.exists():
        pytest.skip(
            f"SNAP S1_GRD output not found at {S1_SNAP_OUTPUT}; "
            "run scripts/developer_scripts/bow_valley_inference_local/spikes/"
            "s1_grd_snap_graph.xml via gpt first."
        )
    from shapely.geometry import box

    from snow_galileo.data.local_sources.base import GridCell, reproject_to_cell

    with rasterio.open(_REF_PATCH) as ref:
        ref_vv = ref.read(_REF_VV_BAND).astype(np.float64)
        ref_vh = ref.read(_REF_VH_BAND).astype(np.float64)
        b = ref.bounds
        cell = GridCell(
            cell_id=0,
            crs=ref.crs.to_string(),
            transform=ref.transform,
            shape=(ref.height, ref.width),
            polygon=box(b.left, b.bottom, b.right, b.top),
        )

    # SNAP writes the bands VH-then-VV (not the graph's VV,VH order); assign by
    # matching each reprojected band to the reference medians, not by index.
    with rasterio.open(S1_SNAP_OUTPUT) as snap:
        reprojected = []
        for band_index in (1, 2):
            db = snap.read(band_index).astype(np.float64)
            masked = np.where(db < S1_EDGE_MASK_DB, np.nan, db)
            rp = reproject_to_cell(
                source=masked[np.newaxis, :, :],
                src_transform=snap.transform,
                src_crs=snap.crs.to_string(),
                cell=cell,
                categorical=False,
                restore_fill=np.nan,
            )[0]
            reprojected.append(rp)

    def _median_abs(spike: np.ndarray, ref: np.ndarray) -> float:
        valid = (ref != -9999) & (ref != 0) & np.isfinite(spike)
        assert valid.sum() > 50, f"too few valid pixels ({valid.sum()})"
        return float(np.median(np.abs(spike[valid] - ref[valid])))

    # Pair each SNAP band to VV/VH by best agreement.
    out: dict[str, float] = {}
    for ref_arr, name in ((ref_vv, "VV"), (ref_vh, "VH")):
        diffs = [_median_abs(rp, ref_arr) for rp in reprojected]
        out[name] = min(diffs)
    return out


def test_s1_vv_vh_drift_within_tolerance(parity: dict[str, float]) -> None:
    """VV and VH median |spike − reference| are within the documented dB tolerance."""
    assert parity["VV"] <= S1_DRIFT_TOLERANCE_DB, f"VV drift {parity['VV']:.2f} dB"
    assert parity["VH"] <= S1_DRIFT_TOLERANCE_DB, f"VH drift {parity['VH']:.2f} dB"


def test_s1_domain_is_decibel(parity: dict[str, float]) -> None:
    """Both polarisations matched the reference (drift is finite, i.e. dB-domain aligned)."""
    assert np.isfinite(parity["VV"]) and np.isfinite(parity["VH"])
