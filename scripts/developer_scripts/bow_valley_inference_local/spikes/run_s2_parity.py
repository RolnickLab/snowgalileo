"""CLI wrapper: run the S2 L1C → S2_HARMONIZED parity spike for one cell.

Thin entrypoint over :func:`snow_galileo.data.local_sources.parity.s2.run_s2_spike`. Run
from the repo root so the repo-relative defaults and ``src`` imports resolve::

    uv run python scripts/developer_scripts/bow_valley_inference_local/spikes/run_s2_parity.py \
        --ref tests/fixtures/gee_reference_patches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from snow_galileo.data.local_sources.parity.s2 import run_s2_spike

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="S2 L1C parity spike (TASK-005).")
    parser.add_argument(
        "--granule",
        type=Path,
        default=Path(
            "data/bow_valley_selection_raw/sentinel2/"
            "S2B_MSIL1C_20250403T184919_N0511_R113_T11UNS_20250403T222302.zip"
        ),
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("tests/fixtures/gee_reference_patches"),
        help="Reference-patch dir or a single patch tif.",
    )
    args = parser.parse_args()

    ref = args.ref
    if ref.is_dir():
        ref = ref / ("PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif")

    run_s2_spike(granule_zip=args.granule, reference_patch=ref)
    logger.info("s2_spike_done", reference=str(ref))


if __name__ == "__main__":
    main()
