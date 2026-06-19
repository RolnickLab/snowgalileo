"""CLI wrapper: run the S1 GRD → S1_GRD parity spike (SNAP) for one cell.

Thin entrypoint over :func:`snow_galileo.data.local_sources.parity.s1.run_s1_spike`. It
supplies the SNAP graph that ships alongside this script. Run from the repo root
so the repo-relative defaults and ``src`` imports resolve::

    uv run python scripts/developer_scripts/bow_valley_inference_local/spikes/run_s1_parity.py \
        --ref tests/fixtures/gee_reference_patches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from snow_galileo.data.local_sources.parity.s1 import DEFAULT_GPT, run_s1_spike

logger = structlog.get_logger(__name__)

#: SNAP S1_GRD graph, shipped next to this wrapper.
_GRAPH = Path(__file__).with_name("s1_grd_snap_graph.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="S1 GRD parity spike (TASK-005, SNAP).")
    parser.add_argument(
        "--granule",
        type=Path,
        default=Path(
            "data/bow_valley_selection_raw/sentinel1/"
            "S1C_IW_GRDH_1SDV_20250330T013724_20250330T013749_001664_002BB2_88AD.zip"
        ),
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path("tests/fixtures/gee_reference_patches"),
    )
    parser.add_argument("--gpt", type=Path, default=DEFAULT_GPT)
    parser.add_argument("--graph", type=Path, default=_GRAPH)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()

    ref = args.ref
    if ref.is_dir():
        ref = ref / "PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif"

    run_s1_spike(
        granule_zip=args.granule,
        reference_patch=ref,
        graph=args.graph,
        workdir=args.workdir,
        gpt=args.gpt,
    )
    logger.info("s1_spike_done", reference=str(ref))


if __name__ == "__main__":
    main()
