"""CLI wrapper: run the S3 OLCI ortho parity spike and compare to the swath-warp adapter.

Thin entrypoint over :func:`snow_galileo.data.local_sources.parity.s3.run_s3_ortho_spike`. It
supplies the SNAP ortho graph shipped alongside this script, then reports median |Δ|
and correlation for the ortho output vs the production :class:`S3Adapter` swath-warp,
both against the GEE reference patch. Run from the repo root::

    uv run python scripts/developer_scripts/bow_valley_inference_local/spikes/run_s3_parity.py
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import rasterio
import structlog

from snow_galileo.data.local_sources.parity.s3 import (
    DEFAULT_GPT,
    S3_BANDS,
    _grid_from_patch,
    run_s3_ortho_spike,
    stats,
)
from snow_galileo.data.local_sources.s3 import S3Adapter

logger = structlog.get_logger(__name__)

#: SNAP OLCI ortho graph, shipped next to this wrapper.
_GRAPH = Path(__file__).with_name("s3_olci_ortho_graph.xml")

#: 38-band dynamic block: Oa17 at offset 15, Oa21 at 16 (matches test_s3_adapter).
_DYNAMIC_PER_TS = 38
_OFF = {"Oa17_radiance": 15, "Oa21_radiance": 16}


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 OLCI ortho parity spike (TASK-011, SNAP).")
    parser.add_argument(
        "--product",
        type=Path,
        default=Path(
            "data/clipped_bow_valley_selection_raw/sentinel3/"
            "S3A_OL_1_EFR____20250401T183122_20250401T183422_20250402T192445"
            "_0179_124_184_1980_PS1_O_NT_004.zip"
        ),
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=Path(
            "tests/fixtures/gee_reference_patches/"
            "PR_20250406_562863.8459204244427383_5653083.7883343594148755.tif"
        ),
    )
    parser.add_argument("--day", type=str, default="2025-04-01")
    parser.add_argument("--timestep", type=int, default=2)
    parser.add_argument("--gpt", type=Path, default=DEFAULT_GPT)
    parser.add_argument("--graph", type=Path, default=_GRAPH)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args()

    day = datetime.date.fromisoformat(args.day)

    # Ortho spike on the patch grid.
    ortho = run_s3_ortho_spike(
        product_zip=args.product,
        reference_patch=args.ref,
        graph=args.graph,
        workdir=args.workdir,
        gpt=args.gpt,
    )

    # Baseline: the current swath-warp adapter on the same cell/day.
    cell = _grid_from_patch(args.ref)
    adapter = S3Adapter(archive_root=args.product.parent)
    warp = adapter.fetch(cell, day=day)
    warp_by_band = {b: warp[i].astype(float) for i, b in enumerate(S3_BANDS)}

    with rasterio.open(args.ref) as ds:
        for band in S3_BANDS:
            ref = ds.read(_DYNAMIC_PER_TS * args.timestep + _OFF[band] + 1).astype(float)
            o_med, o_corr, o_n = stats(ortho[band], ref)
            w = warp_by_band[band]
            w[w == -9999] = float("nan")
            w_med, w_corr, w_n = stats(w, ref)
            logger.info(
                "s3_parity_compare",
                band=band,
                ortho_median_abs=round(o_med, 2),
                ortho_corr=round(o_corr, 3),
                ortho_n=o_n,
                warp_median_abs=round(w_med, 2),
                warp_corr=round(w_corr, 3),
                warp_n=w_n,
                corr_delta=round(o_corr - w_corr, 3),
            )

    logger.info("s3_ortho_spike_done", product=args.product.name, ref=args.ref.name)


if __name__ == "__main__":
    main()
