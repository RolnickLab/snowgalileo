r"""One-granule S1 SNAP diagnostic: production graph vs border-noise-removed (A/B).

Three S1C granule segments came out of the production SNAP chain as ~4x5.5 km slivers
despite full-size raw input and 27-61% AOI footprint overlap. The legacy
``Remove-GRD-Border-Noise`` op is the prime suspect. This runs the SAME granule through
both the production graph and a variant with that node removed, into a SCRATCH dir (the
real cache is never touched), then reports each output's size / extent / valid-pixel count
so the cause is obvious at a glance.

Mirrors ``s1_snap.build_granule_cache`` exactly (zip extract -> manifest -> identical AOI
region WKT -> same gpt invocation) so the only variable is the graph.

Run (defaults to the 0406 sliver granule):
    uv run python scripts/developer_scripts/bow_valley_inference_local/spikes/diagnose_s1_border_noise.py

    # or another of the three:
    uv run python scripts/developer_scripts/bow_valley_inference_local/spikes/diagnose_s1_border_noise.py \
        --granule S1C_IW_GRDH_1SDV_20250423T013725_20250423T013750_002014_004184_F75D
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import zipfile
from pathlib import Path

import rasterio
import structlog

from snow_galileo.data.local_sources.clip.settings import load_aoi_polygon
from snow_galileo.data.local_sources.paths import LocalPaths
from snow_galileo.data.local_sources.s1_snap import (
    _DEFAULT_GPT,
    _DEFAULT_GRAPH,
    _aoi_region_wkt,
)

logger = structlog.get_logger(__name__)

#: The border-noise-removed diagnostic graph (this directory).
_NO_BN_GRAPH = Path(__file__).with_name("s1_grd_graph_no_border_noise.xml")

#: Default granule: the 0406 sliver (60.9% AOI overlap, output 393x550).
_DEFAULT_GRANULE = "S1C_IW_GRDH_1SDV_20250406T012913_20250406T012938_001766_00323E_71B5"


def _run_graph(*, manifest: Path, region_wkt: str, out_tif: Path, gpt: Path, graph: Path) -> None:
    """Invoke gpt with the EXACT production flags (-c 2G -q 4)."""
    cmd = [
        str(gpt),
        str(graph),
        f"-Pinput={manifest}",
        f"-Pregion={region_wkt}",
        f"-Poutput={out_tif}",
        "-c",
        "2G",
        "-q",
        "4",
    ]
    logger.info("gpt_start", graph=graph.name, out=out_tif.name)
    subprocess.run(cmd, check=True)


def _report(label: str, tif: Path) -> None:
    """Print size / extent (km) / decimated valid-VV count for an output tif."""
    if not tif.exists():
        print(f"  {label:18s} NO OUTPUT (gpt produced nothing — empty crop?)")
        return
    with rasterio.open(tif) as ds:
        b = ds.bounds
        w_km = (b.right - b.left) / 1000
        h_km = (b.top - b.bottom) / 1000
        # Decimated read of band 1 (Sigma0_VH linear) — cheap valid-pixel proxy.
        import numpy as np

        arr = ds.read(1, out_shape=(1, min(512, ds.height), min(512, ds.width)))
        valid = int(np.sum(arr > 0))
        print(
            f"  {label:18s} {ds.width:6d}x{ds.height:<6d} "
            f"{w_km:6.1f}x{h_km:5.1f}km  crs={ds.crs}  VV>0(decim)={valid}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granule", default=_DEFAULT_GRANULE, help="Granule stem (no .zip).")
    parser.add_argument("--gpt", type=Path, default=_DEFAULT_GPT)
    parser.add_argument("--prod-graph", type=Path, default=_DEFAULT_GRAPH)
    parser.add_argument("--no-bn-graph", type=Path, default=_NO_BN_GRAPH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/bow_valley_processing/scratch/s1_border_noise_diag"),
        help="Scratch output dir (NOT the real cache).",
    )
    args = parser.parse_args()

    paths = LocalPaths()
    granule_zip = paths.raw_root / "sentinel1" / f"{args.granule}.zip"
    if not granule_zip.exists():
        raise SystemExit(f"Raw granule not found: {granule_zip}")
    if not args.gpt.exists():
        raise SystemExit(f"gpt not found: {args.gpt}")

    aoi = load_aoi_polygon(paths.aoi_path)
    region_wkt = _aoi_region_wkt(aoi)  # IDENTICAL AOI bbox the production build uses

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prod_tif = args.out_dir / f"PROD_{args.granule}.tif"
    no_bn_tif = args.out_dir / f"NOBN_{args.granule}.tif"
    prod_tif.unlink(missing_ok=True)
    no_bn_tif.unlink(missing_ok=True)

    # Extract the raw zip ONCE; run both graphs off the same .SAFE so only the graph differs.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(granule_zip) as zf:  # read-only: raw archive never mutated
            zf.extractall(tmp_dir)
        manifest = next(tmp_dir.glob("*.SAFE")) / "manifest.safe"

        _run_graph(
            manifest=manifest,
            region_wkt=region_wkt,
            out_tif=prod_tif,
            gpt=args.gpt,
            graph=args.prod_graph,
        )
        _run_graph(
            manifest=manifest,
            region_wkt=region_wkt,
            out_tif=no_bn_tif,
            gpt=args.gpt,
            graph=args.no_bn_graph,
        )

    print(f"\n=== S1 border-noise diagnostic: {args.granule} ===")
    print("  (a sliver from PROD that fills out in NOBN => Remove-GRD-Border-Noise is the cause)")
    _report("PROD (with BN)", prod_tif)
    _report("NOBN (no BN)", no_bn_tif)
    print(f"\n  outputs kept in {args.out_dir} for inspection.")


if __name__ == "__main__":
    main()
