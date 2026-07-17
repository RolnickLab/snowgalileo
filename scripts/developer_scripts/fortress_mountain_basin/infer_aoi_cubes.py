r"""Operator entry point — run daily FSC inference over cubes already on disk.

Step 2 of the two-step AOI pipeline. Step 1
(:mod:`scripts.developer_scripts.build_aoi_cubes_gee_url`) tiles the AOI and downloads one
8-day cube per ``(cell, day)`` via the GEE URL pipeline; this script reads those cubes,
runs the finetuned ``EncoderWithHead`` over them, and stitches the per-cell 10x10 FSC
patches into one daily COG with the canonical :class:`DailyMosaicWriter`.

The two steps are split because their cost profiles are opposite: cube download is slow,
network-bound and GEE-throttled, while inference is local and re-run often (new
checkpoint, new threshold). Cubes on disk are the hand-off boundary.

**The cube CSV is the single source of truth.** It carries, per ``(cell, day)``:
``date, crs, center_lat, center_lon, min_x, min_y, max_x, max_y``. From it this script
derives everything — the grid (from the UTM bounds), the cube filename (reconstructed
deterministically, never globbed or parsed), and the cell centre in true degrees. No AOI
re-tiling happens here, so the grid cannot drift from the one the cubes were built on.

**Why the symlinks.** The GEE exporter names cubes
``PR_{date}_{lat:.16f}_{lon:.16f}.tif_EPSG:32611.tif`` (two layers each append an
extension). The unchanged loader — :meth:`LandsatEvalDataset._tif_to_array` — parses the
cell centre straight out of the filename (``float(stem.split("_")[3])``) to build the
model's cartesian location channels, and that stray ``.tif`` makes the parse raise
``ValueError``. Rather than touch the loader (shared with training and evaluation) or the
exporter (a separate pipeline whose naming is not ours to harmonise today), each cube is
symlinked, read-only, to a loader-parsable name whose lat/lon come **from the CSV**. The
symlinks live in a temp dir and are deleted on exit; no cube on disk is renamed or moved.

Example:
    uv run python scripts/developer_scripts/infer_aoi_cubes.py \\
            --cube-csv configs/aoi_cubes/cube_cells.csv \\
        --cube-dir data/aoi_cubes \\
        --out-dir data/outputs/fortress_fsc
"""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import structlog
import torch
import typer
from einops import rearrange

from snow_galileo.data.local_sources.base import CELL_TARGET_CRS, CELL_TARGET_PX, GridCell
from snow_galileo.data.local_sources.settings import InferenceSettings
from snow_galileo.fsc.patch_predict import EncoderWithHead
from snow_galileo.inference._loader_bridge import masked_output_for_tif
from snow_galileo.inference.model import build_model
from snow_galileo.inference.mosaic import DEFAULT_FSC_PX_PER_CELL, DailyMosaicWriter

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Run daily FSC inference over pre-built AOI cubes.")

#: Indices of the six valid-data masks in the loader's 13-tuple (1=valid, 0=invalid).
#: Mirrors ``inference.driver._MASK_INDICES`` — a cell whose masks are all zero carries no
#: signal, so its prediction is dropped to nodata rather than fabricated.
_MASK_INDICES: tuple[int, ...] = (6, 7, 8, 9, 10, 11)

#: Encoder forward patch sizes for the FSC head (10x10 high-res -> 10x10 output).
#: Mirrors ``inference.driver`` and ``LandsatEval._predict_and_store_output``.
_PATCH_SIZE_HIGH_RES: int = 10
_PATCH_SIZE_MED_RES: int = 1
_PATCH_SIZE_LOW_RES: int = 1

#: Columns the cube CSV must carry (the GEE-UTM reader dialect, see grid.GEE_UTM_CSV_COLUMNS).
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "crs",
    "center_lat",
    "center_lon",
    "min_x",
    "min_y",
    "max_x",
    "max_y",
)


def _load_cube_csv(cube_csv: Path) -> pd.DataFrame:
    """Read and validate the cube CSV emitted by the build step.

    Args:
        cube_csv: Path to the CSV written by ``build_aoi_cubes_gee_url.py``.

    Returns:
        The frame, with ``date`` parsed to :class:`datetime.date`.

    Raises:
        ValueError: If a required column is missing, or a row's CRS is not the cell CRS
            (the mosaic writer places blocks by integer offset and never reprojects, so a
            foreign CRS would be silently misplaced).
    """
    frame = pd.read_csv(cube_csv)

    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Cube CSV {cube_csv} is missing column(s) {missing}.")

    foreign = sorted(set(frame["crs"].unique()) - {CELL_TARGET_CRS})
    if foreign:
        raise ValueError(
            f"Cube CSV {cube_csv} carries CRS {foreign}, but the mosaic writer assumes "
            f"{CELL_TARGET_CRS} and never reprojects."
        )

    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d").dt.date
    return frame


def _build_grid(frame: pd.DataFrame) -> tuple[list[GridCell], dict[tuple[float, float], int]]:
    """Derive the inference grid from the CSV's UTM bounds.

    The cubes were exported from these exact bounds, so rebuilding the grid here (rather
    than re-tiling the AOI) makes grid/cube drift impossible.

    Args:
        frame: The validated cube CSV.

    Returns:
        The grid cells, and a map from a cell's ``(min_x, min_y)`` origin to its ``cell_id``
        (used to attach each CSV row to its cell).
    """
    bounds = (
        frame[["min_x", "min_y", "max_x", "max_y"]]
        .drop_duplicates()
        .sort_values(["min_y", "min_x"])
        .reset_index(drop=True)
    )

    grid: list[GridCell] = []
    cell_id_by_origin: dict[tuple[float, float], int] = {}
    for cell_id, row in bounds.iterrows():
        grid.append(
            GridCell.from_utm_bounds(
                cell_id=int(cell_id),
                min_x=float(row["min_x"]),
                min_y=float(row["min_y"]),
                max_x=float(row["max_x"]),
                max_y=float(row["max_y"]),
            )
        )
        cell_id_by_origin[(float(row["min_x"]), float(row["min_y"]))] = int(cell_id)

    return grid, cell_id_by_origin


def _gee_cube_name(*, day: datetime.date, lat: float, lon: float, crs: str) -> str:
    """Reconstruct the cube filename the GEE URL exporter wrote for one CSV row.

    Deterministic reconstruction, not a glob: ``eo_eval._export_for_polygon`` names the
    download ``f"{identifier}_{crs}.tif"`` where the identifier is itself
    ``f"PR_{date}_{lat:.16f}_{lon:.16f}.tif"`` — hence the doubled extension. Reproducing
    that string exactly is the join key; nothing parses floats back out of the filename.

    Args:
        day: The window-end day.
        lat: Cell-centre latitude in true degrees (CSV ``center_lat``).
        lon: Cell-centre longitude in true degrees (CSV ``center_lon``).
        crs: The row's CRS, e.g. ``EPSG:32611``.

    Returns:
        The cube's filename as written on disk.
    """
    return f"PR_{day:%Y%m%d}_{lat:.16f}_{lon:.16f}.tif_{crs}.tif"


def _loader_safe_link(
    *, cube: Path, day: datetime.date, lat: float, lon: float, link_dir: Path
) -> Path:
    """Symlink ``cube`` to a name the unchanged loader can parse, and return the link.

    The loader derives the model's location channels via ``float(stem.split("_")[2:4])``,
    which the GEE name breaks. The link's lat/lon come from the CSV (authoritative), so the
    location channels are exact — this is a name carrier, not a lossy re-parse.

    Args:
        cube: The real cube on disk (never modified).
        day: Window-end day; the loader also reads the month from this field.
        lat: Cell-centre latitude in true degrees.
        lon: Cell-centre longitude in true degrees.
        link_dir: Temp dir the symlinks live in.

    Returns:
        Path to the symlink.
    """
    link = link_dir / f"PR_{day:%Y%m%d}_{lat:.16f}_{lon:.16f}_SC00.tif"
    if not link.exists():
        link.symlink_to(cube.resolve())
    return link


def _check_cube_shape(cube: Path) -> None:
    """Fail loudly if a cube is not exactly ``CELL_TARGET_PX`` square.

    This guard is not cosmetic. ``dataset.subset_image`` crops an oversized cube down to
    100x100 with an *unseeded* ``np.random.choice`` offset — so a 108x108 cube (what the
    build step emits at ``--buffer-m 40``) yields a prediction for a randomly shifted
    window, silently misregistered by up to 80 m, differently on every tile and every day.
    The mosaic would look plausible and be wrong. Cubes must be built at ``--buffer-m 0``.

    Args:
        cube: The cube to check.

    Raises:
        ValueError: If the cube is not ``CELL_TARGET_PX`` x ``CELL_TARGET_PX``.
    """
    with rasterio.open(cube) as src:
        shape = (src.height, src.width)
    if shape != (CELL_TARGET_PX, CELL_TARGET_PX):
        raise ValueError(
            f"Cube {cube.name} is {shape[0]}x{shape[1]}, expected "
            f"{CELL_TARGET_PX}x{CELL_TARGET_PX}. Oversized cubes are randomly cropped by the "
            "loader, which misregisters every prediction. Rebuild the cubes with --buffer-m 0."
        )


def _is_fully_masked(masked_output: object) -> bool:
    """Return ``True`` if every valid-data mask is all-zero (the cube carries no signal)."""
    return all(
        not torch.as_tensor(masked_output[i]).any()  # type: ignore[index]
        for i in _MASK_INDICES
    )


def _predict_day(
    *,
    model: EncoderWithHead,
    cubes_by_cell: dict[int, Path],
    device: str,
    batch_size: int,
    fsc_px_per_cell: int,
) -> dict[int, npt.NDArray[np.float32] | None]:
    """Run the model over one day's cubes and return the per-cell FSC patches.

    Args:
        model: The loaded ``EncoderWithHead``, already on ``device`` and in eval mode.
        cubes_by_cell: Map ``cell_id -> loader-safe cube path`` for this day.
        device: Torch device string.
        batch_size: Cells per forward pass.
        fsc_px_per_cell: FSC pixels per cell side (10 -> 100 m px on a 1 km cell).

    Returns:
        Map ``cell_id -> (fsc_px_per_cell, fsc_px_per_cell)`` float array, or ``None`` for a
        cell whose cube is fully masked (left as nodata in the mosaic).
    """
    fsc_by_cell: dict[int, npt.NDArray[np.float32] | None] = {}
    cell_ids = sorted(cubes_by_cell)

    for start in range(0, len(cell_ids), batch_size):
        batch_ids = cell_ids[start : start + batch_size]

        masked_outputs = [masked_output_for_tif(cubes_by_cell[cell_id]) for cell_id in batch_ids]
        all_masked = [_is_fully_masked(mo) for mo in masked_outputs]

        # Stack each of the 13 tensors across the batch dim, move to device.
        batched = [
            torch.stack([torch.as_tensor(mo[i]) for mo in masked_outputs]).to(device)
            for i in range(len(masked_outputs[0]))
        ]

        with torch.no_grad():
            logits = model(
                *batched,
                patch_size_high_res=_PATCH_SIZE_HIGH_RES,
                patch_size_med_res=_PATCH_SIZE_MED_RES,
                patch_size_low_res=_PATCH_SIZE_LOW_RES,
            )

        for row, (cell_id, masked) in enumerate(zip(batch_ids, all_masked, strict=True)):
            if masked:
                logger.warning("cell_fully_masked", cell_id=cell_id)
                fsc_by_cell[cell_id] = None
                continue
            fsc_by_cell[cell_id] = (
                rearrange(
                    logits[row].squeeze(-1),
                    "(h w) -> h w",
                    h=fsc_px_per_cell,
                    w=fsc_px_per_cell,
                )
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return fsc_by_cell


@app.command()
def main(
    cube_csv: Annotated[
        Path, typer.Option(help="Cube CSV written by build_aoi_cubes_gee_url.py.")
    ] = Path("configs/aoi_cubes/cube_cells.csv"),
    cube_dir: Annotated[Path, typer.Option(help="Directory holding the downloaded cubes.")] = Path(
        "data/aoi_cubes"
    ),
    out_dir: Annotated[Path, typer.Option(help="Where the daily FSC COGs are written.")] = Path(
        "data/outputs/aoi_fsc"
    ),
    config: Annotated[
        Path, typer.Option(help="Inference YAML (checkpoint, eval config, batch size, device).")
    ] = Path("configs/bow_valley/inference.yaml"),
    device: Annotated[
        Optional[str], typer.Option(help="Torch device override, e.g. 'cpu' or 'cuda'.")
    ] = None,
    batch_size: Annotated[
        Optional[int], typer.Option(help="Cells per forward pass; overrides the config.")
    ] = None,
    limit_days: Annotated[
        Optional[int], typer.Option(help="Only process the first N days (smoke run).")
    ] = None,
) -> None:
    """Read the AOI cubes, run FSC inference, and write one stitched COG per day."""
    infer = InferenceSettings.from_yaml(config)
    if device is not None:
        infer = infer.model_copy(update={"device": device})
    if batch_size is not None:
        infer = infer.model_copy(update={"batch_size": batch_size})

    if infer.device.startswith("cuda") and not torch.cuda.is_available():
        raise typer.BadParameter(
            f"device={infer.device!r} but CUDA is unavailable. Pass --device cpu."
        )

    frame = _load_cube_csv(cube_csv)
    grid, cell_id_by_origin = _build_grid(frame)
    days = sorted(frame["date"].unique())
    if limit_days is not None:
        days = days[:limit_days]

    logger.info(
        "inference_start",
        cube_csv=str(cube_csv),
        cube_dir=str(cube_dir),
        cells=len(grid),
        days=len(days),
        device=infer.device,
        batch_size=infer.batch_size,
    )

    writer = DailyMosaicWriter(grid=grid, out_dir=out_dir, fsc_px_per_cell=DEFAULT_FSC_PX_PER_CELL)
    model = build_model(infer)
    written: list[Path] = []

    # Symlinks are the loader's view of the cubes; they never outlive the run.
    with tempfile.TemporaryDirectory(prefix="aoi_cube_links_") as tmp:
        link_dir = Path(tmp)

        for day in days:
            day_rows = frame[frame["date"] == day]

            cubes_by_cell: dict[int, Path] = {}
            missing: list[str] = []
            for _, row in day_rows.iterrows():
                lat, lon = float(row["center_lat"]), float(row["center_lon"])
                name = _gee_cube_name(day=day, lat=lat, lon=lon, crs=str(row["crs"]))
                cube = cube_dir / name
                if not cube.exists():
                    missing.append(name)
                    continue
                _check_cube_shape(cube)
                cell_id = cell_id_by_origin[(float(row["min_x"]), float(row["min_y"]))]
                cubes_by_cell[cell_id] = _loader_safe_link(
                    cube=cube, day=day, lat=lat, lon=lon, link_dir=link_dir
                )

            # A missing cube is a hole in the mosaic. Fail loudly rather than quietly
            # emitting a partial day that looks complete.
            if missing:
                raise typer.BadParameter(
                    f"{len(missing)} cube(s) for {day} are absent from {cube_dir}, "
                    f"e.g. {missing[0]}. Re-run the build step for this date."
                )

            fsc_by_cell = _predict_day(
                model=model,
                cubes_by_cell=cubes_by_cell,
                device=infer.device,
                batch_size=infer.batch_size,
                fsc_px_per_cell=DEFAULT_FSC_PX_PER_CELL,
            )
            cog = writer.write_day(day, fsc_by_cell)
            written.append(cog)
            logger.info("day_written", day=day.isoformat(), cog=str(cog), cells=len(fsc_by_cell))

    logger.info("inference_complete", days=len(written), out_dir=str(out_dir))
    typer.echo(f"Wrote {len(written)} daily FSC COG(s) to {out_dir}.")


if __name__ == "__main__":
    app()
