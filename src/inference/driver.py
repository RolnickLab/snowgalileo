"""Inference grid driver (TASK-015, SPEC FR-21 / AC-31).

:class:`InferenceGridDriver` is the **additive** direct-source inference entry
point: for each day in the configured window it exports every cell's 8-day cube,
runs an (injected) ``EncoderWithHead`` over the cells, and hands the per-cell
10×10 FSC predictions to :class:`~src.inference.mosaic.DailyMosaicWriter`.

**Downstream is sacred.** The driver injects a ready ``EncoderWithHead`` (built by
the TASK-016 entry-point script from a checkpoint) and a ready
``LocalSourceExporter``; it edits no downstream code. It drives the loader only
through :func:`src.inference._loader_bridge.masked_output_for_tif`, which calls
the unchanged ``LandsatEvalDataset`` inference path. The GEE
``_predict_and_store_output`` runner is untouched and keeps working in parallel.

**The loop ignores the CSV ``date`` column (Q4 / AC-31).** Days come solely from
``[window_start, window_end]`` via :func:`~src.inference.windows.inference_days`;
the cells are ``GridCell`` objects (geometry only). Two cells whose legacy CSV
``date`` differ are predicted on the *same* configured day.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog
import torch
from einops import rearrange

from src.data.local_sources.base import GridCell
from src.data.local_sources.exporter import LocalSourceExporter
from src.data.local_sources.layout import build_cube_filename
from src.data.local_sources.parallel_export import export_cells_parallel
from src.fsc.patch_predict import EncoderWithHead
from src.inference._loader_bridge import masked_output_for_tif
from src.inference.mosaic import DEFAULT_FSC_PX_PER_CELL, DailyMosaicWriter
from src.inference.windows import inference_days

logger = structlog.get_logger(__name__)

#: Indices of the six valid-data masks within the loader's 13-tuple MaskedOutput
#: (s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m). Convention: 1=valid, 0=invalid.
_MASK_INDICES: tuple[int, ...] = (6, 7, 8, 9, 10, 11)

#: Encoder forward patch sizes for the FSC head (10×10 high-res → 10×10 output),
#: matching ``_predict_and_store_output`` and ``test_tracer_end_to_end.py``.
_PATCH_SIZE_HIGH_RES: int = 10
_PATCH_SIZE_MED_RES: int = 1
_PATCH_SIZE_LOW_RES: int = 1


class InferenceGridDriver:
    """Drives per-day, per-cell FSC inference and daily mosaicking.

    Args:
        exporter: Cube exporter (assembles a cell's 308-band cube tif). Injected so
            production passes a real-adapter exporter and tests pass a placeholder one.
        model: A ready ``EncoderWithHead`` (eval mode is set internally). Injected so
            tests pass a tiny untrained encoder — no checkpoint, no GPU required.
        grid: The inference grid cells (geometry only; CSV ``date`` is never read).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).
        out_dir: Daily-FSC COG output directory (the deliverable).
        device: Torch device for inference (default CPU).
        batch_size: Cells per encoder forward pass.
        fsc_px_per_cell: FSC pixels per cell side (10 → 100 m px on a 1 km cell).
    """

    def __init__(
        self,
        *,
        exporter: LocalSourceExporter,
        model: EncoderWithHead,
        grid: list[GridCell],
        window_start: datetime.date,
        window_end: datetime.date,
        out_dir: Path,
        device: str | torch.device = "cpu",
        batch_size: int = 8,
        fsc_px_per_cell: int = DEFAULT_FSC_PX_PER_CELL,
        export_workers: int | None = None,
    ) -> None:
        self.exporter = exporter
        self.model = model.to(device).eval()
        self.grid = grid
        self.window_start = window_start
        self.window_end = window_end
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.fsc_px_per_cell = fsc_px_per_cell
        self.export_workers = export_workers
        #: Per-day ``cell_id -> cube tif`` map filled by the parallel pre-export (cleared
        #: each day). Empty → ``_run_batch`` falls back to the injected exporter's serial
        #: ``.export`` (the path tests with a stub exporter rely on).
        self._tif_for_cell: dict[int, Path] = {}
        self.mosaic = DailyMosaicWriter(
            grid=grid, out_dir=out_dir, fsc_px_per_cell=fsc_px_per_cell
        )

    def run(self) -> list[Path]:
        """Run the full sweep; return the daily-FSC COG paths (one per day).

        Returns:
            The written daily-FSC COG paths in ascending day order.
        """
        outputs: list[Path] = []
        for day in inference_days(self.window_start, self.window_end):
            fsc_by_cell = self._predict_day(day)
            outputs.append(self.mosaic.write_day(day, fsc_by_cell))
        logger.info(
            "inference_sweep_complete",
            days=len(outputs),
            cells=len(self.grid),
            window=f"{self.window_start.isoformat()}..{self.window_end.isoformat()}",
        )
        return outputs

    def _predict_day(
        self, day: datetime.date
    ) -> dict[int, npt.NDArray[np.float32] | None]:
        """Export, batch, and run the encoder for every cell on one day.

        Args:
            day: The inference (window-end) day.

        Returns:
            Map ``cell_id -> 10×10 FSC array`` (or ``None`` for a cell whose every
            input is masked → no prediction, left as nodata in the mosaic).
        """
        self._tif_for_cell = self._pre_export_day(day)

        fsc_by_cell: dict[int, npt.NDArray[np.float32] | None] = {}
        batch: list[GridCell] = []

        def flush() -> None:
            if batch:
                self._run_batch(batch, day, fsc_by_cell)
                batch.clear()

        for cell in self.grid:
            batch.append(cell)
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return fsc_by_cell

    def _pre_export_day(self, day: datetime.date) -> dict[int, Path]:
        """Export every cell's cube for ``day`` up front, in parallel where possible.

        Only engages the process pool when ``export_workers`` resolves to >1 **and** the
        injected exporter is a real :class:`LocalSourceExporter` (the pool rebuilds an
        exporter per worker from its ``out_dir``/``archive_root``, which a stub lacks).
        Otherwise returns an empty map and ``_run_batch`` exports serially through the
        injected exporter — the path the stub-exporter tests exercise.
        """
        if (
            self.export_workers is None
            or self.export_workers <= 1
            or not isinstance(self.exporter, LocalSourceExporter)
        ):
            return {}

        paths = export_cells_parallel(
            cells=self.grid,
            window_end=day,
            out_dir=self.exporter.out_dir,
            archive_root=self.exporter.archive_root,
            workers=self.export_workers,
        )
        # Map back to cell_id by the filename the exporter wrote (PR_<date>_<lat>_<lon>).
        by_name = {p.name: p for p in paths}
        tif_for_cell: dict[int, Path] = {}
        for cell in self.grid:
            lat, lon = self.exporter._cell_centre_lat_lon(cell)
            name = build_cube_filename(window_end=day, lat=lat, lon=lon)
            if name in by_name:
                tif_for_cell[cell.cell_id] = by_name[name]
        return tif_for_cell

    def _run_batch(
        self,
        cells: list[GridCell],
        day: datetime.date,
        fsc_by_cell: dict[int, npt.NDArray[np.float32] | None],
    ) -> None:
        """Export+infer one batch of cells, writing results into ``fsc_by_cell``."""
        masked_outputs = []
        all_masked: list[bool] = []
        for cell in cells:
            # Prefer the parallel pre-export's tif; else export serially (stub-exporter path).
            tif = self._tif_for_cell.get(cell.cell_id)
            if tif is None:
                tif = self.exporter.export(cell=cell, window_end=day)
            mo = masked_output_for_tif(tif)
            masked_outputs.append(mo)
            all_masked.append(self._is_fully_masked(mo))

        # Stack each of the 13 tensors across the batch dim, move to device.
        batched = [
            torch.stack([torch.as_tensor(mo[i]) for mo in masked_outputs]).to(self.device)
            for i in range(len(masked_outputs[0]))
        ]

        with torch.no_grad():
            logits = self.model(
                *batched,
                patch_size_high_res=_PATCH_SIZE_HIGH_RES,
                patch_size_med_res=_PATCH_SIZE_MED_RES,
                patch_size_low_res=_PATCH_SIZE_LOW_RES,
            )

        n = self.fsc_px_per_cell
        for row, (cell, masked) in enumerate(zip(cells, all_masked, strict=True)):
            if masked:
                fsc_by_cell[cell.cell_id] = None
                continue
            patch = (
                rearrange(logits[row].squeeze(-1), "(h w) -> h w", h=n, w=n)
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            fsc_by_cell[cell.cell_id] = patch

    @staticmethod
    def _is_fully_masked(masked_output: object) -> bool:
        """Return ``True`` if every valid-data mask is all-zero (no information).

        The loader's mask convention is 1=valid, 0=invalid; a cell with no valid
        input on any of the six channel groups carries no signal, so its
        prediction is dropped to nodata in the mosaic (SPEC AC-28).
        """
        mo = masked_output  # the 13-tuple MaskedOutput
        return all(
            not torch.as_tensor(mo[i]).any()  # type: ignore[index]
            for i in _MASK_INDICES
        )
