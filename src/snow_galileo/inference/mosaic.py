"""Daily FSC mosaic writer (TASK-015, SPEC FR-22 / AC-28 / AC-29).

:class:`DailyMosaicWriter` stitches per-cell 10×10 fractional-snow-cover (FSC)
predictions into **one COG per day** over the AOI, in EPSG:32611.

**Direct UTM placement (no cross-CRS reproject).** Each cell's cube grid is
already EPSG:32611 (PLAN §5, corrected 2026-06-04; ``cube.yaml`` ``cell_crs``), so
each 10×10 FSC patch is already in UTM 11N at 100 m/px (``1000 m / 10 px``). The
mosaic grid is the union of all cell bounds on that same 100 m lattice, so a
patch maps to an **exact integer pixel offset** — placement is a block copy, not
a warp. The SPEC's "reproject from EPSG:4326 with nearest-neighbour" wording
predates the CRS correction; nearest is still the *only* resampling that would
ever be permitted on a prediction raster, but with aligned 100 m cells none is
needed (the placed values are bit-identical to the input patch — no interpolated
FSC is ever invented).

Cells are non-overlapping by construction (``grid.py`` guarantees pairwise
intersection area 0), so the target blocks are disjoint: the writer asserts no
pixel is double-written (AC-29 seam guard). A cell with no valid prediction
(``None``) leaves its block at ``nodata`` (AC-28). The per-day AOI-coverage
fraction (valid placed px / total cell px) is recorded as an output tag.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from affine import Affine

from snow_galileo.data.config import NO_DATA_VALUE
from snow_galileo.data.local_sources.base import CELL_TARGET_CRS, GridCell

logger = structlog.get_logger(__name__)

#: Side length (px) of one cell's FSC patch — 10 (the encoder's 10×10 output).
DEFAULT_FSC_PX_PER_CELL: int = 10


class DailyMosaicWriter:
    """Places per-cell 10×10 FSC patches into one daily COG (EPSG:32611).

    The mosaic grid is computed once from the union of all cell bounds at the FSC
    pixel size (cell side / ``fsc_px_per_cell`` metres). Every cell maps to an
    integer block within it.

    Args:
        grid: The inference grid cells (all in :data:`CELL_TARGET_CRS`).
        out_dir: Directory the daily COGs are written to.
        fsc_px_per_cell: FSC pixels per cell side (10 → 100 m px on a 1 km cell).

    Raises:
        ValueError: If ``grid`` is empty or cells are not all square / same size.
    """

    def __init__(
        self,
        *,
        grid: list[GridCell],
        out_dir: Path,
        fsc_px_per_cell: int = DEFAULT_FSC_PX_PER_CELL,
    ) -> None:
        if not grid:
            raise ValueError("DailyMosaicWriter requires a non-empty grid.")
        self.grid = grid
        self.out_dir = out_dir
        self.fsc_px_per_cell = fsc_px_per_cell

        bounds = [cell.polygon.bounds for cell in grid]  # (min_x, min_y, max_x, max_y)
        self._min_x = min(b[0] for b in bounds)
        self._min_y = min(b[1] for b in bounds)
        self._max_x = max(b[2] for b in bounds)
        self._max_y = max(b[3] for b in bounds)

        # FSC pixel size: cell side length / fsc_px_per_cell. Derived from the
        # first cell and asserted uniform so the lattice is regular.
        cell_w = bounds[0][2] - bounds[0][0]
        cell_h = bounds[0][3] - bounds[0][1]
        if not np.isclose(cell_w, cell_h):
            raise ValueError(f"Non-square cell {grid[0].cell_id}: {cell_w} x {cell_h}.")
        self._px_size = cell_w / fsc_px_per_cell

        width_m = self._max_x - self._min_x
        height_m = self._max_y - self._min_y
        self.width = int(round(width_m / self._px_size))
        self.height = int(round(height_m / self._px_size))
        # North-up transform: origin at the mosaic's top-left (min_x, max_y).
        self.transform: Affine = rasterio.transform.from_origin(
            self._min_x, self._max_y, self._px_size, self._px_size
        )

    def _cell_offset(self, cell: GridCell) -> tuple[int, int]:
        """Return the ``(row, col)`` of a cell's FSC block top-left in the mosaic.

        Computed from the cell's UTM top-left relative to the mosaic top-left,
        divided by the FSC pixel size. Cells lie on the 100 m lattice, so the
        result is an exact integer.
        """
        min_x, _min_y, _max_x, max_y = cell.polygon.bounds
        col = int(round((min_x - self._min_x) / self._px_size))
        row = int(round((self._max_y - max_y) / self._px_size))
        return row, col

    def write_day(
        self,
        day: datetime.date,
        fsc_by_cell: dict[int, npt.NDArray[np.floating] | None],
    ) -> Path:
        """Write one daily FSC COG; return its path.

        Args:
            day: The inference day (names the output ``fsc_YYYYMMDD.tif``).
            fsc_by_cell: Map ``cell_id -> 10×10 FSC array`` (or ``None`` for a
                cell with no valid prediction → left as ``nodata``).

        Returns:
            Path of the written COG.

        Raises:
            ValueError: If a provided FSC patch is not ``fsc_px_per_cell`` square.
            AssertionError: If two cells would write the same pixel (seam guard).
        """
        mosaic = np.full((self.height, self.width), float(NO_DATA_VALUE), dtype=np.float32)
        written = np.zeros((self.height, self.width), dtype=bool)
        n = self.fsc_px_per_cell

        for cell in self.grid:
            patch = fsc_by_cell.get(cell.cell_id)
            if patch is None:
                continue
            if patch.shape != (n, n):
                raise ValueError(
                    f"Cell {cell.cell_id} FSC patch shape {patch.shape}, expected {(n, n)}."
                )
            row, col = self._cell_offset(cell)
            target = np.s_[row : row + n, col : col + n]
            # Non-overlapping cells → disjoint blocks; double-write is a grid bug.
            assert not written[target].any(), (
                f"Seam overlap: cell {cell.cell_id} block at ({row},{col}) "
                "overwrites already-written pixels."
            )
            mosaic[target] = patch.astype(np.float32)
            written[target] = True

        total_cell_px = len(self.grid) * n * n
        valid_px = int(written.sum())
        coverage = valid_px / total_cell_px if total_cell_px else 0.0

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"fsc_{day.strftime('%Y%m%d')}.tif"
        with rasterio.open(
            out_path,
            "w",
            driver="COG",
            height=self.height,
            width=self.width,
            count=1,
            dtype="float32",
            crs=CELL_TARGET_CRS,
            transform=self.transform,
            nodata=NO_DATA_VALUE,
        ) as dst:
            dst.write(mosaic, 1)
            dst.update_tags(aoi_coverage_fraction=f"{coverage:.6f}")

        logger.info(
            "wrote_daily_fsc",
            day=day.isoformat(),
            path=str(out_path),
            cells=len(self.grid),
            valid_px=valid_px,
            aoi_coverage_fraction=round(coverage, 6),
        )
        return out_path
