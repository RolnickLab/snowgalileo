"""ESA WorldCover adapter — the static ``Map`` land-cover band (TASK-006).

Replaces the WorldCover placeholder with the real v200 (2021) ``Map`` band on the
cell grid. WorldCover is **static** (ignores ``day``) and **categorical**: class
codes are preserved exactly and resampled **nearest-neighbour** (bilinear would
invent non-existent class codes). The single ``Map`` band is emitted as-is — the
downstream loader one-hot-encodes it (``landsat_eval.py``); the adapter must
**not** one-hot or remap, or the loader's one-hot channel order breaks.

CRS: the clipped source tiles are ``EPSG:4326``; the adapter NN-reprojects them
onto the :class:`~src.data.local_sources.base.GridCell`'s ``EPSG:32611`` (UTM 11N)
grid via the shared :func:`~src.data.local_sources.base.reproject_to_cell`, so the
assembled cube is single-CRS like every other band.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import box

from src.data.local_sources.base import GridCell, LocalSourceAdapter, reproject_to_cell

logger = structlog.get_logger(__name__)

#: WorldCover source CRS (the clipped tiles are geographic).
_SOURCE_CRS: str = "EPSG:4326"

#: Margin (degrees) added around the cell when selecting/cropping source tiles.
_TILE_MARGIN_DEG: float = 0.02


class WorldCoverAdapter(LocalSourceAdapter):
    """Static ESA WorldCover ``Map`` adapter (categorical, NN, ``day``-independent).

    Args:
        archive_root: The clipped WorldCover archive root (holds ``*_Map.tif`` tiles).
    """

    bands_out = ["Map"]
    spatial_kind = "space"
    native_fill = None

    def __init__(self, *, archive_root: Path) -> None:
        self.archive_root = archive_root

    def _tiles(self) -> list[Path]:
        """Return all clipped WorldCover ``Map`` GeoTIFF tiles in the archive."""
        tiles = sorted(self.archive_root.rglob("*_Map.tif"))
        if not tiles:
            tiles = sorted(self.archive_root.rglob("*.tif"))
        if not tiles:
            raise FileNotFoundError(f"No WorldCover tiles under {self.archive_root}")
        return tiles

    def _mosaic_for_cell(
        self, cell: GridCell
    ) -> tuple[npt.NDArray[np.floating], rasterio.Affine, str]:
        """Mosaic the source tiles overlapping the cell (in the 4326 source CRS).

        Args:
            cell: Target grid cell (its UTM polygon is reprojected to 4326 to pick tiles).

        Returns:
            ``(mosaic, transform, crs)`` — the cropped 4326 mosaic covering the cell.
        """
        # Cell footprint in the source CRS, with a small margin so NN has neighbours
        # right at the edges.
        b = cell.polygon.bounds  # UTM metres
        lon0, lat0, lon1, lat1 = transform_bounds(cell.crs, _SOURCE_CRS, *b)
        m = _TILE_MARGIN_DEG
        want = box(lon0 - m, lat0 - m, lon1 + m, lat1 + m)

        srcs = []
        for tile in self._tiles():
            src = rasterio.open(tile)
            if box(*src.bounds).intersects(want):
                srcs.append(src)
            else:
                src.close()
        if not srcs:
            raise FileNotFoundError(
                f"No WorldCover tile intersects cell {cell.cell_id} ({want.bounds})."
            )
        try:
            mosaic, transform = merge(srcs, bounds=want.bounds)
            crs = srcs[0].crs.to_string()
        finally:
            for src in srcs:
                src.close()
        return mosaic[0].astype(np.float64), transform, crs

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray[np.floating]:
        """Return the WorldCover ``Map`` band on the cell grid (``day`` ignored).

        Args:
            cell: Target :class:`GridCell` (supplies the UTM ``crs``/``transform``/``shape``).
            day: Ignored — WorldCover is a static layer.

        Returns:
            ``(1, H, W)`` ``float32`` array of class codes on the cell's UTM grid,
            nearest-neighbour resampled (codes preserved, never one-hot).
        """
        mosaic, src_transform, src_crs = self._mosaic_for_cell(cell)
        reprojected = reproject_to_cell(
            source=mosaic[np.newaxis, :, :],
            src_transform=src_transform,
            src_crs=src_crs,
            cell=cell,
            categorical=True,  # NN — never invent class codes
        )
        logger.info(
            "worldcover_fetch",
            cell_id=cell.cell_id,
            codes=sorted(np.unique(reprojected).astype(int).tolist())[:12],
        )
        return reprojected
