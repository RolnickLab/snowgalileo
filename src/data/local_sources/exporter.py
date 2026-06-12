"""The local-source cube exporter — assembles the canonical 308-band cube.

:class:`LocalSourceExporter` is the business-logic core of the direct-source
pipeline (PLAN §4). It depends only on the
:class:`~src.data.local_sources.base.LocalSourceAdapter` *port*, never on a
concrete adapter, so swapping a placeholder (TASK-004) for a real adapter
(TASK-006…TASK-014) does not touch this module.

Per ``(cell, window_end)`` it:

1. Derives the 8-day window ``[window_end - 7d … window_end]`` (one day per
   timestep, ``DAYS_PER_TIMESTEP = 1``).
2. For each day, calls the five **dynamic** adapters in canonical order and
   concatenates their outputs into that day's 38-band block.
3. Stacks the eight day-blocks (304 bands), then appends the four **static**
   bands once (DEM/slope/aspect/Map) → **308 bands** in
   ``create_ee_image`` order (dynamic × T, then static).
4. Writes a multiband ``float32`` GeoTIFF on the cell's **EPSG:32611** target
   grid (``-9999`` nodata) under the ``PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif``
   filename the unchanged loader parses (``layout.build_cube_filename``).

The band order and filename are the fixed contract; they live in
:mod:`src.data.local_sources.layout` and are never retyped here.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
import structlog
from pyproj import Transformer

from src.data.config import DAYS_PER_TIMESTEP, NO_DATA_VALUE, NUM_TIMESTEPS
from src.data.local_sources.base import (
    CELL_TARGET_CRS,
    GridCell,
    LocalSourceAdapter,
)
from src.data.local_sources.layout import (
    TOTAL_BANDS,
    build_cube_filename,
    full_band_order,
)
from src.data.local_sources.placeholder import (
    PlaceholderAdapter,
    dynamic_adapters,
    static_adapters,
)
from src.data.local_sources.settings import CubeSettings

logger = structlog.get_logger(__name__)

#: ``EPSG:4326`` is the geographic CRS the filename's lat/lon are expressed in —
#: the loader feeds ``parts[2]/parts[3]`` to ``to_cartesian`` which asserts the
#: ±90/±180 degree range, so the filename carries the cell *centre* in degrees
#: (a separate channel from the UTM pixel grid).
_GEOGRAPHIC_CRS: str = "EPSG:4326"


class LocalSourceExporter:
    """Assembles and writes the canonical 308-band cube for one ``(cell, day)``.

    Args:
        out_dir: Directory the cube tif is written to. Defaults to
            :pyattr:`CubeSettings.cubes_dir` so production runs land in
            ``data/bow_valley_processing/cubes/``.
        placeholder: When ``True``, every band is the all-``-9999`` placeholder
            (the deterministic tracer mode). When ``False``, real adapters are
            substituted for the bands implemented so far (WorldCover → ``Map``,
            TASK-006); the remaining bands stay placeholders until their tasks land.
        archive_root: The clipped archive the real adapters read. Defaults to
            :pyattr:`CubeSettings.archive_root`.
    """

    def __init__(
        self,
        *,
        out_dir: Path | None = None,
        placeholder: bool = True,
        archive_root: Path | None = None,
        auto_build_s1_cache: bool = True,
    ) -> None:
        settings = CubeSettings()
        self.out_dir = out_dir if out_dir is not None else settings.cubes_dir
        self.archive_root = archive_root if archive_root is not None else settings.archive_root
        self.placeholder = placeholder
        # In real mode, guarantee the S1 SNAP cache covers each cell's window before
        # assembly (see _ensure_s1_cache) — never silently emit an all-(-9999) S1 block.
        self.auto_build_s1_cache = auto_build_s1_cache
        # The clipped S1 SAFE archive + the SNAP dB+angle cache the S1Adapter reads.
        self.s1_archive_root = self.archive_root / "sentinel1"
        self.s1_cache_dir = self.archive_root / "sentinel1_snap"
        self._dynamic: list[LocalSourceAdapter] = self._build_dynamic_adapters()
        self._static: list[LocalSourceAdapter] = self._build_static_adapters()

    def _build_dynamic_adapters(self) -> list[LocalSourceAdapter]:
        """Dynamic-modality adapters in canonical band order.

        In placeholder mode all five groups are placeholders. In real mode each real
        adapter that owns a **contiguous slice** of a group's bands replaces that slice,
        and the unclaimed bands stay placeholders — a single group may be tiled by
        several reals while preserving band order. Wired so far:

        - **HIGH** group (``VV,VH,angle,B2,B3,B4,B8,B11,B12,B2_landsat..B7_landsat``) →
          S1 (TASK-014) for ``VV,VH,angle`` + S2 (TASK-013) for ``B2..B12`` + Landsat
          (TASK-012) for ``B2_landsat..B7_landsat``.
        - **TIME** group (``M5,M7,M10,M11`` + ``skin..v``) → VIIRS-coarse (TASK-010) head
          + ERA5 (TASK-008) tail.
        - **LOW** group (``sur_refl_b01..b07`` + ``I1,I3``) → MODIS (TASK-009) head +
          VIIRS-fine (TASK-010) tail.
        - **MED** group (``Oa17_radiance,Oa21_radiance``) → S3 OLCI (TASK-011), the whole group.
        - **CLOUD** group (``state_1km,QA60,QA_PIXEL``) → MODIS ``state_1km`` (TASK-009)
          + S2 ``QA60`` (TASK-013c, reconstructed from MSK_CLASSI) + Landsat ``QA_PIXEL``
          (TASK-012).
        """
        adapters: list[LocalSourceAdapter] = list(dynamic_adapters())
        if self.placeholder:
            return adapters

        from src.data.local_sources.era5 import Era5Adapter
        from src.data.local_sources.landsat import LandsatAdapter, LandsatCloudAdapter
        from src.data.local_sources.modis import ModisAdapter, ModisCloudAdapter
        from src.data.local_sources.s1 import S1Adapter
        from src.data.local_sources.s2 import S2Adapter, S2CloudAdapter
        from src.data.local_sources.s3 import S3Adapter
        from src.data.local_sources.viirs import ViirsCoarseAdapter, ViirsFineAdapter

        modis_root = self.archive_root / "modis"
        viirs_root = self.archive_root / "viirs"
        landsat9_root = self.archive_root / "landsat9"
        landsat8_root = self.archive_root / "landsat8"
        reals: list[LocalSourceAdapter] = [
            # S1 reads the SNAP dB+angle cache (built once, offline, by s1_snap.py),
            # NOT the raw clipped SAFEs — see src/data/local_sources/s1.py.
            S1Adapter(cache_root=self.s1_cache_dir),
            S2Adapter(archive_root=self.archive_root / "sentinel2"),
            S2CloudAdapter(archive_root=self.archive_root / "sentinel2"),
            LandsatAdapter(landsat9_root=landsat9_root, landsat8_root=landsat8_root),
            LandsatCloudAdapter(landsat9_root=landsat9_root, landsat8_root=landsat8_root),
            Era5Adapter(archive_root=self.archive_root / "era5"),
            ModisAdapter(archive_root=modis_root),
            ModisCloudAdapter(archive_root=modis_root),
            ViirsFineAdapter(archive_root=viirs_root),
            ViirsCoarseAdapter(archive_root=viirs_root),
            S3Adapter(archive_root=self.archive_root / "sentinel3"),
        ]

        rebuilt: list[LocalSourceAdapter] = []
        for adapter in adapters:
            rebuilt.extend(self._split_group(adapter, reals))
        return rebuilt

    @staticmethod
    def _split_group(
        group: LocalSourceAdapter, reals: list[LocalSourceAdapter]
    ) -> list[LocalSourceAdapter]:
        """Tile ``group``'s band order with any real adapters owning a contiguous slice.

        Walks ``group.bands_out`` left to right: at each position, emits the real adapter
        whose ``bands_out`` begins there (verifying it is a contiguous slice), otherwise
        accumulates the band into a trailing placeholder. The concatenated result equals
        ``group.bands_out`` exactly. A group may be claimed by several reals (e.g. MODIS
        head + VIIRS-fine tail of the LOW group).
        """
        bands = group.bands_out
        by_first = {r.bands_out[0]: r for r in reals if set(r.bands_out).issubset(bands)}

        out: list[LocalSourceAdapter] = []
        pending: list[str] = []

        def flush() -> None:
            if pending:
                out.append(
                    PlaceholderAdapter(bands_out=list(pending), spatial_kind=group.spatial_kind)
                )
                pending.clear()

        i = 0
        while i < len(bands):
            real = by_first.get(bands[i])
            if real is not None:
                rb = real.bands_out
                if bands[i : i + len(rb)] != rb:
                    raise AssertionError(
                        f"{type(real).__name__} bands are not a contiguous slice of "
                        f"{bands} — band-layout contract broken."
                    )
                flush()
                out.append(real)
                i += len(rb)
            else:
                pending.append(bands[i])
                i += 1
        flush()
        return out

    def _build_static_adapters(self) -> list[LocalSourceAdapter]:
        """Per-static-band adapters in ``STATIC_BANDS`` order.

        In placeholder mode all are placeholders. In real mode, the ``Map`` band
        is the real :class:`~src.data.local_sources.worldcover.WorldCoverAdapter`
        (TASK-006), and ``DEM``/``slope``/``aspect`` are served by the single
        :class:`~src.data.local_sources.dem.DemAdapter` (TASK-007), which emits all
        three terrain bands together.
        """
        adapters: list[LocalSourceAdapter] = list(static_adapters())
        if not self.placeholder:
            from src.data.local_sources.dem import DemAdapter
            from src.data.local_sources.worldcover import WorldCoverAdapter

            wc = WorldCoverAdapter(archive_root=self.archive_root / "worldcover")
            dem = DemAdapter(archive_root=self.archive_root / "dem")
            # The DEM adapter emits [DEM, slope, aspect] as one 3-band block, so it
            # replaces the DEM placeholder and the slope/aspect placeholders are dropped.
            rebuilt: list[LocalSourceAdapter] = []
            for adapter in adapters:
                if adapter.bands_out == ["Map"]:
                    rebuilt.append(wc)
                elif adapter.bands_out == ["DEM"]:
                    rebuilt.append(dem)
                elif adapter.bands_out in (["slope"], ["aspect"]):
                    continue  # subsumed by the DEM adapter's 3-band output
                else:
                    rebuilt.append(adapter)
            adapters = rebuilt
        return adapters

    def _window_days(self, window_end: datetime.date) -> list[datetime.date]:
        """Return the 8 window days ascending, ending at ``window_end``."""
        return [
            window_end - datetime.timedelta(days=DAYS_PER_TIMESTEP * offset)
            for offset in reversed(range(NUM_TIMESTEPS))
        ]

    def _cell_centre_lat_lon(self, cell: GridCell) -> tuple[float, float]:
        """Reproject the cell centre from its UTM CRS to ``EPSG:4326`` degrees.

        Returns:
            ``(lat, lon)`` of the cell centre, rounded to 4 decimals so the
            filename stays compact while matching the FR-18 regex.
        """
        transformer = Transformer.from_crs(cell.crs, _GEOGRAPHIC_CRS, always_xy=True)
        centre_x, centre_y = cell.polygon.centroid.x, cell.polygon.centroid.y
        lon, lat = transformer.transform(centre_x, centre_y)
        return round(lat, 4), round(lon, 4)

    def _ensure_s1_cache(self, cell: GridCell, window_end: datetime.date) -> None:
        """Pre-flight: guarantee the S1 cache covers this cell's window before assembly.

        Real mode only. Builds any missing-but-coverable ``(granule, cell)`` cache tifs
        (or raises if SNAP/SAFEs are unavailable), so the S1 adapter never silently falls
        back to an all-``-9999`` block. A cell with genuinely no S1 in its window needs
        nothing built and is left S1-free — see :func:`s1_snap.ensure_s1_cache`.
        """
        if self.placeholder or not self.auto_build_s1_cache:
            return
        from src.data.local_sources.s1_snap import ensure_s1_cache

        ensure_s1_cache(
            archive_root=self.s1_archive_root,
            cells=[cell],
            cache_dir=self.s1_cache_dir,
            window_days=self._window_days(window_end),
        )

    def _assemble(
        self,
        cell: GridCell,
        window_end: datetime.date,
    ) -> npt.NDArray[np.float32]:
        """Build the ``(308, H, W)`` band stack in canonical order.

        Args:
            cell: Target grid cell.
            window_end: The window-end (prediction) day.

        Returns:
            The assembled cube as a ``(TOTAL_BANDS, *cell.shape)`` ``float32`` array.

        Raises:
            AssertionError: If the assembled band count is not :data:`TOTAL_BANDS`.
        """
        blocks: list[npt.NDArray] = []
        for day in self._window_days(window_end):
            for adapter in self._dynamic:
                blocks.append(adapter.fetch(cell, day))
        # Static layers are time-invariant: day is ignored (passed None).
        for adapter in self._static:
            blocks.append(adapter.fetch(cell, None))

        cube = np.concatenate(blocks, axis=0).astype(np.float32)
        assert cube.shape[0] == TOTAL_BANDS, (
            f"Assembled {cube.shape[0]} bands, expected {TOTAL_BANDS}."
        )
        return cube

    def export(self, *, cell: GridCell, window_end: datetime.date) -> Path:
        """Assemble and write one cube tif; return its path.

        Args:
            cell: Target grid cell (supplies CRS, transform, shape).
            window_end: The 8-day window's end (prediction) day.

        Returns:
            The path of the written ``PR_*.tif``.
        """
        self._ensure_s1_cache(cell, window_end)
        cube = self._assemble(cell, window_end)
        lat, lon = self._cell_centre_lat_lon(cell)
        filename = build_cube_filename(window_end=window_end, lat=lat, lon=lon)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / filename

        height, width = cell.shape
        band_names = full_band_order()
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=TOTAL_BANDS,
            dtype="float32",
            crs=CELL_TARGET_CRS,
            transform=cell.transform,
            nodata=NO_DATA_VALUE,
        ) as dst:
            dst.write(cube)
            for index, name in enumerate(band_names, start=1):
                dst.set_band_description(index, name)

        logger.info(
            "exported_cube",
            cell_id=cell.cell_id,
            window_end=window_end.isoformat(),
            bands=TOTAL_BANDS,
            path=str(out_path),
            placeholder=self.placeholder,
        )
        return out_path


def _main() -> None:
    """Minimal CLI for the TASK-004 verification commands (Section 6).

    ``python -m src.data.local_sources.exporter --cell 0 --window-end 2025-04-06
    --placeholder`` builds one placeholder cube using a Bow Valley UTM cell.
    """
    import argparse

    from src.data.local_sources.grid import build_grid

    parser = argparse.ArgumentParser(description="Export one placeholder cube.")
    parser.add_argument("--cell", type=int, default=0, help="Grid cell index.")
    parser.add_argument(
        "--window-end",
        type=datetime.date.fromisoformat,
        required=True,
        help="Window-end day (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--placeholder", action="store_true", help="Use placeholder adapters (required)."
    )
    args = parser.parse_args()

    cells = build_grid()
    cell = cells[args.cell]
    exporter = LocalSourceExporter(placeholder=True)
    path = exporter.export(cell=cell, window_end=args.window_end)
    print(path)


if __name__ == "__main__":
    _main()
