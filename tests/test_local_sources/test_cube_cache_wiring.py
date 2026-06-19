"""CubeCache wiring into the exporter (PLAN-CUBE-CACHE-WIRING, steps 1–2).

Two tiers:

* **Mechanism** (fast, archive-free): a counting stub adapter injected into a real
  ``LocalSourceExporter._dynamic`` exercises the exact ``_dynamic_block`` read-through/
  write-back path — hit/miss accounting, the day-grain memo, and the stale-shape guard —
  without touching the clipped archive.
* **Bit-identity** (slow, real archive): a full ``_assemble`` produces a byte-identical
  cube whether the cache is on or off. This is the load-bearing correctness gate — the
  cache must be a pure memo or it must not ship (PLAN §Risks).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from snow_galileo.data.local_sources.base import GridCell, LocalSourceAdapter, SpatialKind
from snow_galileo.data.local_sources.exporter import LocalSourceExporter

_ARCHIVE = Path("data/clipped_bow_valley_selection_raw")

requires_archive = pytest.mark.skipif(
    not _ARCHIVE.exists(), reason="clipped archive not present on this machine"
)


def _cell(cell_id: int = 0) -> GridCell:
    """A small real-CRS cell (10×10 px) — big enough for shape checks, cheap to alloc."""
    return GridCell.from_utm_bounds(
        cell_id=cell_id,
        min_x=560_000.0,
        min_y=5_650_000.0,
        max_x=560_100.0,
        max_y=5_650_100.0,
        px=10,
    )


class _CountingAdapter(LocalSourceAdapter):
    """A deterministic stub that records every ``fetch`` and returns a day-stamped block."""

    def __init__(self, *, bands: list[str], spatial_kind: SpatialKind) -> None:
        self.bands_out = bands
        self.spatial_kind = spatial_kind
        self.calls: list[date] = []

    def fetch(self, cell: GridCell, day: date | None) -> npt.NDArray[np.floating]:
        self.calls.append(day)  # type: ignore[arg-type]
        h, w = cell.shape
        # Value encodes the day ordinal so a wrong-day cache hit would be detectable.
        ordinal = day.toordinal() if day is not None else 0
        return np.full((len(self.bands_out), h, w), float(ordinal), dtype=np.float32)


def _exporter_with_stub(
    tmp_path: Path, *, cache: bool, adapter: _CountingAdapter
) -> LocalSourceExporter:
    """A real exporter (placeholder mode, no archive) whose dynamic list is one stub.

    Built in placeholder mode so ``__init__`` does no archive work, then the cache and
    the dynamic adapter list are overridden directly — this isolates ``_dynamic_block``
    from the 308-band assembly while still running the real exporter method.
    """
    exporter = LocalSourceExporter(
        out_dir=tmp_path / "cubes", placeholder=True, verify_s1_cache=False
    )
    exporter._dynamic = [adapter]
    exporter._cache = None
    if cache:
        from snow_galileo.data.local_sources.cube_cache import CubeCache

        exporter._cache = CubeCache(tmp_path / "cube_cache", max_entries=10_000)
    return exporter


def _block(exporter: LocalSourceExporter, adapter: _CountingAdapter, cell: GridCell, day: date):
    return exporter._dynamic_block(adapter, cell, day)


# --------------------------------------------------------------------------- #
# Mechanism (fast, archive-free)                                              #
# --------------------------------------------------------------------------- #
def test_cache_off_always_fetches(tmp_path: Path) -> None:
    """With no cache, every call re-fetches — byte-identical to the un-cached path."""
    adapter = _CountingAdapter(bands=["a", "b"], spatial_kind="time")
    exporter = _exporter_with_stub(tmp_path, cache=False, adapter=adapter)
    cell, day = _cell(), date(2025, 4, 6)

    first = _block(exporter, adapter, cell, day)
    second = _block(exporter, adapter, cell, day)

    assert len(adapter.calls) == 2  # no memo
    assert np.array_equal(first, second)
    assert not (tmp_path / "cube_cache").exists()  # nothing written


def test_cache_hit_skips_fetch(tmp_path: Path) -> None:
    """A second call for the same (modality, cell, day) is served from cache, no fetch."""
    adapter = _CountingAdapter(bands=["a", "b"], spatial_kind="time")
    exporter = _exporter_with_stub(tmp_path, cache=True, adapter=adapter)
    cell, day = _cell(), date(2025, 4, 6)

    cold = _block(exporter, adapter, cell, day)
    warm = _block(exporter, adapter, cell, day)

    assert len(adapter.calls) == 1  # second call hit the cache
    assert np.array_equal(cold, warm)


def test_overlapping_windows_reuse_shared_days(tmp_path: Path) -> None:
    """Two windows sharing days fetch each shared (cell, day) exactly once.

    Mirrors the real sweep: assembling day d then d+1 (8-day back-windows overlapping by
    7) must fetch the 7 shared days once, not twice — the whole point of the cache.
    """
    adapter = _CountingAdapter(bands=["a"], spatial_kind="low")
    exporter = _exporter_with_stub(tmp_path, cache=True, adapter=adapter)
    cell = _cell()
    window_a = [date(2025, 4, 6) + timedelta(days=i) for i in range(8)]
    window_b = [date(2025, 4, 7) + timedelta(days=i) for i in range(8)]  # shifts by 1

    for d in window_a:
        _block(exporter, adapter, cell, d)
    for d in window_b:
        _block(exporter, adapter, cell, d)

    # 8 distinct days in A + only the 1 new day in B (the other 7 are cached).
    assert sorted(set(adapter.calls)) == sorted(set(window_a) | set(window_b))
    assert len(adapter.calls) == 9  # 8 + 1, not 16


def test_stale_shape_hit_is_refetched(tmp_path: Path) -> None:
    """A cached entry with the wrong band count is discarded, re-fetched, and overwritten."""
    adapter = _CountingAdapter(bands=["a", "b", "c"], spatial_kind="high")
    exporter = _exporter_with_stub(tmp_path, cache=True, adapter=adapter)
    cell, day = _cell(), date(2025, 4, 6)
    tag = exporter._modality_tag(adapter)

    # Poison the cache with a wrong-shape (1-band) array for this key.
    h, w = cell.shape
    assert exporter._cache is not None
    exporter._cache.put(
        modality=tag, cell_id=cell.cell_id, day=day, array=np.zeros((1, h, w), np.float32)
    )

    block = _block(exporter, adapter, cell, day)

    assert block.shape[0] == 3  # not the poisoned 1-band shape
    assert len(adapter.calls) == 1  # the stale hit forced a real fetch
    fixed = exporter._cache.get(modality=tag, cell_id=cell.cell_id, day=day)
    assert fixed is not None and fixed.shape[0] == 3  # cache corrected in place


def test_modality_tags_unique_across_dynamic_slots() -> None:
    """Every real-mode dynamic slot maps to a distinct cache tag (no key collisions)."""
    exporter = LocalSourceExporter(
        out_dir=Path("/tmp/_tag_check"),
        placeholder=False,
        archive_root=_ARCHIVE if _ARCHIVE.exists() else Path("/tmp/_noarch"),
        verify_s1_cache=False,
    )
    tags = [exporter._modality_tag(a) for a in exporter._dynamic]
    assert len(tags) == len(set(tags)), f"tag collision among {tags}"


# --------------------------------------------------------------------------- #
# Bit-identity (slow, real archive) — the load-bearing correctness gate        #
# --------------------------------------------------------------------------- #
@requires_archive
@pytest.mark.slow
@pytest.mark.xdist_group("slow_archive")
def test_assembled_cube_identical_with_and_without_cache(tmp_path: Path) -> None:
    """``_assemble`` is byte-identical with the cache on (cold or warm) vs off.

    The cube feeds the model; the cache must be a pure memo. A divergence here means the
    cache is not transparent and must not ship (PLAN §Risks).
    """
    import pandas as pd
    from affine import Affine
    from shapely.geometry import box

    df = pd.read_csv("configs/bow_valley/cube_cells.csv")
    r = df.iloc[0]
    res = 10.0
    h = int(round((r.max_y - r.min_y) / res))
    w = int(round((r.max_x - r.min_x) / res))
    cell = GridCell(
        cell_id=0,
        crs=r.crs,
        transform=Affine(res, 0, r.min_x, 0, -res, r.max_y),
        shape=(h, w),
        polygon=box(r.min_x, r.min_y, r.max_x, r.max_y),
    )
    day = date(2025, 4, 6)

    no_cache = LocalSourceExporter(
        placeholder=False, archive_root=_ARCHIVE, verify_s1_cache=False
    )._assemble(cell, day)

    cached = LocalSourceExporter(
        placeholder=False,
        archive_root=_ARCHIVE,
        verify_s1_cache=False,
        cube_cache_dir=tmp_path / "cube_cache",
    )
    cold = cached._assemble(cell, day)
    warm = cached._assemble(cell, day)  # all hits

    assert np.array_equal(no_cache, cold, equal_nan=True)
    assert np.array_equal(cold, warm, equal_nan=True)
