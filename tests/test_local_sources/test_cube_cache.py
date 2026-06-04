"""Cube-cache tests (TASK-003, SPEC FR-20 / AC-6).

Asserts the per-(modality, cell, day) ``.npz`` cache:
- round-trips an array (``put`` then ``get`` returns equal data),
- writes to the **per-cell shard** path ``cube_cache/{cell_id}/{day}_{modality}.npz``
  (flat layout would put ~300k files in one dir — ext4/xfs O(N) degradation,
  REVIEW_AUDIT #3),
- evicts **FIFO** when the configurable entry cap is exceeded,
- reports a miss as ``None`` (the exporter's signal to call the adapter).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from src.data.local_sources.cube_cache import CubeCache

DAY = date(2025, 4, 6)


@pytest.fixture
def cache(tmp_path):
    return CubeCache(root=tmp_path / "cube_cache", max_entries=3)


def test_miss_returns_none(cache):
    """A key never written is a miss (None) — the exporter's call-adapter signal."""
    assert cache.get(modality="s2", cell_id=0, day=DAY) is None


def test_put_get_roundtrip(cache):
    """`put` then `get` returns array-equal data (AC-6)."""
    arr = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    cache.put(modality="s2", cell_id=7, day=DAY, array=arr)
    out = cache.get(modality="s2", cell_id=7, day=DAY)
    assert out is not None
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == arr.dtype


def test_shard_path_is_per_cell(cache, tmp_path):
    """Files land at cube_cache/{cell_id}/{day}_{modality}.npz (AC-6, FR-20)."""
    arr = np.zeros((1, 2, 2), dtype=np.float32)
    cache.put(modality="modis", cell_id=42, day=DAY, array=arr)
    expected = tmp_path / "cube_cache" / "42" / "20250406_modis.npz"
    assert expected.exists(), sorted((tmp_path / "cube_cache").rglob("*"))


def test_fifo_eviction_respects_cap(cache, tmp_path):
    """When entries exceed the cap, the oldest-written entry is evicted (AC-6)."""
    arr = np.ones((1, 2, 2), dtype=np.float32)
    # cap = 3; insert 4 distinct keys in known order.
    keys = [
        ("s1", 0, date(2025, 4, 6)),
        ("s2", 0, date(2025, 4, 7)),
        ("modis", 1, date(2025, 4, 8)),
        ("era5", 1, date(2025, 4, 9)),  # this insert evicts the first (s1/0/4-6)
    ]
    for modality, cell_id, day in keys:
        cache.put(modality=modality, cell_id=cell_id, day=day, array=arr * cell_id)

    # Oldest (s1, 0, 4-6) evicted; the rest survive.
    assert cache.get(modality="s1", cell_id=0, day=date(2025, 4, 6)) is None
    for modality, cell_id, day in keys[1:]:
        assert cache.get(modality=modality, cell_id=cell_id, day=day) is not None
    # Total .npz files on disk never exceeds the cap.
    assert len(list((tmp_path / "cube_cache").rglob("*.npz"))) == 3


def test_put_same_key_twice_no_double_count(cache, tmp_path):
    """Re-putting an existing key overwrites, it does not grow the entry count."""
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for _ in range(5):
        cache.put(modality="s2", cell_id=0, day=DAY, array=arr)
    assert len(list((tmp_path / "cube_cache").rglob("*.npz"))) == 1
    assert cache.get(modality="s2", cell_id=0, day=DAY) is not None


def test_eviction_survives_reopen(tmp_path):
    """A fresh CubeCache over a populated dir rebuilds its FIFO order from disk.

    The exporter may run in successive processes; the cap must hold across
    re-instantiation rather than resetting and overflowing the directory.
    """
    root = tmp_path / "cube_cache"
    arr = np.ones((1, 2, 2), dtype=np.float32)
    c1 = CubeCache(root=root, max_entries=2)
    c1.put(modality="s1", cell_id=0, day=date(2025, 4, 6), array=arr)
    c1.put(modality="s2", cell_id=0, day=date(2025, 4, 7), array=arr)

    c2 = CubeCache(root=root, max_entries=2)
    c2.put(modality="modis", cell_id=0, day=date(2025, 4, 8), array=arr)
    # cap still 2 after reopen → exactly 2 files, oldest evicted.
    assert len(list(root.rglob("*.npz"))) == 2
