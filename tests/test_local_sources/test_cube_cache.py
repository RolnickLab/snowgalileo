"""Cube-cache tests (TASK-003, SPEC FR-20 / AC-6).

Asserts the per-(modality, cell, day) ``.npz`` cache:
- round-trips an array (``put`` then ``get`` returns equal data),
- writes to the **per-cell shard** path ``cube_cache/{cell_id}/{day}_{modality}.npz``
  (flat layout would put ~300k files in one dir — ext4/xfs O(N) degradation,
  REVIEW_AUDIT #3),
- evicts by **day frontier in the parent only** (``prune_before_day``) — never per-``put``,
  never in a worker (PLAN-CUBE-CACHE-DAY-EVICTION),
- reports a miss as ``None`` (the exporter's signal to call the adapter).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from src.data.local_sources.cube_cache import (
    _VERSION_STAMP,
    CACHE_VERSION,
    CubeCache,
)

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


def test_put_does_not_evict(cache, tmp_path):
    """`put` never evicts, even past the cap — eviction is the parent prune's job alone.

    Filling beyond ``max_entries`` via ``put`` keeps every entry; the cross-process race
    is avoided by having no worker (``put`` caller) delete anything (PLAN-DAY-EVICTION).
    """
    arr = np.ones((1, 2, 2), dtype=np.float32)
    # cap = 3; insert 4 distinct keys — all must survive (put doesn't evict).
    keys = [
        ("s1", 0, date(2025, 4, 6)),
        ("s2", 0, date(2025, 4, 7)),
        ("modis", 1, date(2025, 4, 8)),
        ("era5", 1, date(2025, 4, 9)),
    ]
    for modality, cell_id, day in keys:
        cache.put(modality=modality, cell_id=cell_id, day=day, array=arr)
    for modality, cell_id, day in keys:
        assert cache.get(modality=modality, cell_id=cell_id, day=day) is not None
    assert len(list((tmp_path / "cube_cache").rglob("*.npz"))) == 4  # over cap, all kept


def test_put_same_key_twice_no_double_count(cache, tmp_path):
    """Re-putting an existing key overwrites, it does not grow the entry count."""
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for _ in range(5):
        cache.put(modality="s2", cell_id=0, day=DAY, array=arr)
    assert len(list((tmp_path / "cube_cache").rglob("*.npz"))) == 1
    assert cache.get(modality="s2", cell_id=0, day=DAY) is not None


def test_entries_survive_reopen(tmp_path):
    """A fresh CubeCache over a populated dir rebuilds its order from disk (scan recovery).

    The exporter may run in successive processes; ``__len__`` and the prune trigger must
    see the existing files rather than forgetting them.
    """
    root = tmp_path / "cube_cache"
    arr = np.ones((1, 2, 2), dtype=np.float32)
    c1 = CubeCache(root=root, max_entries=100)
    c1.put(modality="s1", cell_id=0, day=date(2025, 4, 6), array=arr)
    c1.put(modality="s2", cell_id=0, day=date(2025, 4, 7), array=arr)

    c2 = CubeCache(root=root, max_entries=100)
    assert len(c2) == 2  # recovered both entries from disk
    c2.put(modality="modis", cell_id=0, day=date(2025, 4, 8), array=arr)
    assert len(list(root.rglob("*.npz"))) == 3


# --- day-frontier eviction (PLAN-CUBE-CACHE-DAY-EVICTION) -------------------- #


def _fill_days(root, *, days, modalities=3, max_entries=100):
    """Seed one cache entry per (day, modality) and return the cache.

    ``days`` is an iterable of ``date``; each gets ``modalities`` entries under cell 0.
    """
    cache = CubeCache(root=root, max_entries=max_entries)
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for day in days:
        for m in range(modalities):
            cache.put(modality=f"m{m}", cell_id=0, day=day, array=arr)
    return cache


def test_prune_is_lazy_under_cap(tmp_path):
    """At/under the cap, prune_before_day removes nothing (Mode A behaviour-identical)."""
    root = tmp_path / "cube_cache"
    days = [DAY + timedelta(days=i) for i in range(10)]
    cache = _fill_days(root, days=days, modalities=3, max_entries=10_000)
    before = len(cache)
    removed = cache.prune_before_day(days[-1], window_days=7)
    assert removed == 0
    assert len(cache) == before


def test_prune_drops_dead_frontier_over_cap(tmp_path):
    """Over cap, prune drops entries with day < current - window; keeps the live window."""
    root = tmp_path / "cube_cache"
    # 20 days × 3 modalities = 60 entries; cap below that so prune fires.
    days = [DAY + timedelta(days=i) for i in range(20)]
    cache = _fill_days(root, days=days, modalities=3, max_entries=10)
    current = days[-1]  # DAY + 19
    removed = cache.prune_before_day(current, window_days=7)

    frontier = current - timedelta(days=7)  # DAY + 12
    # Everything strictly before the frontier is gone; frontier and later survive.
    for i in range(20):
        day = DAY + timedelta(days=i)
        present = cache.get(modality="m0", cell_id=0, day=day) is not None
        assert present == (day >= frontier), (i, day, frontier)
    assert removed == 3 * 12  # days DAY..DAY+11 (12 days) × 3 modalities


def test_prune_keeps_live_window_boundary(tmp_path):
    """The boundary day (current - window) is live and survives; one older is dropped."""
    root = tmp_path / "cube_cache"
    days = [DAY + timedelta(days=i) for i in range(20)]
    cache = _fill_days(root, days=days, modalities=2, max_entries=5)
    current = days[-1]
    cache.prune_before_day(current, window_days=7)

    boundary = current - timedelta(days=7)
    assert cache.get(modality="m0", cell_id=0, day=boundary) is not None
    assert cache.get(modality="m0", cell_id=0, day=boundary - timedelta(days=1)) is None


def test_prune_over_cap_after_keeps_live(tmp_path, caplog):
    """If the live window alone exceeds cap, no live entry is removed; a warning logs."""
    root = tmp_path / "cube_cache"
    # 8 live days (DAY+12..DAY+19) × 5 modalities = 40 live entries, cap = 5.
    days = [DAY + timedelta(days=i) for i in range(20)]
    cache = _fill_days(root, days=days, modalities=5, max_entries=5)
    current = days[-1]
    cache.prune_before_day(current, window_days=7)

    frontier = current - timedelta(days=7)
    # Every live entry (>= frontier) survives despite still being over cap.
    live_days = [d for d in days if d >= frontier]
    for d in live_days:
        assert cache.get(modality="m0", cell_id=0, day=d) is not None
    assert len(cache) > cache.max_entries  # deliberately still over cap, live kept


def test_prune_leaves_unparseable_files(tmp_path):
    """A stray non-conforming .npz is never deleted by a prune."""
    root = tmp_path / "cube_cache"
    days = [DAY + timedelta(days=i) for i in range(20)]
    cache = _fill_days(root, days=days, modalities=2, max_entries=5)
    stray = root / "0" / "garbage.npz"
    stray.write_bytes(b"not a real npz")

    cache.prune_before_day(days[-1], window_days=7)
    assert stray.exists()


def test_prune_removes_empty_shard_dirs(tmp_path):
    """After a prune empties a cell's shard dir, that dir is removed."""
    root = tmp_path / "cube_cache"
    arr = np.ones((1, 2, 2), dtype=np.float32)
    cache = CubeCache(root=root, max_entries=1)  # 2 entries > cap → prune fires
    # cell 0 holds only old days → fully pruned; cell 1 holds a live day → kept.
    old = DAY
    live = DAY + timedelta(days=30)
    cache.put(modality="m0", cell_id=0, day=old, array=arr)
    cache.put(modality="m0", cell_id=1, day=live, array=arr)

    cache.prune_before_day(live, window_days=7)
    assert not (root / "0").exists()  # emptied shard dir gone
    assert (root / "1").exists()


# --- invalidation: version stamp + overwrite (PLAN-CUBE-CACHE-INVALIDATION) -- #


def _seed(root, n=2):
    """Populate a cache dir with `n` distinct entries and return the cache."""
    cache = CubeCache(root=root, max_entries=100)
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for i in range(n):
        cache.put(modality="s2", cell_id=i, day=DAY, array=arr)
    return cache


def test_fresh_dir_writes_stamp(tmp_path):
    """A brand-new dir is stamped with CACHE_VERSION and nothing is cleared."""
    root = tmp_path / "cube_cache"
    CubeCache(root=root, max_entries=3)
    stamp = root / _VERSION_STAMP
    assert stamp.exists()
    assert int(stamp.read_text().strip()) == CACHE_VERSION


def test_matching_stamp_reuses_entries(tmp_path):
    """Same version on reopen → entries survive, count unchanged (no spurious clear)."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    reopened = CubeCache(root=root, max_entries=100)  # stamp matches CACHE_VERSION
    assert len(reopened) == 2
    assert reopened.get(modality="s2", cell_id=0, day=DAY) is not None


def test_version_mismatch_force_clears(tmp_path):
    """A stale stamp force-clears entries and rewrites the current stamp."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    (root / _VERSION_STAMP).write_text(f"{CACHE_VERSION + 1}\n")  # simulate old format

    reopened = CubeCache(root=root, max_entries=100)
    assert len(reopened) == 0
    assert not list(root.rglob("*.npz"))
    assert int((root / _VERSION_STAMP).read_text().strip()) == CACHE_VERSION


def test_version_mismatch_overrides_overwrite_false(tmp_path):
    """Version mismatch clears even with overwrite=False (it is unconditional)."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    (root / _VERSION_STAMP).write_text("999\n")
    reopened = CubeCache(root=root, max_entries=100, overwrite=False)
    assert len(reopened) == 0


def test_corrupt_stamp_treated_as_mismatch(tmp_path):
    """An unreadable/non-integer stamp is a mismatch → force-clear + rewrite."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    (root / _VERSION_STAMP).write_text("not-an-int\n")
    reopened = CubeCache(root=root, max_entries=100)
    assert len(reopened) == 0
    assert int((root / _VERSION_STAMP).read_text().strip()) == CACHE_VERSION


def test_overwrite_clears_and_rewrites_stamp(tmp_path):
    """overwrite=True clears existing entries and (re)writes the stamp."""
    root = tmp_path / "cube_cache"
    _seed(root, n=3)
    cleared = CubeCache(root=root, max_entries=100, overwrite=True)
    assert len(cleared) == 0
    assert not list(root.rglob("*.npz"))
    assert int((root / _VERSION_STAMP).read_text().strip()) == CACHE_VERSION


def test_overwrite_on_fresh_dir_just_stamps(tmp_path):
    """overwrite=True on an empty dir writes the stamp, has nothing to clear."""
    root = tmp_path / "cube_cache"
    cache = CubeCache(root=root, max_entries=3, overwrite=True)
    assert len(cache) == 0
    assert (root / _VERSION_STAMP).exists()


def test_stamp_not_counted_as_entry_and_survives_clear(tmp_path):
    """The stamp file is never a cache entry and persists across a clear."""
    root = tmp_path / "cube_cache"
    cache = _seed(root, n=2)
    assert len(cache) == 2  # stamp not counted among the 2 entries
    cache._clear()
    assert len(cache) == 0
    assert (root / _VERSION_STAMP).exists()
    # _scan_existing globs *.npz only → the stamp is invisible to it.
    assert _VERSION_STAMP not in cache._scan_existing()


def test_clear_removes_empty_shard_dirs(tmp_path):
    """_clear drops now-empty per-cell shard subdirs (leaves root + stamp)."""
    root = tmp_path / "cube_cache"
    cache = _seed(root, n=2)
    assert (root / "0").is_dir() and (root / "1").is_dir()
    cache._clear()
    assert not (root / "0").exists()
    assert not (root / "1").exists()
    assert root.is_dir()
