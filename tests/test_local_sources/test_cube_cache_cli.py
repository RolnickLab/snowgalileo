"""Cache-policy resolution (PLAN-CUBE-CACHE-INVALIDATION §CLI, test #6).

`resolve_cache_policy` runs once in the parent process and clears the cache **at most
once** (never in a worker). Verifies the reuse/overwrite/prompt mapping, the no-TTY error
that prevents silent staleness, and that the exporter forwards ``overwrite_cache``.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from src.data.local_sources.cube_cache import _VERSION_STAMP, CubeCache
from src.data.local_sources.cube_cache_cli import (
    CachePolicy,
    CachePolicyError,
    resolve_cache_policy,
)

DAY = datetime.date(2025, 4, 6)


def _seed(root, n=2):
    cache = CubeCache(root=root, max_entries=100)
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for i in range(n):
        cache.put(modality="s2", cell_id=i, day=DAY, array=arr)
    return cache


def test_reuse_keeps_entries(tmp_path):
    root = tmp_path / "cube_cache"
    _seed(root, n=3)
    resolve_cache_policy(root=root, policy=CachePolicy.REUSE)
    assert len(list(root.rglob("*.npz"))) == 3


def test_overwrite_clears_once(tmp_path):
    root = tmp_path / "cube_cache"
    _seed(root, n=3)
    resolve_cache_policy(root=root, policy=CachePolicy.OVERWRITE)
    assert not list(root.rglob("*.npz"))
    assert (root / _VERSION_STAMP).exists()


def test_prompt_empty_dir_proceeds_silently(tmp_path):
    """Prompt + absent/empty cache → no question, no clear, no error (nothing to lose)."""
    root = tmp_path / "cube_cache"  # never created
    resolve_cache_policy(root=root, policy=CachePolicy.PROMPT, is_tty=False)
    # No error despite non-TTY, because there is nothing to reuse.


def test_prompt_nonempty_non_tty_errors(tmp_path):
    """Prompt + non-empty + no TTY → actionable error, never a silent reuse."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    with pytest.raises(CachePolicyError, match="--cache-policy"):
        resolve_cache_policy(root=root, policy=CachePolicy.PROMPT, is_tty=False)
    # The cache is untouched — the run is expected to abort and be re-invoked.
    assert len(list(root.rglob("*.npz"))) == 2


def test_prompt_tty_overwrite_answer_clears(tmp_path, monkeypatch):
    """Prompt + TTY + operator answers 'o' → clears."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    monkeypatch.setattr("builtins.input", lambda *_: "o")
    resolve_cache_policy(root=root, policy=CachePolicy.PROMPT, is_tty=True)
    assert not list(root.rglob("*.npz"))


def test_prompt_tty_reuse_answer_keeps(tmp_path, monkeypatch):
    """Prompt + TTY + operator answers 'r' → keeps."""
    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    monkeypatch.setattr("builtins.input", lambda *_: "reuse")
    resolve_cache_policy(root=root, policy=CachePolicy.PROMPT, is_tty=True)
    assert len(list(root.rglob("*.npz"))) == 2


def test_exporter_forwards_overwrite_cache(tmp_path):
    """LocalSourceExporter(overwrite_cache=True) clears the dir on construction."""
    from src.data.local_sources.exporter import LocalSourceExporter

    root = tmp_path / "cube_cache"
    _seed(root, n=2)
    exporter = LocalSourceExporter(
        out_dir=tmp_path / "cubes",
        placeholder=False,
        cube_cache_dir=root,
        overwrite_cache=True,
    )
    assert exporter._cache is not None
    assert len(exporter._cache) == 0
    assert not list(root.rglob("*.npz"))
