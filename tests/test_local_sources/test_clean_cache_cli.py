"""`clean-cache` CLI command (PLAN-CUBE-CACHE-INVALIDATION test #7).

The command lives in ``export_bow_valley_cube.py`` (operator decision Q1) and wipes
``CubeSettings.cube_cache_dir`` on demand, reporting entries removed. It clears via a
single parent-process ``CubeCache(overwrite=True)`` — the only construction that clears —
so there is no cross-worker race.

The script is under ``scripts/developer_scripts/`` (no package ``__init__``), so it is
loaded by path with ``importlib`` rather than imported.
"""

from __future__ import annotations

import datetime
import importlib.util
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from snow_galileo.data.local_sources.cube_cache import CubeCache

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "developer_scripts"
    / "bow_valley_inference_local"
    / "export_bow_valley_cube.py"
)


def _load_cli():
    """Import the CLI script module by file path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("export_bow_valley_cube", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_cube_yaml(tmp_path: Path) -> Path:
    """Minimal cube.yaml pointing processing_root at a temp dir."""
    config = tmp_path / "cube.yaml"
    config.write_text(f"processing_root: {tmp_path / 'proc'}\n")
    return config


def test_clean_cache_empties_populated_cache(tmp_path, monkeypatch):
    """clean-cache wipes a populated cache dir and reports the entry count removed."""
    # Keep settings deterministic: ignore any developer env overrides.
    monkeypatch.delenv("CUBE_processing_root", raising=False)
    config = _write_cube_yaml(tmp_path)

    cache_root = tmp_path / "proc" / "cube_cache"
    cache = CubeCache(root=cache_root, max_entries=100)
    arr = np.ones((1, 2, 2), dtype=np.float32)
    for i in range(3):
        cache.put(modality="s2", cell_id=i, day=datetime.date(2025, 4, 6), array=arr)
    assert len(list(cache_root.rglob("*.npz"))) == 3

    module = _load_cli()
    result = CliRunner().invoke(module.app, ["clean-cache", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "Cleared 3 cube cache entries" in result.output
    assert not list(cache_root.rglob("*.npz"))


def test_clean_cache_on_empty_cache_reports_zero(tmp_path, monkeypatch):
    """clean-cache on an absent/empty cache reports zero and does not error."""
    monkeypatch.delenv("CUBE_processing_root", raising=False)
    config = _write_cube_yaml(tmp_path)

    module = _load_cli()
    result = CliRunner().invoke(module.app, ["clean-cache", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "Cleared 0 cube cache entries" in result.output


def test_export_prompt_nonempty_non_tty_aborts(tmp_path, monkeypatch):
    """`export --cache-policy prompt` on a non-empty cache (non-TTY) aborts before work.

    CliRunner provides a non-TTY stdin, so the prompt policy must error out with the
    actionable --cache-policy message rather than hang or silently reuse — and it must
    fail *before* any grid build / export happens.
    """
    monkeypatch.delenv("CUBE_processing_root", raising=False)
    config = _write_cube_yaml(tmp_path)

    cache_root = tmp_path / "proc" / "cube_cache"
    cache = CubeCache(root=cache_root, max_entries=100)
    cache.put(
        modality="s2",
        cell_id=0,
        day=datetime.date(2025, 4, 6),
        array=np.ones((1, 2, 2), dtype=np.float32),
    )

    module = _load_cli()
    result = CliRunner().invoke(
        module.app, ["export", "--config", str(config), "--cache-policy", "prompt"]
    )

    # Contract: a BadParameter abort (exit 2), not a crash (1) or hang. Don't assert the
    # rich-rendered message text — Typer renders BadParameter through a rich panel whose
    # width/version-dependent wrapping can split tokens (e.g. "--cache-policy") across lines
    # or boxes, which made the old substring check flaky across CI vs local. The exit code is
    # the stable signal: 2 is Click's usage/parameter error.
    assert result.exit_code == 2, result.output
    # Cache untouched (the run aborted before any clear/export work).
    assert len(list(cache_root.rglob("*.npz"))) == 1
