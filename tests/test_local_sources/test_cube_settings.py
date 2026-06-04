"""Cube-config tests (TASK-003 subtask 7, SPEC FR-6 / FR-20b).

Asserts ``CubeSettings``:
- loads the committed ``configs/bow_valley/cube.yaml``,
- points ``archive_root`` at the **clipped** archive (the raw path must appear
  only in the clip stage's config — FR-6),
- derives the Stage-2 subdirs (``cube_cache/``, ``cubes/``, ``daily_fsc/``,
  ``manifests/``, ``scratch/``) under ``processing_root`` (FR-20b),
- carries mode / window / crs / cache cap,
- is env-overridable (``CUBE_`` prefix) without editing the YAML.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.local_sources.settings import CubeSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
CUBE_YAML = REPO_ROOT / "configs" / "bow_valley" / "cube.yaml"


def test_committed_yaml_loads():
    """The committed cube.yaml parses into CubeSettings."""
    settings = CubeSettings.from_yaml(CUBE_YAML)
    assert settings.mode in {"A", "B"}
    assert settings.cell_crs == "EPSG:32611"


def test_archive_root_is_clipped_not_raw():
    """archive_root is the clipped archive; the raw path never appears (FR-6)."""
    settings = CubeSettings.from_yaml(CUBE_YAML)
    assert "clipped_bow_valley_selection_raw" in str(settings.archive_root)
    # The raw archive must not be the adapters' read root.
    assert settings.archive_root.name != "bow_valley_selection_raw"


def test_derived_subdirs_under_processing_root():
    """All Stage-2 subdirs derive from processing_root, each its own dir (FR-20b)."""
    settings = CubeSettings.from_yaml(CUBE_YAML)
    pr = settings.processing_root
    assert settings.cube_cache_dir == pr / "cube_cache"
    assert settings.cubes_dir == pr / "cubes"
    assert settings.daily_fsc_dir == pr / "daily_fsc"
    assert settings.manifests_dir == pr / "manifests"
    assert settings.scratch_dir == pr / "scratch"


def test_window_dates_parse():
    """Inference window parses to dates; end is not before start."""
    settings = CubeSettings.from_yaml(CUBE_YAML)
    assert isinstance(settings.window_start, date)
    assert isinstance(settings.window_end, date)
    assert settings.window_end >= settings.window_start


def test_cache_cap_positive():
    """The FIFO cache cap is a positive entry count."""
    settings = CubeSettings.from_yaml(CUBE_YAML)
    assert settings.cache_max_entries > 0


def test_env_override(monkeypatch, tmp_path):
    """CUBE_-prefixed env vars override YAML values without editing the file."""
    monkeypatch.setenv("CUBE_MODE", "B")
    monkeypatch.setenv("CUBE_CACHE_MAX_ENTRIES", "5")
    settings = CubeSettings.from_yaml(CUBE_YAML)
    assert settings.mode == "B"
    assert settings.cache_max_entries == 5


def test_invalid_mode_rejected(tmp_path):
    """A bad mode in YAML is rejected at load (no silent default)."""
    bad = tmp_path / "cube.yaml"
    bad.write_text("mode: Z\n")
    with pytest.raises(Exception):
        CubeSettings.from_yaml(bad)
