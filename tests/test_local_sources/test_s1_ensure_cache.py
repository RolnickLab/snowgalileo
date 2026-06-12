"""Tests for the S1 cache pre-flight ``ensure_s1_cache`` (silent-dropout guard).

``ensure_s1_cache`` decides which ``(granule, cell)`` SNAP cache tifs a cube export needs
and either builds the missing ones or fails loudly — so a cube is never silently assembled
with an all-``-9999`` S1 block when S1 data *was* available. These tests exercise the
decision logic without SNAP or real SAFEs: the per-granule footprint and the SNAP builder
are monkeypatched, so only the needed/missing/coverage branching is under test.
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

import src.data.local_sources.s1_snap as snap
from src.data.local_sources.base import GridCell
from src.data.local_sources.s1_snap import (
    S1CacheUnavailableError,
    cache_tif_name,
    ensure_s1_cache,
)

# Two acquisition days; the export window covers both.
_DAY_A = datetime.date(2025, 5, 12)
_DAY_B = datetime.date(2025, 5, 17)
_WINDOW = [_DAY_A, _DAY_B]


def _cell() -> GridCell:
    # A small UTM 11N cell; its 4326 bbox is what footprints are tested against.
    return GridCell.from_utm_bounds(
        cell_id=73, min_x=563000.0, min_y=5653000.0, max_x=563200.0, max_y=5653200.0, px=20
    )


def _granule(archive: Path, acq: datetime.date, uid: str) -> Path:
    """Create an empty placeholder raw-granule .zip whose stem parses to ``acq``."""
    archive.mkdir(parents=True, exist_ok=True)
    stem = f"S1C_IW_GRDH_1SDV_{acq:%Y%m%d}T013724_{acq:%Y%m%d}T013749_001664_002BB2_{uid}"
    path = archive / f"{stem}.zip"
    path.write_bytes(b"")  # never opened — footprint + builder are monkeypatched
    return path


@pytest.fixture()
def covering_footprint(monkeypatch: pytest.MonkeyPatch) -> Polygon:
    """Make every granule footprint a polygon that covers any cell (always intersects)."""
    big = box(-180.0, -90.0, 180.0, 90.0)
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: big)
    return big


@pytest.fixture()
def record_builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[int]]]:
    """Record build_granule_cache calls instead of running SNAP."""
    calls: list[tuple[str, list[int]]] = []

    def _fake(*, granule_zip: Path, cells: list[GridCell], cache_dir, gpt, graph):  # type: ignore[no-untyped-def]
        calls.append((granule_zip.stem, [c.cell_id for c in cells]))
        return []

    monkeypatch.setattr(snap, "build_granule_cache", _fake)
    return calls


def _present_gpt(tmp_path: Path) -> Path:
    gpt = tmp_path / "gpt"
    gpt.write_text("#!/bin/sh\n")
    return gpt


def test_builds_only_missing_covered_tifs(
    tmp_path: Path, covering_footprint: Polygon, record_builds: list
) -> None:
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    g_a = _granule(archive, _DAY_A, "AAAA")
    g_b = _granule(archive, _DAY_B, "BBBB")
    # Pre-seed the DAY_A cache tif so only DAY_B is missing.
    (cache / cache_tif_name(g_a.stem, cell.cell_id)).write_bytes(b"x")

    ensure_s1_cache(
        archive_root=archive, cells=[cell], cache_dir=cache,
        window_days=_WINDOW, gpt=_present_gpt(tmp_path),
    )

    assert record_builds == [(g_b.stem, [cell.cell_id])]


def test_no_build_when_all_present(
    tmp_path: Path, covering_footprint: Polygon, record_builds: list
) -> None:
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    for acq, uid in ((_DAY_A, "AAAA"), (_DAY_B, "BBBB")):
        g = _granule(archive, acq, uid)
        (cache / cache_tif_name(g.stem, cell.cell_id)).write_bytes(b"x")

    ensure_s1_cache(
        archive_root=archive, cells=[cell], cache_dir=cache,
        window_days=_WINDOW, gpt=_present_gpt(tmp_path),
    )

    assert record_builds == []


def test_raises_when_missing_and_no_gpt(
    tmp_path: Path, covering_footprint: Polygon, record_builds: list
) -> None:
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    _granule(archive, _DAY_A, "AAAA")  # missing tif, gpt absent

    with pytest.raises(S1CacheUnavailableError, match="missing"):
        ensure_s1_cache(
            archive_root=archive, cells=[cell], cache_dir=cache,
            window_days=_WINDOW, gpt=tmp_path / "nonexistent_gpt",
        )
    assert record_builds == []


def test_genuinely_absent_s1_is_ok_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_builds: list
) -> None:
    """A cell no granule covers needs nothing built and must NOT raise (S1-free is fine)."""
    # Footprint that never intersects the cell (a far-away polygon).
    far = box(0.0, 0.0, 1.0, 1.0)
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: far)
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    _granule(archive, _DAY_A, "AAAA")

    # gpt absent — would raise IF anything were considered missing; it must not.
    ensure_s1_cache(
        archive_root=archive, cells=[cell], cache_dir=cache,
        window_days=_WINDOW, gpt=tmp_path / "nonexistent_gpt",
    )
    assert record_builds == []


def test_snap_per_cell_failure_is_tolerated(
    tmp_path: Path, covering_footprint: Polygon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SNAP CalledProcessError on a granule (the 'Empty region' anomaly) must not abort.

    The cell is left S1-free with a warning, not propagated — genuinely-unproducible S1 is
    acceptable. Only a systemic no-gpt failure raises.
    """
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    _granule(archive, _DAY_A, "AAAA")

    def _boom(*, granule_zip, cells, cache_dir, gpt, graph):  # type: ignore[no-untyped-def]
        raise subprocess.CalledProcessError(returncode=1, cmd=["gpt"])

    monkeypatch.setattr(snap, "build_granule_cache", _boom)

    # Must not raise — gpt is present, build fails per-cell, cell left S1-free.
    ensure_s1_cache(
        archive_root=archive, cells=[cell], cache_dir=cache,
        window_days=_WINDOW, gpt=_present_gpt(tmp_path),
    )


def test_out_of_window_granule_ignored(
    tmp_path: Path, covering_footprint: Polygon, record_builds: list
) -> None:
    """A covering granule acquired outside the window is not needed (no build, no raise)."""
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    cell = _cell()
    _granule(archive, datetime.date(2025, 1, 1), "CCCC")  # outside _WINDOW

    ensure_s1_cache(
        archive_root=archive, cells=[cell], cache_dir=cache,
        window_days=_WINDOW, gpt=tmp_path / "nonexistent_gpt",
    )
    assert record_builds == []
