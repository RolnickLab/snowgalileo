"""Tests for the S1 cache pre-flight ``ensure_s1_cache`` (silent-dropout guard) and the
per-granule ``build_granule_cache`` (idempotency + raw-safety + empty-region skip).

``ensure_s1_cache`` is **verify-only**: it decides which per-granule SNAP cache tifs a cube
export needs (raw granules whose acquisition day is in the window and whose footprint
overlaps the region) and **raises** if any are missing — so a cube is never silently
assembled with an all-``-9999`` S1 block when S1 was available. It does NOT build (the
cache is built offline). These tests monkeypatch the footprint so only the
needed/missing/coverage branching is under test, plus a SNAP-free check of
``build_granule_cache``'s atomic-write idempotency and read-only-raw guarantees.
"""

from __future__ import annotations

import datetime
import subprocess
import zipfile
from pathlib import Path

import pytest
from shapely.geometry import Polygon, box

import src.data.local_sources.s1_snap as snap
from src.data.local_sources.s1_snap import (
    S1CacheUnavailableError,
    build_granule_cache,
    cache_tif_name,
    ensure_s1_cache,
)

# Two acquisition days; the export window covers both.
_DAY_A = datetime.date(2025, 5, 12)
_DAY_B = datetime.date(2025, 5, 17)
_WINDOW = [_DAY_A, _DAY_B]

#: A region (4326) the granule footprints are tested against — the cell/AOI bbox.
_REGION = box(-116.34, 50.72, -116.30, 50.75)


def _granule(archive: Path, acq: datetime.date, uid: str, *, real_zip: bool = False) -> Path:
    """Create a raw-granule .zip whose stem parses to ``acq``.

    By default an empty placeholder (footprint is monkeypatched, the zip is never
    opened). With ``real_zip`` it holds a minimal ``*.SAFE/manifest.safe`` so
    ``build_granule_cache``'s extract step has something to open.
    """
    archive.mkdir(parents=True, exist_ok=True)
    stem = f"S1C_IW_GRDH_1SDV_{acq:%Y%m%d}T013724_{acq:%Y%m%d}T013749_001664_002BB2_{uid}"
    path = archive / f"{stem}.zip"
    if real_zip:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("S1C.SAFE/manifest.safe", "<xml/>")
    else:
        path.write_bytes(b"")
    return path


@pytest.fixture()
def covering_footprint(monkeypatch: pytest.MonkeyPatch) -> Polygon:
    """Make every granule footprint a polygon that covers any region (always intersects)."""
    big = box(-180.0, -90.0, 180.0, 90.0)
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: big)
    return big


def _present_gpt(tmp_path: Path) -> Path:
    gpt = tmp_path / "gpt"
    gpt.write_text("#!/bin/sh\n")
    return gpt


# --------------------------------------------------------------------------- #
# ensure_s1_cache — verify-only (no building); raises on a missing needed tif
# --------------------------------------------------------------------------- #
def test_passes_when_all_needed_tifs_present(
    tmp_path: Path, covering_footprint: Polygon
) -> None:
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    g_a = _granule(archive, _DAY_A, "AAAA")
    g_b = _granule(archive, _DAY_B, "BBBB")
    # Pre-seed both per-granule cache tifs → nothing missing.
    (cache / cache_tif_name(g_a.stem)).write_bytes(b"x")
    (cache / cache_tif_name(g_b.stem)).write_bytes(b"x")

    # Must not raise.
    ensure_s1_cache(
        raw_archive_root=archive, aoi_4326=_REGION, cache_dir=cache, window_days=_WINDOW
    )


def test_raises_when_a_needed_tif_is_missing(
    tmp_path: Path, covering_footprint: Polygon
) -> None:
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    g_a = _granule(archive, _DAY_A, "AAAA")
    _granule(archive, _DAY_B, "BBBB")  # DAY_B tif absent
    (cache / cache_tif_name(g_a.stem)).write_bytes(b"x")  # only DAY_A present

    with pytest.raises(S1CacheUnavailableError, match="missing"):
        ensure_s1_cache(
            raw_archive_root=archive, aoi_4326=_REGION, cache_dir=cache, window_days=_WINDOW
        )


def test_genuinely_absent_s1_is_ok_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A region no granule covers needs nothing and must NOT raise (S1-free is fine)."""
    far = box(0.0, 0.0, 1.0, 1.0)  # footprint that never intersects _REGION
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: far)
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    _granule(archive, _DAY_A, "AAAA")  # missing tif, but footprint does not cover region

    ensure_s1_cache(
        raw_archive_root=archive, aoi_4326=_REGION, cache_dir=cache, window_days=_WINDOW
    )


def test_out_of_window_granule_ignored(
    tmp_path: Path, covering_footprint: Polygon
) -> None:
    """A covering granule acquired outside the window is not needed (no raise)."""
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    _granule(archive, datetime.date(2025, 1, 1), "CCCC")  # outside _WINDOW, tif absent

    ensure_s1_cache(
        raw_archive_root=archive, aoi_4326=_REGION, cache_dir=cache, window_days=_WINDOW
    )


# --------------------------------------------------------------------------- #
# build_granule_cache — idempotency, raw-safety, empty-region skip (SNAP mocked)
# --------------------------------------------------------------------------- #
def test_build_skips_existing_tif_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A granule whose per-granule tif already exists is skipped (no SNAP) unless overwrite."""
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: box(-180, -90, 180, 90))
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    granule = _granule(archive, _DAY_A, "AAAA", real_zip=True)
    (cache / cache_tif_name(granule.stem)).write_bytes(b"prebuilt")

    calls: list[Path] = []
    monkeypatch.setattr(
        snap, "_run_snap_chain",
        lambda **k: calls.append(k["out_tif"]) or k["out_tif"],  # type: ignore[func-returns-value]
    )

    out = build_granule_cache(
        granule_zip=granule, aoi_4326=_REGION, cache_dir=cache, gpt=_present_gpt(tmp_path)
    )
    assert out == [cache / cache_tif_name(granule.stem)]
    assert calls == []  # SNAP not invoked — cache hit


def test_build_writes_atomically_via_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SNAP writes a .partial in cache_dir; success renames it to the final tif.

    Proves a crash mid-write leaves no truncated FINAL tif (the .partial is not matched
    by the s1_grd_*.tif cache glob), keeping the build safely idempotent.
    """
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: box(-180, -90, 180, 90))
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    granule = _granule(archive, _DAY_A, "AAAA", real_zip=True)
    final = cache / cache_tif_name(granule.stem)

    seen_targets: list[Path] = []

    def _fake_chain(*, safe_manifest, region_wkt, out_tif, gpt, graph):  # type: ignore[no-untyped-def]
        seen_targets.append(out_tif)
        assert out_tif.suffix == ".partial", "SNAP must write the .partial, not the final tif"
        assert not final.exists(), "final tif must not exist during the SNAP write"
        out_tif.write_bytes(b"snap-output")
        return out_tif

    monkeypatch.setattr(snap, "_run_snap_chain", _fake_chain)

    out = build_granule_cache(
        granule_zip=granule, aoi_4326=_REGION, cache_dir=cache, gpt=_present_gpt(tmp_path)
    )
    assert out == [final]
    assert final.exists() and final.read_bytes() == b"snap-output"
    assert not final.with_suffix(final.suffix + ".partial").exists()  # renamed away
    assert seen_targets and seen_targets[0].suffix == ".partial"


def test_build_skips_empty_region_granule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An 'Empty region!' SNAP failure removes the partial and returns [] (no abort).

    A granule whose AOI crop has no pixels makes SNAP exit non-zero; the build skips it
    (cleaning the partial) instead of aborting the whole offline run, and no false cache
    hit is left behind.
    """
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: box(-180, -90, 180, 90))
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    granule = _granule(archive, _DAY_A, "AAAA", real_zip=True)
    final = cache / cache_tif_name(granule.stem)

    def _boom(*, safe_manifest, region_wkt, out_tif, gpt, graph):  # type: ignore[no-untyped-def]
        out_tif.write_bytes(b"partial-corrupt")  # SNAP wrote some then failed
        raise subprocess.CalledProcessError(returncode=1, cmd=["gpt"])

    monkeypatch.setattr(snap, "_run_snap_chain", _boom)

    out = build_granule_cache(
        granule_zip=granule, aoi_4326=_REGION, cache_dir=cache, gpt=_present_gpt(tmp_path)
    )
    assert out == []
    assert not final.exists()
    assert not final.with_suffix(final.suffix + ".partial").exists()  # partial removed


def test_build_does_not_mutate_raw_granule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raw granule .zip is read-only: its bytes are unchanged after a build run."""
    monkeypatch.setattr(snap, "sentinel_safe_footprint", lambda *a, **k: box(-180, -90, 180, 90))
    archive = tmp_path / "sentinel1"
    cache = tmp_path / "sentinel1_snap"
    cache.mkdir(parents=True)
    granule = _granule(archive, _DAY_A, "AAAA", real_zip=True)
    before = granule.read_bytes()

    monkeypatch.setattr(
        snap, "_run_snap_chain",
        lambda **k: k["out_tif"].write_bytes(b"x") or k["out_tif"],  # type: ignore[func-returns-value]
    )
    build_granule_cache(
        granule_zip=granule, aoi_4326=_REGION, cache_dir=cache, gpt=_present_gpt(tmp_path)
    )
    assert granule.read_bytes() == before  # raw archive untouched
