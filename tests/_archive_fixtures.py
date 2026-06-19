"""Resolver for adapter-test data — committed slim fixtures, gitignored archive fallback.

No adapter test reads ``data/`` directly. Source data is resolved under
``tests/fixtures/`` in two tiers:

- ``clipped/<source>/`` — **committed** windowed excerpts (in git). These are
  cropped to the reference-patch footprints (+ a halo wide enough for the
  reprojection stencil) and are **bit-exact** vs the full tile for the adapters'
  windowed reads, so they satisfy structural *and* parity tests. Built by
  ``scripts/developer_scripts/bow_valley_inference_local/test_data_building/``
  (``build_test_fixtures.py`` / ``build_slim_s2_safe.py``).
- ``archive/<source>/`` — **gitignored** larger data a developer downloads locally
  (hosted separately) for sources too big to commit (e.g. full S3 SEN3 products).
  Used only as a fallback when ``clipped/`` lacks the source.

:func:`resolve_source_root` returns ``clipped/<source>`` if present, else
``archive/<source>``, else ``None`` so the caller can ``pytest.skip``. It never
looks in ``data/``. ``resolve_structural_root`` / ``resolve_archive_root`` are kept
as aliases for call-site readability (both now share the clipped→archive lookup;
the historical distinction — slim crops corrupting parity — no longer holds, the
crops are bit-exact).
"""

from __future__ import annotations

from pathlib import Path

_FIXTURES = Path(__file__).parent / "fixtures"

#: Committed slim excerpts (in git).
CLIPPED_ROOT = _FIXTURES / "clipped"

#: Developer-downloaded larger data (gitignored), used when ``clipped/`` lacks a source.
ARCHIVE_ROOT = _FIXTURES / "archive"


def _first_with(root: Path, pattern: str) -> Path | None:
    return root if root.is_dir() and any(root.rglob(pattern)) else None


def resolve_source_root(source: str, *, pattern: str) -> Path | None:
    """Resolve a source's data root: committed ``clipped/`` first, else gitignored ``archive/``.

    Args:
        source: Sub-path under the fixture roots for this source (e.g. ``"dem"``).
        pattern: An ``rglob`` pattern the source's files match (e.g. ``"*_DEM.tif"``).

    Returns:
        ``clipped/<source>`` when it holds ≥1 matching file; else ``archive/<source>``
        when it does; else ``None`` (caller skips).
    """
    return _first_with(CLIPPED_ROOT / source, pattern) or _first_with(
        ARCHIVE_ROOT / source, pattern
    )


#: Readability aliases — both resolve clipped→archive (slim crops are bit-exact, so
#: parity no longer needs a separate "full data only" path).
resolve_structural_root = resolve_source_root
resolve_archive_root = resolve_source_root
