r"""Audit S2 acquisition-date coverage of the full clipped archive (TASK-013b).

For each reference patch, checks that the **full** clipped S2 archive under
``data/clipped_bow_valley_selection_raw/sentinel2`` carries every acquisition date
the patch's timesteps need, and reports the missing-date download backlog.

This is the full-archive audit that used to live in ``test_s2_adapter.py``; it was
moved here because it is meaningless against the slim test-fixture subset (the test
keeps only a minimal, subset-aware version). Run it against the real archive::

    uv run python scripts/developer_scripts/bow_valley_inference_local/audit_s2_coverage.py

Exit code is non-zero if any patch has **no** covered date (a hard gap).
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_S2_ROOT = Path("data/clipped_bow_valley_selection_raw/sentinel2")

#: Each reference patch → the S2 acquisition dates its timesteps need. Mirrors
#: ``_NEEDED_DATES`` in ``tests/test_local_sources/test_s2_adapter.py``.
_NEEDED_DATES: dict[str, list[str]] = {
    "PR_20250406": ["2025-03-31", "2025-04-03", "2025-04-05"],
    "PR_20250414": ["2025-04-08", "2025-04-13"],
    "PR_20250423": ["2025-04-17", "2025-04-18", "2025-04-20", "2025-04-23"],
    "PR_20250502": ["2025-04-25", "2025-04-28", "2025-04-30"],
    "PR_20250510": ["2025-05-03", "2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
    "PR_20250519": ["2025-05-13", "2025-05-15", "2025-05-18"],
}


def _archive_acq_dates() -> set[datetime.date]:
    dates: set[datetime.date] = set()
    for zip_path in _S2_ROOT.glob("*.zip"):
        m = re.match(r"S2[ABC]_MSIL1C_(\d{8})T", zip_path.name)
        if m:
            dates.add(datetime.datetime.strptime(m.group(1), "%Y%m%d").date())
    return dates


def main() -> int:
    if not any(_S2_ROOT.glob("*.zip")):
        logger.error("no_s2_archive", root=str(_S2_ROOT))
        return 2

    archive = _archive_acq_dates()
    uncovered: dict[str, list[str]] = {}
    no_coverage: list[str] = []
    for patch, dates in _NEEDED_DATES.items():
        missing = [d for d in dates if datetime.date.fromisoformat(d) not in archive]
        covered = [d for d in dates if datetime.date.fromisoformat(d) in archive]
        if missing:
            uncovered[patch] = missing
        if not covered:
            no_coverage.append(patch)

    if uncovered:
        logger.warning("s2_coverage_backlog", missing_per_patch=uncovered)
    if no_coverage:
        logger.error("s2_patches_with_no_covered_date", patches=no_coverage)
        return 1
    logger.info("s2_coverage_ok", patches=len(_NEEDED_DATES), archive_dates=len(archive))
    return 0


if __name__ == "__main__":
    sys.exit(main())
