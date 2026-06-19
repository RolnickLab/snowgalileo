"""CLI-side cube-cache policy resolution (PLAN-CUBE-CACHE-INVALIDATION §CLI).

The cube cache can serve **stale** band data after an adapter or clip change that the
:data:`~src.data.local_sources.cube_cache.CACHE_VERSION` stamp can't catch (a re-clip
leaves no code change). This module turns that staleness into an *explicit operator
decision* — reuse the existing cache or clear it — resolved **once in the parent process,
before any worker pool spawns**.

Clearing in the parent (never in a worker) is the single most important correctness
constraint: if every worker constructed ``CubeCache(overwrite=True)``, worker 2 would wipe
worker 1's freshly-written entries mid-run. The CLIs call :func:`resolve_cache_policy` up
front, which clears at most once, then build the exporter / pool with ``overwrite=False``.
"""

from __future__ import annotations

import enum
import sys
from pathlib import Path

import structlog

from src.data.local_sources.cube_cache import DEFAULT_MAX_ENTRIES, CubeCache

logger = structlog.get_logger(__name__)


class CachePolicy(str, enum.Enum):
    """How to treat an existing (non-empty) cube cache at startup.

    Attributes:
        PROMPT: Ask interactively when the cache is non-empty (the default). On a
            non-TTY this raises rather than guessing — silent staleness is the failure
            this whole mechanism exists to prevent.
        REUSE: Keep the existing cache; never clear.
        OVERWRITE: Clear the cache once, up front, in the parent process.
    """

    PROMPT = "prompt"
    REUSE = "reuse"
    OVERWRITE = "overwrite"


class CachePolicyError(RuntimeError):
    """Raised when ``prompt`` cannot be resolved (non-empty cache, no TTY)."""


def _count_entries(root: Path) -> int:
    """Number of cached ``.npz`` entries under ``root`` (0 if the dir is absent)."""
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.npz"))


def _prompt_reuse_or_overwrite(entries: int) -> bool:
    """Ask the operator to reuse or overwrite; return ``True`` to overwrite.

    Loops until a recognised answer is given. Only called when stdin is a TTY.
    """
    while True:
        answer = (
            input(f"Existing cube cache has {entries} entries. [r]euse / [o]verwrite? ")
            .strip()
            .lower()
        )
        if answer in ("r", "reuse"):
            return False
        if answer in ("o", "overwrite"):
            return True
        print("Please answer 'r' (reuse) or 'o' (overwrite).")


def resolve_cache_policy(
    *,
    root: Path,
    policy: CachePolicy,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    is_tty: bool | None = None,
) -> None:
    """Resolve the cache policy and clear the cache **once** if required.

    Runs in the parent process before any exporter is built or pool is spawned. After
    this returns, callers construct everything with ``overwrite=False`` — the dir is
    already in its intended state (cleared or kept). The version stamp still invalidates
    a known-incompatible dir on the next ``CubeCache`` construction regardless.

    Args:
        root: The cube-cache root directory.
        policy: The requested :class:`CachePolicy`.
        max_entries: Cap passed to the throwaway clearing ``CubeCache`` (does not affect
            entries when only reusing).
        is_tty: Override TTY detection (tests). ``None`` → ``sys.stdin.isatty()``.

    Raises:
        CachePolicyError: ``policy`` is ``PROMPT``, the cache is non-empty, and stdin is
            not a TTY (cannot ask, must not silently reuse).
    """
    entries = _count_entries(root)
    overwrite = False

    if policy is CachePolicy.REUSE:
        overwrite = False
    elif policy is CachePolicy.OVERWRITE:
        overwrite = True
    else:  # PROMPT
        if entries == 0:
            overwrite = False  # nothing to lose; proceed silently
        else:
            tty = sys.stdin.isatty() if is_tty is None else is_tty
            if not tty:
                raise CachePolicyError(
                    f"Cube cache at {root} has {entries} entries and --cache-policy is "
                    "'prompt', but stdin is not a TTY so the reuse/overwrite question "
                    "cannot be asked. Re-run with --cache-policy reuse or "
                    "--cache-policy overwrite to make the choice explicit."
                )
            overwrite = _prompt_reuse_or_overwrite(entries)

    if overwrite:
        # The ONE clear, in the parent. Workers/exporters then reuse the clean dir.
        CubeCache(root, max_entries, overwrite=True)
        logger.info("cube_cache_cleared_up_front", root=str(root), entries_removed=entries)
    else:
        logger.info("cube_cache_reused", root=str(root), entries=entries)
