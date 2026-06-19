"""Per-(modality, cell, day) ``.npz`` cache for assembled cube bands.

The exporter assembles an 8-day window per ``(cell, window-end-day)``; consecutive
windows overlap by 7 days, so caching at the **per-(modality, cell, day)** grain
(rather than per-window multiband tif) avoids ~8× storage duplication (PLAN §4
"Cube cache layout").

**Directory sharding (filesystem-performance fix, REVIEW_AUDIT #3).** Entries are
sharded **one subdirectory per cell**::

    cube_cache/{cell_id}/{day:%Y%m%d}_{modality}.npz

A flat ``cube_cache/{cell}_{day}_{modality}.npz`` layout would put ~300k files
(mode A: ~344 cells × ~96 archive days × ~9 modalities) in a single directory,
degrading ext4/xfs directory indexing to O(N) on every lookup and eviction scan.
Per-cell sharding keeps each directory under ~1k entries (~864 files/cell). Per-cell
is sufficient at this scale — no hash-prefix tier needed.

**Eviction (day-frontier, parent-only).** The sweep is day-ordered and each cube
reads only an 8-day window ``[day - 7 … day]``, so once the driver advances past a
day every entry older than the live window is provably dead. Eviction therefore
prunes by that **day frontier**, not FIFO recency, and runs **once per day in the
parent process** (``prune_before_day``) — never in a worker, so concurrent workers
can never delete each other's entries (the Mode-B cross-process race; see
PLAN-CUBE-CACHE-DAY-EVICTION.md). It is **lazy**: a no-op while the cache is at or
under the configurable ``max_entries`` cap (Mode A never evicts), pruning only past
the cap and never touching the live window. ``put`` does not evict. Insertion order
is recovered from file mtime on construction so ``__len__`` and the cap hold across
process restarts. Cache + ``scratch/`` are intermediate and cleanable mid-run;
``cubes/`` and ``daily_fsc/`` are the kept deliverables.
"""

from __future__ import annotations

import datetime
from collections import OrderedDict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog

logger = structlog.get_logger(__name__)

#: Key under which the single band array is stored inside each ``.npz``.
_ARRAY_KEY = "array"

#: Cache format/content version. **Bump in the same diff that changes any adapter's
#: ``fetch`` or clip logic** — a stamped dir whose version differs is force-cleared on
#: construction (a known-incompatible cache can never be reused by mistake). The
#: interactive ``--cache-policy`` prompt backstops the *forgot-to-bump* case (see
#: PLAN-CUBE-CACHE-INVALIDATION.md).
CACHE_VERSION: int = 1

#: Name of the stamp file written at the cache root holding :data:`CACHE_VERSION`.
#: Never counted as a cache entry — ``_scan_existing`` globs ``*.npz`` only.
_VERSION_STAMP = ".cache_version"

#: Default FIFO entry cap. PLAN §4 sizes the cache by total bytes (~200 GB);
#: an entry cap is the simpler, deterministic proxy used here — each entry is one
#: per-(modality, cell, day) array of bounded size (~40 KB × bands). The exporter
#: passes an explicit cap derived from ``cube.yaml`` (TASK-003 subtask 7).
DEFAULT_MAX_ENTRIES = 200_000


class CubeCache:
    """A FIFO-evicted, per-cell-sharded ``.npz`` cache of cube band arrays.

    Args:
        root: Cache root directory (``…/bow_valley_processing/cube_cache``).
            Created if absent.
        max_entries: Maximum number of cached ``.npz`` files; the oldest-written
            entry is evicted when a new ``put`` would exceed it.
        overwrite: Clear the dir up front (then rewrite the stamp) before scanning.
            **Must only ever be set by a single, parent-process construction** — if
            concurrent worker constructions each cleared, one would wipe another's
            fresh entries mid-run (PLAN-CUBE-CACHE-INVALIDATION.md §Concurrency rule).

    Raises:
        ValueError: If ``max_entries`` is not positive.
    """

    def __init__(
        self,
        root: Path,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        *,
        overwrite: bool = False,
    ) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}.")
        self.root = Path(root)
        self.max_entries = max_entries
        self.root.mkdir(parents=True, exist_ok=True)

        # Invalidation gate (runs before any scan). Order matters: a version mismatch is
        # force-cleared regardless of `overwrite`, so a known-incompatible cache can never
        # be reused. Then an explicit `overwrite` clears. Otherwise the stamp is reconciled
        # (written if absent) and existing entries are kept.
        if self._stamped_version_mismatch():
            self._clear()
            logger.info(
                "cube_cache_version_invalidated",
                root=str(self.root),
                version=CACHE_VERSION,
            )
        elif overwrite:
            self._clear()
            logger.info("cube_cache_overwritten", root=str(self.root))
        else:
            self._write_stamp_if_absent()

        # Ordered oldest → newest; values are the on-disk paths.
        self._order: OrderedDict[str, Path] = self._scan_existing()

    # --- version stamp / invalidation -------------------------------------- #

    @property
    def _stamp_path(self) -> Path:
        return self.root / _VERSION_STAMP

    def _read_stamp(self) -> int | None:
        """Return the integer in the stamp file, or ``None`` if absent/unreadable."""
        if not self._stamp_path.exists():
            return None
        try:
            return int(self._stamp_path.read_text().strip())
        except (ValueError, OSError):
            # A corrupt/unreadable stamp is treated as a mismatch → force-clear.
            return None

    def _stamped_version_mismatch(self) -> bool:
        """True when a stamp is present (or corrupt) and differs from ``CACHE_VERSION``.

        A fresh dir (no stamp) is **not** a mismatch — it is reconciled by writing the
        current stamp, so a brand-new cache is never spuriously cleared.
        """
        if not self._stamp_path.exists():
            return False
        return self._read_stamp() != CACHE_VERSION

    def _write_stamp(self) -> None:
        """Write ``CACHE_VERSION`` to the stamp file (atomic replace)."""
        tmp = self._stamp_path.with_name(_VERSION_STAMP + ".tmp")
        tmp.write_text(f"{CACHE_VERSION}\n")
        tmp.replace(self._stamp_path)

    def _write_stamp_if_absent(self) -> None:
        """Stamp a fresh dir; leave a matching stamp untouched."""
        if not self._stamp_path.exists():
            self._write_stamp()

    def _clear(self) -> None:
        """Remove every cached ``.npz`` (and now-empty shard dirs), then (re)write the stamp.

        Resets in-memory order. The stamp file is preserved/rewritten and is **never** a
        cache entry (``_scan_existing`` globs ``*.npz`` only).
        """
        for path in self.root.rglob("*.npz"):
            path.unlink(missing_ok=True)
        self._remove_empty_shard_dirs()
        self._order = OrderedDict()
        self._write_stamp()

    def _remove_empty_shard_dirs(self) -> None:
        """Drop now-empty per-cell shard subdirs (leave the root and the stamp file)."""
        for child in sorted(self.root.rglob("*"), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()

    # --- key / path helpers ------------------------------------------------- #

    @staticmethod
    def _entry_key(modality: str, cell_id: int, day: datetime.date) -> str:
        """Stable string key for one cached array."""
        return f"{cell_id}/{day:%Y%m%d}_{modality}"

    def _entry_path(self, modality: str, cell_id: int, day: datetime.date) -> Path:
        """Resolve the sharded ``.npz`` path for a key."""
        return self.root / str(cell_id) / f"{day:%Y%m%d}_{modality}.npz"

    def _scan_existing(self) -> OrderedDict[str, Path]:
        """Rebuild FIFO order from any ``.npz`` already on disk (mtime-ordered).

        Lets the entry cap survive across process restarts: a fresh ``CubeCache``
        over a populated root resumes eviction from the existing files rather than
        forgetting them and overflowing the directory.
        """
        existing = sorted(self.root.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
        order: OrderedDict[str, Path] = OrderedDict()
        for path in existing:
            # key = "{cell_id}/{day}_{modality}" reconstructed from the shard path.
            key = f"{path.parent.name}/{path.stem}"
            order[key] = path
        if order:
            logger.info("cube_cache_scanned", root=str(self.root), entries=len(order))
        return order

    # --- public API --------------------------------------------------------- #

    def get(
        self,
        *,
        modality: str,
        cell_id: int,
        day: datetime.date,
    ) -> npt.NDArray[np.floating] | None:
        """Return the cached array for a key, or ``None`` on a miss.

        Args:
            modality: Source/modality tag (e.g. ``"s2"``, ``"modis"``).
            cell_id: Grid-cell id (the shard subdirectory).
            day: Acquisition day.

        Returns:
            The stored ``(C, H, W)`` array, or ``None`` if not cached.
        """
        path = self._entry_path(modality, cell_id, day)
        if not path.exists():
            return None
        with np.load(path) as data:
            return data[_ARRAY_KEY]

    def put(
        self,
        *,
        modality: str,
        cell_id: int,
        day: datetime.date,
        array: npt.NDArray[np.floating],
    ) -> Path:
        """Write an array to the cache, evicting the oldest entry past the cap.

        Re-putting an existing key overwrites it in place and refreshes its
        recency; it does **not** grow the entry count.

        Args:
            modality: Source/modality tag.
            cell_id: Grid-cell id (the shard subdirectory).
            day: Acquisition day.
            array: The ``(C, H, W)`` band array to store.

        Returns:
            The path the array was written to.
        """
        key = self._entry_key(modality, cell_id, day)
        path = self._entry_path(modality, cell_id, day)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: save to a temp sibling then replace, so a crash mid-write
        # never leaves a truncated .npz that a later get() would fail to load.
        # NOTE: np.savez appends ".npz" to a *path* argument; pass an open file
        # handle so the temp keeps its exact name and the rename target matches.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.savez(handle, **{_ARRAY_KEY: array})
        tmp.replace(path)

        # Track the entry (overwrite moves the key to newest). `put` no longer evicts:
        # eviction is exclusively the parent's day-frontier prune (prune_before_day), so
        # concurrent workers never delete each other's entries (PLAN-CUBE-CACHE-DAY-EVICTION).
        self._order.pop(key, None)
        self._order[key] = path
        return path

    def prune_before_day(self, current_day: datetime.date, *, window_days: int) -> int:
        """Drop entries strictly older than the live window — the **only** eviction path.

        Called once per day, in the **parent** process, before that day's worker pool
        spawns. The sweep is day-ordered and each cube reads only ``[day - window_days …
        day]``; so any entry with ``day < current_day - window_days`` is provably dead and
        safe to remove regardless of worker count. The live window is **never** touched.

        **Lazy:** a no-op while the cache is at or under ``max_entries`` — below the cap
        nothing is evicted (Mode A behaviour-identical). Only past the cap does it prune the
        dead frontier. If the cache is *still* over cap after pruning (the live window alone
        exceeds it), it logs a warning and returns — it never evicts a live entry.

        Args:
            current_day: The day about to be exported (the sweep's current position).
            window_days: Backlook span; entries with ``day < current_day - window_days``
                are dead. The exporter's ``(NUM_TIMESTEPS - 1) * DAYS_PER_TIMESTEP``.

        Returns:
            The number of entries removed.
        """
        if len(self._order) <= self.max_entries:
            return 0

        frontier = current_day - datetime.timedelta(days=window_days)
        removed = 0
        for key, path in list(self._order.items()):
            day = self._key_day(key)
            if day is None:
                # Unrecognised name → never delete (defensive).
                continue
            if day < frontier:
                path.unlink(missing_ok=True)
                del self._order[key]
                removed += 1

        self._remove_empty_shard_dirs()

        if len(self._order) > self.max_entries:
            logger.warning(
                "cube_cache_over_cap_after_prune",
                root=str(self.root),
                entries=len(self._order),
                max_entries=self.max_entries,
                live_window_days=window_days,
            )
        logger.info(
            "cube_cache_pruned",
            root=str(self.root),
            removed=removed,
            before_day=current_day.isoformat(),
            frontier=frontier.isoformat(),
        )
        return removed

    @staticmethod
    def _key_day(key: str) -> datetime.date | None:
        """Parse the ``day`` from an entry key ``{cell_id}/{YYYYMMDD}_{modality}``.

        Returns ``None`` if the date head does not parse (a stray/foreign file) — the
        caller leaves such entries untouched rather than risk deleting an unknown file.
        """
        stem = key.rsplit("/", 1)[-1]  # "{YYYYMMDD}_{modality}"
        head = stem.split("_", 1)[0]
        try:
            return datetime.datetime.strptime(head, "%Y%m%d").date()
        except ValueError:
            return None

    def __len__(self) -> int:
        """Number of entries currently tracked."""
        return len(self._order)
