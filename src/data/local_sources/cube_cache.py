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

**Eviction.** FIFO with a configurable entry cap. Insertion order is recovered
from file mtime on construction so the cap holds across process restarts (the
exporter may run in successive processes). Cache + ``scratch/`` are intermediate
and cleanable mid-run; ``cubes/`` and ``daily_fsc/`` are the kept deliverables.
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
        # Drop now-empty shard subdirs (leave the root and the stamp).
        for child in sorted(self.root.rglob("*"), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
        self._order = OrderedDict()
        self._write_stamp()

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

        # Refresh recency (overwrite moves the key to newest).
        self._order.pop(key, None)
        self._order[key] = path

        self._evict_to_cap()
        return path

    def _evict_to_cap(self) -> None:
        """Drop oldest entries until the cache holds at most ``max_entries``."""
        while len(self._order) > self.max_entries:
            old_key, old_path = self._order.popitem(last=False)  # FIFO: oldest first
            old_path.unlink(missing_ok=True)
            logger.info("cube_cache_evicted", key=old_key)

    def __len__(self) -> int:
        """Number of entries currently tracked."""
        return len(self._order)
