"""Shared pytest configuration for the test suite.

**Slow-test scheduling under xdist.** The suite runs ``-n auto --dist loadgroup``
(see ``pyproject.toml``). The ``@pytest.mark.slow`` tests each GDAL-decode
multi-band real-archive rasters (S2/Landsat parity, lossless-clip checks). When
several are scheduled concurrently across all cores (16 on the dev/CI box) they
oversubscribe disk + GDAL I/O; on a loaded host a worker can stall long enough
that xdist reports the test as failed/crashed rather than merely slow. Observed: a
full suite at 7m51s reported two S2 parity tests "failed" while the identical
tests are deterministic (e.g. ``PR_20250423`` is 96.0 % bit-exact every run, well
above the 0.90 gate) and pass standalone, under ``-n 4``, and on a faster (3m49s)
full run.

**Fix.** Every ``slow`` test also carries
``@pytest.mark.xdist_group(SLOW_XDIST_GROUP)`` (paired statically at each test —
xdist's ``loadgroup`` scheduler reads the group from worker-side collection and
does **not** honour a group added dynamically in a collection hook, so the marker
must be on the test). ``loadgroup`` then runs all ``slow`` tests on a **single**
worker — serialized among themselves (no longer competing 16-wide for I/O) while
the fast suite still fans out across the remaining workers. ``SLOW_XDIST_GROUP``
is exported here so the pairing stays in sync from one constant.
"""

from __future__ import annotations

#: xdist group every ``@pytest.mark.slow`` test is pinned to (paired at each test
#: as ``@pytest.mark.xdist_group(SLOW_XDIST_GROUP)``). Runs them on one worker
#: under ``--dist loadgroup`` so heavy real-archive I/O is serialized.
SLOW_XDIST_GROUP = "slow_archive"
