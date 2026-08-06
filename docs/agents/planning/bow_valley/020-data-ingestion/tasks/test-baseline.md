# Test Baseline

The test suite is **green**. Judge every task by **absolute pass** — zero failures — not
by a delta against a list of tolerated ones.

## Baseline (verified 2026-08-06 at commit `12ec7d04`, branch `refactor-post-bow-valley-crunch`)

`uv run pytest -q` → **365 passed**, 0 failed, in ~5 min.

## Validation rule for every task

1. **Any failure is a regression.** Investigate it; do not compare it against a tolerated
   set. There is no tolerated set.
2. Run the **full suite** (`uv run pytest -q`), plus the task's own new test files
   targeted with `-x` while iterating.
3. If the count changes, say so explicitly and account for the difference — new tests
   added, tests removed, or a real regression.

## History: this file used to describe a red baseline

Until 2026-08-04 this document recorded **6 pre-existing failures** (captured at
`021b4540`, branch `raw_data_prep`) in `test_dataset.py`, `test_retrieve_cloud_state.py`,
`test_retrieve_season_from_filename.py`, and `test_sklearn_preprocessing.py`, and
prescribed a `comm`-based delta check against them. All six are gone —
`test_retrieve_cloud_state.py` no longer exists, and the rest pass; the module
restructure at `df89f502` is the likely fix, though the file was never updated to say so.

Recorded because the obsolete instruction was the actively dangerous part: a doc telling
future sessions to expect red and judge by delta trains them to wave past real failures.
If the suite goes red again, that is news — not the baseline.

## Slow real-archive tests are serialized under xdist (2026-06-09)

`@pytest.mark.slow` tests (S2/Landsat parity, `test_clip_dataset` lossless/CRS, S2
parity spike) each GDAL-decode multi-band real-archive rasters. `-n auto` spawns one
worker per core (16 on this box, zero headroom); when several slow tests are scheduled
concurrently they oversubscribe disk + GDAL I/O, and on a loaded host a worker can stall
long enough that xdist reports a test as **failed/crashed rather than slow**. This was
observed once as `test_s2_adapter::test_parity_b4_against_gee[PR_20250414/PR_20250423]`
"failing" on a 7m51s full run — while the tests are **deterministic** (`PR_20250423` is
96.0 % bit-exact every run, > the 0.90 gate) and pass standalone, under `-n 4`, and on a
faster (3m49s) full run. It was an oversubscription artifact, **not** a parity regression.

**Fix:** each `slow` test also carries `@pytest.mark.xdist_group("slow_archive")`, and
the suite runs `--dist loadgroup` (`pyproject.toml`), so all slow tests run **serialized
on one worker** while the fast suite still fans out. (xdist's `loadgroup` reads the group
from worker-side collection and does **not** honour a group added dynamically in a
collection hook, so the marker is paired statically at each test; `tests/conftest.py`
documents the constant.) If a real-archive parity test ever fails, **re-run it isolated**
(`pytest <nodeid> -p no:xdist`) before treating it as a regression — see KNOWLEDGE.md.
