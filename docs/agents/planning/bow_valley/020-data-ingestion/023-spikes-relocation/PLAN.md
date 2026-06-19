# PLAN — Relocate `scripts/spikes/` out of `scripts/`

## Problem

`scripts/spikes/` is an importable Python package (`__init__.py` present), but the
wheel ships `src` only (`[tool.hatch.build.targets.wheel] packages = ["src"]`).
`tests/test_local_sources/test_s2_parity.py` does
`from scripts.spikes.s2_parity_spike import run_s2_spike`, which works only because
the repo root is on `sys.path` during local test runs — it would break in an
installed wheel. `scripts/` is for runnable scripts, not packages.

## Decisions (user-approved)

1. **Importable parity logic** (`run_*_spike` + their private helpers/constants)
   moves into the shipped package: `src/data/local_sources/parity/`.
2. **Everything else** in `scripts/spikes/` (CLI entrypoints, SNAP `.xml` graph
   assets, pure-CLI diagnostics) moves to
   `scripts/developer_scripts/bow_valley_inference_local/spikes/`, with **no**
   `__init__.py` — plain runnable scripts, not a package.
3. CLI entrypoints become **thin wrappers** that call the `src` logic. Assets stay
   alongside the scripts. Console-script entrypoints in `pyproject` were rejected
   (overkill for spike tooling).

## Contract: the importable surface (what tests/wrappers depend on)

| New module | Public symbol | Imported by |
|---|---|---|
| `src/data/local_sources/parity/s2.py` | `run_s2_spike`, `S2_BANDS`, `S2_HARMONIZE_OFFSET_DN` | `test_s2_parity.py`, wrapper |
| `src/data/local_sources/parity/s1.py` | `run_s1_spike` | wrapper |
| `src/data/local_sources/parity/s3.py` | `run_s3_ortho_spike` | wrapper |

`s1`/`s3` parity logic is not currently imported by any test, but is genuine
importable logic (helpers + `run_*` returning arrays) — it belongs in `src` for
symmetry and to keep the wrappers thin.

## File-by-file

### Promote to `src/data/local_sources/parity/` (new package, has `__init__.py`)

- `s2_parity_spike.py` → `parity/s2.py`. Keep `run_s2_spike`, `_grid_from_patch`,
  `_jp2_path`, `S2_BANDS`, `_JP2_SUFFIX`, `S2_HARMONIZE_OFFSET_DN`. **Drop** `_main`.
- `s1_parity_spike.py` → `parity/s1.py`. Keep `run_s1_spike`, `_grid_from_patch`,
  `_aoi_wkt`, `_run_snap_chain`, `_AOI_MARGIN_DEG`. **Drop** `_main`. The
  `run_s1_spike` already takes `graph: Path = _GRAPH` (confirmed). **Drop** the
  module-level `_GRAPH` (it was a repo-relative `scripts/...` path, invalid in
  `src`); make `graph` a required kwarg. The wrapper supplies
  `Path(__file__).with_name("s1_grd_snap_graph.xml")`. Same for `gpt` if it has a
  repo-relative default.
- `s3_olci_parity_spike.py` → `parity/s3.py`. Keep `run_s3_ortho_spike`,
  `_grid_from_patch`, `_aoi_wkt`, `_run_snap_ortho`, `_stats`, `_AOI_MARGIN_DEG`.
  **Drop** `_main`. Same `_GRAPH` treatment as s1 (`graph` already a kwarg;
  wrapper supplies `s3_olci_ortho_graph.xml`).

### Move whole to `scripts/developer_scripts/bow_valley_inference_local/spikes/`

These are **pure CLI diagnostics** — only `main()`, no importable `run_*`, no
importer anywhere. Move verbatim (update internal path strings only):

- `verify_s1_cache.py`
- `diagnose_s1_border_noise.py`

### SNAP graph assets → `scripts/developer_scripts/bow_valley_inference_local/spikes/`

- `s1_grd_snap_graph.xml`
- `s1_grd_graph_no_border_noise.xml`
- `s3_olci_ortho_graph.xml`

(Note: `src/data/local_sources/s1_grd_graph.xml` is the **production** graph — not
touched.)

### New thin CLI wrappers in `scripts/developer_scripts/bow_valley_inference_local/spikes/`

- `run_s1_parity.py` — argparse `_main` from old `s1`, calls
  `src.data.local_sources.parity.s1.run_s1_spike`, supplies graph path
  (`<this dir>/s1_grd_snap_graph.xml` via `Path(__file__).with_name(...)`).
- `run_s2_parity.py` — argparse `_main` from old `s2`, calls `...parity.s2.run_s2_spike`.
- `run_s3_parity.py` — argparse `_main` from old `s3`, calls `...parity.s3.run_s3_ortho_spike`.

Wrapper defaults that are repo-relative (`data/...`, `tests/fixtures/...`) stay
repo-relative — these are run from repo root (`uv run python scripts/...`).

### Delete

- `scripts/spikes/__init__.py` and the directory `scripts/spikes/` (after moves).
- `scripts/spikes/__pycache__/`.

## Reference updates (honesty — no stale paths)

Tests (functional — **must** update or they break):
- `tests/test_local_sources/test_s2_parity.py:58` import →
  `from src.data.local_sources.parity.s2 import run_s2_spike`. Docstring `:4` path.
- `tests/test_local_sources/test_s1_parity.py` lines 10, 29, 72 — `.xml` path →
  `scripts/developer_scripts/bow_valley_inference_local/spikes/s1_grd_snap_graph.xml`.

Docs (text only):
- `docs/local_data_processing.md:139`
- `docs/agents/KNOWLEDGE.md:234,301`
- `docs/agents/planning/bow_valley/020-data-ingestion/021-parity-spike-notes.md:11,73,106,370`
- `.../tasks/foundation/TASK-005-s1-s2-parity-spikes.md:65,66,68,69,81`
- `.../tasks/adapters-scene/TASK-011-s3-olci-adapter.md:58`

## Validation

1. `uv run ruff check src/data/local_sources/parity scripts/developer_scripts/bow_valley_inference_local/spikes`
2. `uv run mypy src/data/local_sources/parity`
3. `uv run pytest tests/test_local_sources/test_s2_parity.py -q` (the one with a real
   import dependency; respects `@pytest.mark.slow` — may need `-m slow`).
4. `grep -rn "scripts/spikes" --include=*.py --include=*.md .` returns **nothing**.
5. Smoke each wrapper's `--help` to confirm argparse + import path resolve.

## Out of scope

No behavior change to parity logic. No console-script entrypoints. Production
`s1_grd_graph.xml` and `s1_snap.py` untouched. Judge test delta against the
known-red baseline (6 pre-existing failures); the slow/archive parity tests gate on
local data + SNAP and may skip.
