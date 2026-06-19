# Test fixtures: committed crops vs gitignored archive

Adapter tests never read `data/`. Source data is resolved under `tests/fixtures/`
by `tests/_archive_fixtures.py` (`resolve_source_root`), which prefers the
committed tier and falls back to the gitignored one:

| Tier        | Path                               | Git            | Purpose                                                                                                                                  |
| ----------- | ---------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **clipped** | `tests/fixtures/clipped/<source>/` | **committed**  | Slim, windowed, **bit-exact** crops of the real archive. Satisfy structural *and* parity tests for these sources, with no download step. |
| **archive** | `tests/fixtures/archive/<source>/` | **gitignored** | Only sources too large to commit (S3). Downloaded locally / hosted separately. Skip if absent.                                           |

> CI runs `pytest -m 'not slow'`, so it executes the **structural** tests of the
> committed sources (the parity tests are marked `slow`); the parity tests still run
> from this committed data on any full-suite run. No source needs a download for CI.

## Why the crops are safe for parity

The adapters do **windowed reads** (`cell_window` → `reproject_to_cell`), and the
reproject only ever samples source pixels inside the cell footprint. So a crop that
covers the cell window plus a halo is **byte-identical** to the full tile for the
read — parity stays bit-exact. The only nuance is halo width: nearest-resampled
sources (MODIS/VIIRS/S2) need a few pixels; DEM's bilinear stencil reaches further,
so its crop uses a wide (256 px) halo. Verified: MODIS/VIIRS/S2/DEM parity all pass
on the committed crops (max diff 0.0 for nearest; within tolerance for DEM).

## What lives where

Committed (`clipped/`, ~21 MB total):

- **dem** (~5 MB) — per-patch wide-halo crops (`*_DEM.tif`).
- **worldcover** (~0.1 MB), **era5** (~1 MB) — structural; ERA5 is already AOI-native.
- **modis** (~10 MB), **viirs** (~4 MB) — the 8 test dates × the bands the adapter
  opens, windowed (nearest → bit-exact).
- **sentinel2 / sentinel2_raw** (~1.5 / ~0.3 MB) — slim **lossless** SAFE zips with
  only the 6 cube bands + `MSK_CLASSI`, windowed (lossy JP2 corrupts reflectance +
  the mask, so REVERSIBLE/QUALITY=100 is mandatory).

Gitignored (`archive/`, download-only, skips in CI):

- **sentinel3** (~184 MB) — kept **full**: the OLCI swath warp reads
  `geo_coordinates`, so cropping is high-risk. The one source too big to commit.
- **sentinel1** — not stored here at all: the real-archive parity needs ESA SNAP
  (`gpt`) and is hard-skipped without it. Synthetic S1 tests run in CI regardless.

## Rebuilding the fixtures

From the repo root, with the real clipped archive present locally
(`data/clipped_bow_valley_selection_raw/`):

```bash
# DEM + WorldCover + ERA5 committed crops
uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/build_test_fixtures.py

# MODIS + VIIRS committed crops; S3 → gitignored archive
uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/populate_test_archive.py \
    --source dem modis viirs sentinel3

# S2 slim lossless SAFE zips (clipped + raw), committed
uv run python scripts/developer_scripts/bow_valley_inference_local/test_data_building/build_slim_s2_safe.py --include-raw
```

To run the S3 tests, download/restore `tests/fixtures/archive/sentinel3/` (zip the
`archive/` tier to host it elsewhere).

## Full-archive audits (developer scripts, not tests)

Checks that need the *complete* `data/` archive (not the test subset) live as
developer scripts, e.g.
`scripts/developer_scripts/bow_valley_inference_local/audit_s2_coverage.py` (every
patch has ≥1 covered S2 date + the TASK-013b download backlog). The test suite
keeps only minimal, subset-aware counterparts.
