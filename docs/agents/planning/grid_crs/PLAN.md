# PLAN: Generalize the grid CRS beyond UTM 11N

Research and code inventory: [RESEARCH.md](RESEARCH.md).

## 1. Problem & Why

The grid builder and every output writer are welded to `EPSG:32611` through module constants
(`grid.GRID_MATH_CRS`, `base.CELL_TARGET_CRS`). A region outside UTM 11N cannot be expressed
without editing code. The configured `CubeSettings.cell_crs` is read by nothing, so changing it
silently does nothing.

Two silent failure paths make this worse:

- The exporter and the mosaic label their output with the constant rather than `cell.crs`.
  Generalizing the grid alone would write correctly placed pixels labelled with the wrong CRS.
- The local pipeline is a chain of scripts. S1 preprocessing (`process-s1`) picks a target CRS
  (hardcoded 32611) independently of the cube config, and nothing checks that the CRS one step
  wrote matches the CRS the next step expects.

## 2. Outcome & Success Criteria

"Expressible": the grid and its outputs work in any projected metre CRS. Only the cube config
picks the target CRS. Everything upstream of cube assembly is CRS-neutral.

- [ ] The run CRS comes from:
  - `CubeSettings.cell_crs` in the local pipeline (`03`, `04`);
  - `--cell-crs` in `00_generate_grid_csv.py` and `build_aoi_cubes_gee_url.py`;
  - the cube CSV `crs` column in `infer_aoi_cubes.py`.
- [ ] `GRID_MATH_CRS` and `CELL_TARGET_CRS` are deleted.
- [ ] Mode A raises when a cells CSV row's `crs` disagrees with the run CRS (normalized comparison).
- [ ] Mode B tiles in the run CRS.
- [ ] A geographic or non-metre run CRS is rejected at entry.
- [ ] Cube CSVs, exported cube tifs and daily mosaics carry the run CRS.
- [ ] The mosaic writer rejects a mixed-CRS grid.
- [ ] 04 `--cubes-only` checks the CRS of the cubes it reads against the run CRS and fails loudly on
  a mismatch.
- [ ] Two runs with different grids cannot share one cube cache.
- [ ] S1 preprocessing projects each granule to its own UTM zone, independent of the cube config.
- [ ] Stages 01-02 leave Bow Valley output unchanged:
  - mode A and mode B cube CSVs are byte-identical;
  - the `GridCell` lists are equal;
  - cube tifs and the mosaic keep the same CRS and transform.
- [ ] Stage 03 S1 parity against GEE is no worse than today.
- [ ] A test builds a mode A and a mode B grid in a second CRS and asserts the CSV `crs` column
  and `GridCell.crs`.

## 3. Constraints

- The three Bow Valley cube YAMLs and the full-run cubes (`full_run/cubes`, 461,685 files) must
  keep working. The mode-B grid must stay a bijection with them.
- One CRS per run. No mixed-CRS cells within one grid.
- The local pipeline stays a set of separate scripts (`00`-`05`). CRS agreement is checked at
  handoffs, not by merging the steps.
- `make mypy` stays clean, and no new test failures.

## 4. Non-goals

- Running a real second region end to end.
- The rest of the adapter CRS audit (RESEARCH §4), such as the S2 `T11U**` tile-zone assumption.
  S1 is the exception, handled in stage 03.
- Mixed-CRS grids.
- Deriving the CRS from the AOI centroid (wrong at zone boundaries, RESEARCH §3).
- A unified local pipeline.

## 5. Stages

| #   | Slug             | Goal                                                                                               | Done when                                                                                                                                                                           | Depends on | Status |
| --- | ---------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------ |
| 01  | output-crs-label | Outputs write `cell.crs`; consumers check the CRS of files they read. Zero behaviour change.       | Exporter and mosaic write `cell.crs`. The mosaic writer rejects a mixed-CRS grid. The S1 pre-flight and 04 `--cubes-only` fail loudly on a CRS mismatch. Bow Valley tifs unchanged. | -          | active |
| 02  | thread-grid-crs  | Grid, `00`, settings, Fortress scripts and cache take the run CRS from config, CLI or CSV.         | All section 2 criteria except the S1 ones pass                                                                                                                                      | 01         | later  |
| 03  | s1-native-crs    | SNAP terrain-corrects each granule into its own UTM zone; the S1 adapter reprojects to `cell.crs`. | SNAP cache rebuilt; S1 pre-flight CRS check removed; S1 parity vs GEE no worse than today                                                                                           | 02         | later  |

## 6. Stage Contracts

### 01 → 02

- **01 provides:**
  - `exporter.py` and `mosaic.write_day` write `cell.crs`.
  - `DailyMosaicWriter` holds the grid's single CRS and raises on a mixed-CRS grid.
  - CRS handoff checks against `cell.crs`, raising on mismatch:
    - S1 pre-flight (`LocalSourceExporter`, `verify_s1_cache`) checks the SNAP tifs a cell reads;
    - 04 `--cubes-only` checks the cubes it reads.
  - No signature changes in `grid.py` or `base.py`.
- **02 requires of 01:**
  - No output path or handoff check reads `CELL_TARGET_CRS`; all of them use `cell.crs`.
  - 02 can therefore delete the constant without touching the writers or the checks.
- **Frozen:** writers and checks take the CRS from `GridCell.crs`, and `GridCell` is unchanged.
- **Open:**
  - How the CRS reaches the grid (02 decides).
  - The cube CSV `crs` column (`build_cube_dataframe*`) belongs to 02.

### 02 → 03

- **02 provides:**
  - `cell.crs` is the configured run CRS everywhere in stage 2+.
  - The cube-cache stamp carries a grid fingerprint.
- **03 requires of 02:** nothing beyond `cell.crs` being correct. The S1 adapter already
  reprojects from each SNAP file's own CRS (`s1.py`, `reproject_to_cell(src_crs=crs)`).
- **Frozen:** the `cell_crs` configuration surface and the grid API.
- **Open:**
  - How SNAP picks the zone: `AUTO:42001` if Terrain-Correction accepts it, else an EPSG code
    computed from the granule centre (03 decides).
  - The `CACHE_VERSION` bump for the S1 change belongs to 03.

## 7. Parking Lot

- Rest of the adapter CRS audit for a full second-region local run (RESEARCH §4).
- A unified local pipeline, likely once source download is automated. If built, orchestrate the
  existing scripts rather than merge them, and keep the handoff checks: files on disk may come
  from older runs.
- `mosaic.py` and viewer docstrings that state EPSG:32611 as a fact. Fix those touched by 01-03;
  the rest when the adapter audit runs.

## 8. Decisions & Learnings

- 2026-11-27: Scope is "expressible", not a real second region.
- 2026-11-27: Output hardening (RESEARCH §4c) ships first because it removes the silent-mislabel
  path and has no behaviour change.
- 2026-11-27: `cell_crs` is the single source of truth in both modes. Mode A validates the CSV
  against it rather than deriving the CRS from the CSV.
- 2026-11-27: `cell_crs` becomes **required** (no default). This breaks bare `CubeSettings()`
  (one call in `src/scripts/tests`). The three cube YAMLs already set it.
- 2026-11-27: Hybrid pipeline: the cube CSV is the CRS carrier, as it already is for the grid.
  - `build_aoi_cubes_gee_url.py` takes `--cell-crs`.
  - `infer_aoi_cubes.py` reads the single CRS from the CSV and rejects a CSV mixing CRSs.
- 2026-11-27: `--cell-crs` is **required** in `00_generate_grid_csv.py` and
  `build_aoi_cubes_gee_url.py`, matching `cell_crs`.
- 2026-11-27: Where the target CRS is used:
  - The AOI is 4326.
  - The clip stage needs no target CRS: it reprojects the AOI into each source's native CRS
    (`clip/clippers.py`) and keeps native pixels.
  - The target CRS is used only by grid building (`00`, `03`, `04`), the adapters' reprojection
    at export, and the outputs.
  - SNAP is the one upstream exception, a weld rather than a need.
- 2026-11-27: S1 has no native map CRS (radar geometry). The CRS-neutral equivalent is the
  granule's own UTM zone, mirroring how Landsat and S2 keep their native zones.
  - This is stage 03, after 02.
  - It accepts that Bow Valley S1 values change for granules whose zone is not 11, and that the
    SNAP cache must be rebuilt.
  - Expected to improve GEE parity if `COPERNICUS/S1_GRD` scenes are stored in their own UTM
    zone, since that reproduces GEE's terrain-correct-then-reproject chain. The parity spike
    attributed most of the S1 residual to resampling.
  - **Todo: verify** GEE's native S1 projection before stage 03's design (needs EE credentials):
    ```bash
    uv run python -c "import ee; ee.Initialize(); print(ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(ee.Geometry.Point(-115.6,51.2)).filterDate('2025-04-06','2025-04-08').first().select('VV').projection().getInfo())"
    ```
- 2026-11-27: Local pipeline uniformity is enforced by consumer-side checks on GeoTIFF headers.
  - The SNAP check (stage 01) is temporary and is removed by stage 03 once SNAP is CRS-neutral.
  - The `--cubes-only` check is permanent.
- 2026-11-27: The local pipeline stays separate scripts for now: its steps differ by hours of
  runtime and by hardware, and they are re-run independently.
- 2026-11-27: The cube-cache fix is in stage 02: a grid fingerprint (CRS and grid bounds) joins
  the version stamp, so a mismatch clears the cache.
- 2026-11-27: Changing the stamp format clears every existing cube cache once, including the
  Bow Valley mode-B cache, which is rebuilt on the next build sweep. Cubes-only
  runs do not use the cache.
- 2026-11-27: The mode-B area-of-use warning (RESEARCH §3) is dropped: a required `cell_crs`
  removes the forgotten-default case it guarded.
