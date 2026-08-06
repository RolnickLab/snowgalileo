# TASK-017: Generalize the grid CRS beyond UTM 11N

## 1. Goal

Let the grid builder produce cells for a region outside UTM 11N, so mode A can consume a
cells CSV from another region and mode B can tile another region's AOI, without editing
module-level constants.

## 2. Context & Why (deferred from the `legacy_csv` decision, 2026-08-06)

Resolving where the mode-A cells CSV lives (`CubeSettings.cube_cells_csv`) made the grid
builder *region-parameterized on paths* — a different CSV and a different AOI are now
per-run config. It did **not** make it region-parameterized on **CRS**, which is the
actual barrier. The CSV path move is worthless for a second region until this is done.

The two `# TODO Generalize CRS management for other regions` markers already in the tree
(`grid.py:87`, `base.py:56`) mark this exact task.

### Where UTM 11N is currently welded in

| Site                                    | What it does                                                                      |
| --------------------------------------- | --------------------------------------------------------------------------------- |
| `grid.py:90`                            | `GRID_MATH_CRS: str = "EPSG:32611"` — module constant, no override                |
| `grid.py:167-169` (`load_cells`)        | **Raises** if any CSV row's `crs` column is not `GRID_MATH_CRS`                   |
| `grid.py:289-296` (`_cell_to_gridcell`) | Passes `crs=GRID_MATH_CRS` into `GridCell.from_utm_bounds`                        |
| `grid.py:331` (`_tile_aoi_to_cells`)    | Reprojects the AOI 4326 → `GRID_MATH_CRS` before tiling                           |
| `base.py:59`                            | `CELL_TARGET_CRS: str = "EPSG:32611"`, the default for `GridCell.from_utm_bounds` |

A cells CSV in any other zone hard-fails at `load_cells`. A Fortress Mountain CSV passes
only by the luck of also being in zone 11N.

### The dead knob

`CubeSettings.cell_crs` (`settings.py:71`) is set in all three cube YAMLs
(`configs/bow_valley/cube.yaml:25`, `cube_full_run.yaml:35`, `cube_mode_b.yaml:36`) and
**`grid.py` never reads it**. The effective value comes from `base.CELL_TARGET_CRS`.
`mosaic.py:7` documents the target CRS as coming from "`cube.yaml` `cell_crs`", which is
not true. Editing `cell_crs` in a YAML today changes nothing and silently misleads.

Either wire `cell_crs` up as the single source of truth or delete it. Do not leave both.

## 3. Design decision required before implementing

Mode A and mode B learn the CRS from different places, and they must not disagree:

- **Mode A** — the cells CSV already carries a per-row `crs` column. Deriving from the
  data is self-describing and removes the `load_cells` raise. Needs a new guard: all rows
  must agree on one CRS (the current check enforces this incidentally, by pinning to a
  constant).
- **Mode B** — tiling an AOI has no CSV to read. Either declare the CRS in config
  (`cell_crs`) or derive the UTM zone from the AOI centroid. Deriving is convenient and
  wrong at zone boundaries and for AOIs spanning two zones — Bow Valley's own Landsat
  archive is genuinely mixed 32611/32612 (`landsat.py:33`, `_scene_ops.py:176`), so
  straddling is a real case here, not a hypothetical.

Recommendation to evaluate, not a decision: config declares `cell_crs`; mode A validates
the CSV against it and fails loudly on mismatch rather than silently preferring one. That
keeps one source of truth and makes a wrong pairing loud.

## 4. Scope warning — the grid is not the only weld

Generalizing `grid.py` + `base.py` makes a non-11N grid *expressible*. It does **not**
make a non-11N region *work end to end*. The adapters carry their own zone assumptions,
e.g. `s2.py:20` ("every archive tile is `T11U**` = EPSG:32611"). A full second-region run
needs an adapter-by-adapter CRS audit, which is larger than this task and should be sized
separately once the grid half is done.

## 5. Acceptance Criteria

- [ ] The grid CRS is resolved from config/data, not a module constant; both `# TODO Generalize CRS management for other regions` markers are removed.
- [ ] `load_cells` accepts a CSV in any single consistent projected CRS and fails loudly
  on rows that disagree.
- [ ] `cell_crs` is either the wired single source of truth or deleted from
  `CubeSettings` and the three cube YAMLs; `mosaic.py:7` matches reality either way.
- [ ] Bow Valley output is unchanged — a mode A and a mode B build on the existing
  config produce byte-identical cells to the pre-change build (regression, not just
  "tests pass").
- [ ] A test builds a grid in a second CRS (mode A from a fixture CSV in another zone,
  mode B from a non-11N AOI).
- [ ] `make mypy` clean; no new suite failures.

## 6. Out of scope

- The adapter CRS audit (§4) — separate task.
- Where the cells CSV and AOI live in config — done in the `legacy_csv` fix.
- Mixed-CRS cells *within one grid*. Cells stay in one CRS per run; the Landsat adapter
  already handles per-scene zone differences below the grid layer.
