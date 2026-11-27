# Generalize the grid CRS beyond UTM 11N

Status: **Parked, not scheduled** (moved out of `bow_valley/020-data-ingestion/tasks/foundation/`
and un-numbered on 2026-08-06). Scoping it surfaced two welds outside the grid module (§4b) that
make this larger and more dangerous than a `grid.py` parameter thread. Needs its own planning
pass before anyone writes code.

Formerly `TASK-017-generalize-grid-crs.md`. The number is gone because this is no longer part of
the numbered Bow Valley task sequence; the two `grid.py` docstrings that cited `TASK-017` now
cite this path instead.

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

## 4b. Two welds outside `grid.py` (found 2026-08-06, the reason this is parked)

Neither is in the §2 table. Both must be fixed *in the same change* as the grid, not after.

### The output CRS is welded independently of the grid

`exporter.py:400` and `mosaic.py:163` pass `crs=CELL_TARGET_CRS` into `rasterio.open` while
`cell.crs` — already carrying the correct value — sits unused two lines away. Generalize the
grid without fixing these and a zone-12 run writes zone-12 pixels **labelled `EPSG:32611`**.

That is strictly worse than today's behaviour: the current `load_cells` raise fails loudly at
the front door, whereas this produces plausible-looking GeoTIFFs that are silently georeferenced
to the wrong place on Earth. Every downstream consumer — the viewer, the mosaic, any GIS — would
believe the label. Fix: write `cell.crs`, and guard in `DailyMosaicWriter.__init__` that the
grid is single-CRS (it already iterates the cells to compute bounds).

### Nothing rejects a geographic `cell_crs`

`_tile_aoi_to_cells` snaps its lattice with `CELL_SIZE_M = 1000`, in whatever units `cell_crs`
uses. Set `cell_crs: "EPSG:4326"` and mode B silently builds **1000-degree** cells — one cell
covering the planet, no error raised. Fix: reject a non-projected CRS via
`pyproj.CRS(cell_crs).is_projected` at the entry point.

### Design decision reached during scoping (not implemented)

Config declares `cell_crs`; mode A validates the CSV's `crs` column against it and fails loudly
on mismatch. Preferred over deriving mode A's CRS from the CSV data, because that leaves
`cell_crs` meaningful in mode B and dead in mode A — the same silent-mislead this task exists to
remove. Deriving mode B's zone from the AOI centroid stays rejected per §3.

Recorded as the leading candidate, **not** as a settled decision — it has not been tested
against §4b, which is what parked this.

## 5. Acceptance Criteria

- [ ] The grid CRS is resolved from config/data, not a module constant; both `# TODO Generalize CRS management for other regions` markers are removed.
- [ ] `load_cells` accepts a CSV in any single consistent projected CRS and fails loudly
  on rows that disagree.
- [ ] `cell_crs` is either the wired single source of truth or deleted from
  `CubeSettings` and the three cube YAMLs; `mosaic.py:7` matches reality either way.
- [ ] Both §4b welds are closed: `exporter.py` / `mosaic.py` write `cell.crs`, the mosaic
  writer rejects a mixed-CRS grid, and a geographic `cell_crs` is rejected at the entry point.
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
