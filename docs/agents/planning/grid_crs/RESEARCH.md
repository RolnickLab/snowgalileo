# Generalize the grid CRS beyond UTM 11N

Status: **Research input to [PLAN.md](PLAN.md)** (graduated from `upcoming/` on 2026-11-27).
Decisions taken there override this document where they differ.

Formerly `TASK-017-generalize-grid-crs.md`. The number is gone because this is no longer part of
the numbered Bow Valley task sequence; the two `grid.py` docstrings that cited `TASK-017` now
cite this path instead.

References below name symbols, not line numbers: line numbers drifted between 2026-08-06 and the
2026-11-24 review.

## 1. Goal

Let the grid builder produce cells for a region outside UTM 11N, so mode A can consume a
cells CSV from another region and mode B can tile another region's AOI, without editing
module-level constants.

## 2. Context & Why (deferred from the `legacy_csv` decision, 2026-08-06)

Resolving where the mode-A cells CSV lives (`CubeSettings.cube_cells_csv`) made the grid
builder *region-parameterized on paths*: a different CSV and a different AOI are now
per-run config. It did **not** make it region-parameterized on **CRS**, which is the
actual barrier. The CSV path move is worthless for a second region until this is done.

The `# TODO Generalize CRS management for other regions` marker above `base.CELL_TARGET_CRS`
marks this exact task (the one above `grid.GRID_MATH_CRS` was removed on 2026-11-27).

### Where UTM 11N is currently welded in

| Site                                                | What it does                                                                  |
| --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `grid.GRID_MATH_CRS`                                | `"EPSG:32611"`, module constant, no override                                  |
| `grid.load_cells`                                   | **Raises** if any CSV row's `crs` column is not `GRID_MATH_CRS`               |
| `grid._make_transformer`                            | 32611 to 4326 transformer used by the mode-A AOI filter                       |
| `grid._cell_to_gridcell`                            | Passes `crs=GRID_MATH_CRS` into `GridCell.from_utm_bounds`                    |
| `grid.tile_aoi_to_cells`                            | Reprojects the AOI 4326 to `GRID_MATH_CRS` before tiling                      |
| `grid.build_cube_dataframe`                         | Writes `"crs": GRID_MATH_CRS` into every emitted cube-CSV row                 |
| `grid.build_cube_dataframe_for_gee_utm`             | Reprojects cell centres from `GRID_MATH_CRS` to lat/lon for the GEE filenames |
| `base.CELL_TARGET_CRS`                              | `"EPSG:32611"`, the **default** `crs` of `GridCell.from_utm_bounds`           |
| `exporter.py` (`rasterio.open`), `mosaic.write_day` | Write `crs=CELL_TARGET_CRS` instead of `cell.crs` (see §4b)                   |
| `scripts/.../infer_aoi_cubes.py` (cube CSV loader)  | Rejects any row CRS other than `CELL_TARGET_CRS`                              |

A cells CSV in any other zone hard-fails at `load_cells`. A Fortress Mountain CSV passes
only by the luck of also being in zone 11N.

`build_cube_dataframe` is the same silent-mislabel class as §4b: generalize the tiling but not
this, and a zone-12 run emits a cube CSV whose `crs` column says 32611. The GEE reader
(`eo_eval.export_from_csv_utm`) trusts that column per row.

Already CRS-neutral, no change needed: the GEE UTM reader (reads `crs` per row), the prebuilt
cube filenames (`layout.cell_centre_lat_lon` uses `cell.crs`), `_scene_ops` and the adapters'
`reproject(dst_crs=cell.crs)` calls.

### The dead knob

`CubeSettings.cell_crs` is set in all three cube YAMLs (`configs/bow_valley/cube.yaml`,
`cube_full_run.yaml`, `cube_mode_b.yaml`) and **nothing in `src/` reads it**; only
`test_cube_settings.py` asserts its value. The effective value comes from
`base.CELL_TARGET_CRS`. The `mosaic.py` module docstring documents the target CRS as coming
from "`cube.yaml` `cell_crs`", which is not true. Editing `cell_crs` in a YAML today changes
nothing and silently misleads.

Either wire `cell_crs` up as the single source of truth or delete it. Do not leave both.

## 3. Design decision

Mode A and mode B learn the CRS from different places, and they must not disagree:

- **Mode A**: the cells CSV already carries a per-row `crs` column.
- **Mode B**: tiling an AOI has no CSV to read. Deriving the UTM zone from the AOI centroid
  is convenient and wrong at zone boundaries and for AOIs spanning two zones. Bow Valley's
  own Landsat archive is genuinely mixed 32611/32612 (`landsat.py`, `_scene_ops.py`), so
  straddling is a real case here, not a hypothetical. Rejected.

**Decision (recommended, confirm before implementing):** config declares `cell_crs`; mode A
validates the CSV's `crs` column against it and fails loudly on mismatch. Preferred over
deriving mode A's CRS from the CSV data, because that leaves `cell_crs` meaningful in mode B
and dead in mode A, the same silent-mislead this task exists to remove.

The 2026-08-06 note said this was "not tested against §4b". It does not need to be: the §4b
fixes (write `cell.crs`, reject non-metre CRSs) are required whichever way the CRS is sourced.

### Sub-decisions

- **Threading.** `CellGeometry` has no `crs` field. Thread a single `crs: str` argument
  through `load_cells`, `_make_transformer`, `tile_aoi_to_cells`, `_cell_to_gridcell` and
  `build_cube_dataframe*`, entered once from `CubeSettings.cell_crs`. Do not add `crs` to
  `CellGeometry`: one CRS per run (§6) makes a per-cell field redundant.
- **No silent default.** Make `crs` a required argument of `GridCell.from_utm_bounds` and
  delete `base.CELL_TARGET_CRS` and `grid.GRID_MATH_CRS`. A default is a silent fallback to
  32611; removing it lets mypy find every call site.
- **`cell_crs` default.** Keeping `cell_crs: str = "EPSG:32611"` means a mode-B run for a new
  region that forgets to set it tiles in the wrong zone with no error. Mode A is protected
  by the CSV check. Guard: warn (do not raise, straddling AOIs are legitimate) if the AOI
  centroid lies outside `pyproj.CRS(cell_crs).area_of_use`. Making the field required is the
  stricter alternative; it breaks every caller that builds `CubeSettings()` bare.
- **CRS comparison.** Compare normalized CRSs, not raw strings: `"epsg:32612"`, `"EPSG:32612"`
  and WKT must all match. Normalize both sides with `pyproj.CRS(x).to_string()` once at
  entry; keep the normalized string in `GridCell.crs`.

## 4. Scope warning: the grid is not the only weld

Generalizing `grid.py` + `base.py` makes a non-11N grid *expressible*. It does **not**
make a non-11N region *work end to end*. The adapters carry their own zone assumptions,
e.g. `s2.py` ("every archive tile is `T11U**` = EPSG:32611"), and the S1 SNAP graph
hardcodes `<mapProjection>EPSG:32611</mapProjection>` (`s1_grd_graph.xml`), with
`s1_snap._utm_area_m2` defaulting to `utm_epsg=32611`. A full second-region run needs an
adapter-by-adapter CRS audit, which is larger than this task and should be sized
separately once the grid half is done.

## 4b. Welds outside `grid.py` (found 2026-08-06)

### The output CRS is welded independently of the grid

`exporter.py` and `mosaic.write_day` pass `crs=CELL_TARGET_CRS` into `rasterio.open` while
`cell.crs`, already carrying the correct value, sits unused. Generalize the grid without
fixing these and a zone-12 run writes zone-12 pixels **labelled `EPSG:32611`**.

That is strictly worse than today's behaviour: the current `load_cells` raise fails loudly at
the front door, whereas this produces plausible-looking GeoTIFFs that are silently georeferenced
to the wrong place on Earth. Every downstream consumer (the viewer, the mosaic, any GIS) would
believe the label. Fix: write `cell.crs`, and guard in `DailyMosaicWriter.__init__` that the
grid is single-CRS (it already iterates the cells to compute bounds); store that CRS on the
writer for `write_day`.

### Nothing rejects a non-metre `cell_crs`

`tile_aoi_to_cells` snaps its lattice with `CELL_SIZE_M = 1000`, in whatever units `cell_crs`
uses. Set `cell_crs: "EPSG:4326"` and mode B silently builds **1000-degree** cells, one cell
covering the planet, no error raised. `is_projected` alone is not enough: a projected CRS in
feet (e.g. EPSG:2263) builds 1000-ft cells. Fix: at the entry point, reject unless
`crs.is_projected` and `crs.axis_info[0].unit_name == "metre"`.

## 4c. Precursor change (do first, separately)

Zero-behaviour-change hardening, shippable now because every `cell.crs` is 32611 today:

- `exporter.py` and `mosaic.write_day` write `cell.crs`.
- `DailyMosaicWriter.__init__` rejects a mixed-CRS grid.
- `build_cube_dataframe` writes the cells' CRS instead of `GRID_MATH_CRS` (needs the `crs`
  argument; if that drags in the threading, leave this item to the main change).

This removes the silent-mislabel path before any CRS threading starts.

## 5. Acceptance Criteria

- [ ] The grid CRS is resolved from `CubeSettings.cell_crs`, not a module constant;
  `GRID_MATH_CRS` and `CELL_TARGET_CRS` are deleted and both
  `# TODO Generalize CRS management for other regions` markers are removed.
- [ ] `GridCell.from_utm_bounds` takes `crs` as a required argument.
- [ ] `load_cells` accepts a CSV in any single projected metre CRS equal (normalized) to
  `cell_crs`, and fails loudly on rows that disagree with it.
- [ ] `build_cube_dataframe` and `build_cube_dataframe_for_gee_utm` use the run's CRS.
- [ ] `mosaic.py` module docstring matches reality.
- [ ] Both §4b welds are closed: `exporter.py` / `mosaic.py` write `cell.crs`, the mosaic
  writer rejects a mixed-CRS grid, and a geographic or non-metre `cell_crs` is rejected at the
  entry point.
- [ ] Mode B warns when the AOI centroid lies outside `cell_crs`'s area of use.
- [ ] `scripts/.../infer_aoi_cubes.py` validates against the configured CRS, not the constant.
- [ ] Bow Valley output is unchanged. Regression, not just "tests pass": for a mode A and a
  mode B build on the existing config, the emitted cube CSV is byte-identical, the `GridCell`
  list is equal, and exported cube tifs and the daily mosaic carry the same CRS and transform
  as the pre-change build.
- [ ] A test builds a grid in a second CRS (mode A from a fixture CSV in another zone,
  mode B from a non-11N AOI), and asserts the emitted CSV `crs` column and `GridCell.crs`.
- [ ] Tests that assert against `CELL_TARGET_CRS` / `GRID_MATH_CRS` are updated to assert
  against the configured CRS rather than re-importing the constant.
- [ ] `make mypy` clean; no new suite failures.

## 6. Out of scope

- The adapter CRS audit (§4), separate task.
- Where the cells CSV and AOI live in config, done in the `legacy_csv` fix.
- Mixed-CRS cells *within one grid*. Cells stay in one CRS per run; the Landsat adapter
  already handles per-scene zone differences below the grid layer.

### Known risk, not fixed here

`CubeCache` keys entries as `{cell_id}/{day}_{modality}`, and `cell_id` is a positional
index. Two regions (or two AOIs) sharing one cache root silently serve each other's arrays.
This already happens with any AOI change; a second region makes it likely. Until fixed, use
a separate `processing_root` per region, or include the CRS and grid bounds in the cache
version stamp.
