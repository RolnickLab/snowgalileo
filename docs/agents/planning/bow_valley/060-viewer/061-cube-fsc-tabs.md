# PLAN v2 — Add cube + daily-FSC tabs to the data viewer

*Formerly `clip-viewer/PLAN-V2-CUBE-FSC-TABS.md`.*

Extends the clip-viewer (`PLAN.md`) with **two new tabs** that visualise Stage-2
**outputs** (not the clipped archive): assembled per-cell cubes and daily fractional
snow-cover (FSC) COGs. Same UX as the existing clip tab: a leafmap map with the AOI
outline and the selected item placed on it.

## 1. Goal

- **Tab "Clip"** — the existing source/product viewer, unchanged (moved into a tab).
- **Tab "Cube"** — inspect one assembled cube **layer by layer, date by date**:
  pick prediction date → cell, then a **selection mode** ("Select by timestep" /
  "Select by variable") that orders and cross-filters the **variable** (38 dynamic + 4
  static) and **timestep** (t0–t7) dropdowns by *data availability* (see "Cube selection
  modes" below). The selected band renders on the map (EPSG:32611 → 4326), AOI overlaid.
- **Tab "Daily FSC"** — a **date slider** over every `fsc_<date>.tif` in `daily_fsc/`;
  the selected day's FSC (0–1) renders **turbo**-colormapped on the map with a continuous
  0–1 colour scale (bottom-right), AOI overlaid, nodata transparent, zoomed to the data
  footprint.

### Cube selection modes (data-availability filtering — revises §3/§7)

Every dynamic `(var, timestep)` band **exists** in the cube, but most are **entirely
nodata (`-9999`)** at most timesteps — e.g. Landsat reflectance is real at only one or two
of the eight timesteps, S1 may be all-nodata in a cube. So "exists at timestep" is a
**data** property (band not all-nodata), **not** a structural/description property. This is
computed by `cube_availability(path)` reading every band once (~0.5 s on the ~100×100
cubes; memoised per path in the viewer).

- **Select by timestep** — timestep dropdown first (full `t0..t7`); the variable dropdown
  then lists only variables that carry real data at that timestep, **plus all statics**
  (statics have no timestep axis and are always selectable).
- **Select by variable** — variable dropdown first (all dynamic + statics); the timestep
  dropdown then lists only the timesteps at which that variable is real. A static shows
  "no timestep"; an all-nodata dynamic var shows a warning.

Timestep is a **dropdown** (`Select`), not a slider. Stale selections clamp to the valid
set when date/cell/mode changes, so an all-nodata band is never rendered.

## 2. Scope & non-goals

**In scope**

- List cubes from `processing_root/cubes/` and FSC COGs from `processing_root/daily_fsc/`
  by filesystem scan (these are pipeline outputs, **not** clip-manifest rows — no
  `load_products` reuse).
- Reuse the existing render contract end-to-end: `QuicklookResult` (`georef_raster`),
  `result_to_geotiff`, `_to_4326`, leafmap `add_raster` + AOI overlay.
- Single-band display for both tabs; cube bands are diagnostic, FSC is a continuous field.

**Non-goals**

- No re-export, no inference, no writes into any `data/` tree (read-only, like the clip tab).
- No RGB compositing of cube bands (single-band inspection is the point; revisit later).
- No click-to-select-cell map interaction (date→cell dropdowns; user-chosen).
- No new heavy deps — solara + leafmap + rasterio already present.

## 3. Data contracts (verified on disk)

- **Cube** `PR_<YYYYMMDD>_<lat>_<lon>_SC00.tif`: 308 bands, **EPSG:32611**, 100×100 @ 10 m,
  `float32`, nodata **-9999**, **band descriptions present** (`VV_t0`…`QA_PIXEL_t7`,
  then `DEM`, `slope`, `aspect`, `Map`). Layout: 38 dynamic bands × 8 timesteps, then
  4 statics. Band index for dynamic var `v` at timestep `t` = `38*t + dyn_offset(v) + 1`;
  statics are the last 4 (305–308).
- **Daily FSC** `fsc_<YYYYMMDD>.tif`: single band, EPSG:32611, nodata -9999, values ∈ [0,1]
  (tiny float spill just outside is clamped for display).
- Filename → prediction date parsed from the `PR_<YYYYMMDD>` / `fsc_<YYYYMMDD>` token.

## 4. Module layout (additive — touches only the viewer package + entrypoint)

```
src/data/local_sources/viewer/
  outputs.py     # NEW: scan cubes/ + daily_fsc/, parse filenames → CubeRow / FscRow,
                 #      cube band-name catalogue (dynamic vars, statics, timesteps)
  renderers.py   # ADD: render_cube_band(row,var,timestep,long_edge) -> QuicklookResult
                 #      render_fsc(path,long_edge) -> QuicklookResult (colormapped)
  settings.py    # ADD: processing_root + cubes_dir/daily_fsc_dir (LocalPaths default,
                 #      VIEWER_* overridable)
scripts/developer_scripts/bow_valley_inference_local/data_viewer.py  # WRAP existing Page body in solara.lab.Tabs;
                                          # add CubeTab + FscTab components
```

No edits to `quicklook.py` contract, `manifest.py`, `aoi.py`, `archives.py`, or any
`src/fsc/*` / loader / downstream code. The two new renderers are **not** registered in
the `RENDERERS` source-dispatch dict (that dict is keyed by clip *source*); they are
called directly by the new tabs, returning the same `QuicklookResult` the map path
already consumes.

## 5. `outputs.py` (new)

- `@dataclass(frozen=True) CubeRow(path, pred_date: date, lat: float, lon: float, cell_label: str)`.
- `@dataclass(frozen=True) FscRow(path, pred_date: date)`.
- `list_cubes(settings) -> list[CubeRow]` — glob `cubes/PR_*.tif`, parse via the existing
  `CUBE_FILENAME_REGEX` (reuse `layout` regex; do not re-invent the format), sort by
  (date, lat, lon). `list_fsc(settings) -> list[FscRow]` — glob `daily_fsc/fsc_*.tif`,
  parse date, sort.
- `dates_for_cubes(rows) -> list[date]`; `cubes_for_date(rows, d) -> list[CubeRow]`.
- **Band catalogue** read from the cube's own descriptions (single source of truth, not a
  hardcoded list): `cube_variables(path) -> (dynamic: list[str], statics: list[str], n_ts: int)`
  by stripping the `_t<idx>` suffix from descriptions and de-duplicating in order; the 4
  trailing un-suffixed names are the statics. `band_index(path, var, timestep) -> int`
  resolves the 1-based rasterio band for `(var, timestep)` by matching the description
  `f"{var}_t{timestep}"` (dynamic) or `var` (static) — **match by description, never by
  arithmetic offset**, so a future band-order change can't silently mis-map.
- **Data-availability catalogue** (drives the cube selection modes): `cube_availability(path) -> CubeAvailability(dynamic_real: dict[str, set[int]], dynamic_order, statics, n_timesteps)`
  reads **every band once** and marks a `<var>_t<i>` band *available* iff it is **not
  entirely `-9999`/non-finite**. Helpers `vars_at_timestep(avail, t)` (real dynamic vars at
  `t`, then all statics) and `timesteps_for_var(avail, var)` (real timesteps for a dynamic
  var; `[]` for a static) feed the cross-filtered dropdowns.

## 6. Renderers (added to `renderers.py`)

- `render_cube_band(*, path, var, timestep, statics, long_edge) -> QuicklookResult`:
  resolve band via `band_index`, decimated `_read_band_decimated` (EPSG:32611 → reproject
  4326 via existing `_to_4326`), mask nodata -9999 → NaN, **percentile-stretch to uint8**
  for display (cube bands span wildly different domains: dB, reflectance, Kelvin), label
  `f"{var} @ t{timestep}"` (or `f"{var} (static)"`). Returns `georef_raster`.
- `render_fsc(*, path, long_edge) -> QuicklookResult`: decimated read, nodata -9999 → NaN,
  **apply a fixed [0,1] colormap** (matplotlib **`turbo`**) → uint8 RGB (fixed scale, NOT
  percentile — FSC is an absolute fraction; a per-image stretch would lie), clamp to [0,1],
  NaN → forced pure black → transparent (handled by `result_to_geotiff`'s alpha-on-zero).
  Returns `georef_raster`. Companion `fsc_colorbar() -> (hex_stops, vmin, vmax)` samples 11
  stops (0.0,0.1,…,1.0) **from the same colormap** so the on-map legend and the pixels
  cannot drift.
  - **Colormap choice (revises the original Blues/viridis note):** `turbo` reads clearly
    over the dark satellite basemap *and* gives distinct mid-range steps (0.3–0.7) so
    partial snow cover is legible — viridis's dark-purple low end and `cool`'s muddy
    periwinkle midband were both rejected after viewing the real COG. `turbo(0)=(48,18,59)`
    is dark but **non-black**, so valid FSC=0 keeps a non-zero colour band and isn't dropped
    by `result_to_geotiff`'s all-zero-RGB transparency heuristic; only the forced-black
    NaN/nodata pixels are.
  - **Sparsity (observed on the real COG):** valid FSC covers only ~2 % of pixels but in
    coherent patches (no salt-and-pepper) scattered across ~the whole AOI. The FSC tab
    therefore zooms to the **data footprint** (span-aware zoom, not the empty extent) and
    renders at `opacity=1.0`.

## 7. Tab wiring (`data_viewer.py`)

- Wrap the current `Page` map+sidebar logic into a `ClipTab` component (verbatim move).
- `CubeTab`: date `Select` → cell `Select` (CubeRow labels) → **mode `ToggleButtonsSingle`**
  ("Select by timestep" / "Select by variable") → two cross-filtered `Select` dropdowns
  whose **order follows the mode** (free axis first, dependent axis second), filtered by
  `vars_at_timestep` / `timesteps_for_var` over the memoised `cube_availability`. Timestep
  is a **dropdown**, not a slider; statics ignore it. Builds the `QuicklookResult`, writes
  it via `result_to_geotiff`, places it with `add_raster` + AOI overlay
  (`_render_on_map(result, key)` shared helper).
- `FscTab`: date `SliderInt` over `list_fsc` dates → `render_fsc` → `_render_on_map(..., zoom_to_data=True, colorbar=fsc_colorbar()+caption)` (turbo gradient + 0–1 scale,
  bottom-right). With a single date on disk the slider is replaced by an info line.
- `Page` becomes `solara.lab.Tabs([Tab("Clip", ClipTab), Tab("Cube", CubeTab), Tab("Daily FSC", FscTab)])`.
- Empty-state: if `cubes/` or `daily_fsc/` is empty, the tab shows a `solara.Info`
  ("no cubes exported yet — run export_bow_valley_cube.py") instead of crashing.

## 8. Tests (`tests/test_local_sources/test_viewer_outputs.py`, new — pure, no Solara/leafmap)

- `list_cubes` / `list_fsc` parse filenames → correct date/lat/lon, sorted, empty-dir → `[]`.
- `cube_variables` returns 38 dynamic + 4 statics + n_ts==8 from a tiny synthetic
  multi-band GeoTIFF with descriptions.
- `band_index` maps `(var, timestep)` to the right 1-based band by description; raises on
  an unknown var.
- `cube_availability` marks a band real iff not all-nodata; `vars_at_timestep` filters
  dynamics yet always appends statics; `timesteps_for_var` returns real timesteps only and
  `[]` for a static (synthetic cube with deliberately all-nodata bands at some timesteps).
- `render_fsc` on a synthetic 1-band 32611 COG returns a `georef_raster` in [0,1] domain,
  nodata→NaN handled.
- `render_cube_band` on a synthetic described cube returns the requested band, georef.
- (Solara components are not unit-tested — manual smoke via `solara run`, per PLAN §8 style.)

## 9. Verification

```bash
cd /home/dev/projects/presto-v3
uv run pytest tests/test_local_sources/test_viewer_outputs.py -v
uv run ruff check src/data/local_sources/viewer/ scripts/developer_scripts/bow_valley_inference_local/data_viewer.py
uv run mypy  src/data/local_sources/viewer/outputs.py src/data/local_sources/viewer/renderers.py
# Manual: uv run solara run scripts/developer_scripts/bow_valley_inference_local/data_viewer.py
#   → Clip / Cube / Daily FSC tabs; cube var+timestep on map; FSC date slider on map; AOI on all.
# Full-suite delta (TEST_BASELINE.md): NEW-failures list MUST be empty. NOT pytest -x.
```

## 10. Delivery order (incremental, approval-gated per CLAUDE.md — STOP after each)

1. `settings.py` paths + `outputs.py` + its tests (Red→Green). STOP.
2. `render_cube_band` + `render_fsc` + their tests (Red→Green). STOP.
3. `data_viewer.py` tabs (ClipTab move + CubeTab + FscTab), manual smoke. STOP.
4. **Cube selection modes + FSC display polish** (this revision): `cube_availability` +
   `vars_at_timestep` + `timesteps_for_var` + their tests (Red→Green); CubeTab mode toggle
   - dropdown timestep + cross-filtering; FSC `turbo` colormap + `fsc_colorbar` on-map scale
   - zoom-to-data. Manual smoke. STOP.
5. Docs/checkoff. Commit only on approval.
