# Potential Future Work

Known-suboptimal designs that are **not** currently paying rent in bugs. Each entry records
the diagnosis, the fix, what blocks it, and the trigger that would make it worth doing — so a
future session can act without re-deriving the analysis.

Nothing here is scheduled. An entry earning a schedule should graduate to its own `PLAN.md`.

Last updated: 2026-09-02

______________________________________________________________________

## 1. The cube filename is the transport channel for a model input

Status: **Diagnosed, low urgency. Do it the next time the loader is already being touched and
parity is being re-baselined.**

### Symptom

`LandsatEvalDataset._tif_to_array` (`src/snow_galileo/fsc/landsat_eval.py:259-267`) parses the
cell centre lat/lon **out of the cube's filename**, then feeds it to `to_cartesian`
(`landsat_eval.py:357`) to build the model's `static_x` location channels. The filename is
therefore not metadata — it is the wire carrying a tensor into the encoder.

Consequences:

- **Renaming or copying a cube silently changes its prediction.** Nothing about that should be
  true.
- Parsing is positional (`parts[2]`, `parts[3]`) and branches on an `LC`/`LE` prefix, with a
  TODO at `landsat_eval.py:258` already admitting the convention is brittle.
- The 4 dp rounding in `layout.cell_centre_lat_lon` is a silent ~11 m quantization of a model
  input, chosen so **filenames stay compact**. A storage concern picked the precision of a
  tensor.
- Producer and consumer must derive the same string independently, or a cube becomes
  unfindable — or worse, paired with the wrong cell.

The last point was the live one and is **already fixed** (2026-07-16): the derivation is now a
single `layout.cell_centre_lat_lon`, called by the exporter, the driver, and
`PrebuiltCubeSource`, pinned by a contract test in
`tests/test_local_sources/test_filename_contract.py`. That removed the drift risk. It did not
remove the layering inversion below.

### What is *not* the problem

The UTM→4326 reprojection itself is irreducible — do not "optimize" it away.

The model needs a **global** position: `to_cartesian` maps lat/lon onto the unit sphere. The
cell grid is UTM, i.e. zone-local metres, and the Landsat archive is genuinely mixed
32611/32612 — easting 600000 in zone 11 and in zone 12 are different places on Earth. A
globally-consistent location encoding therefore *requires* leaving zone-local metres. lat/lon
is only the waypoint to xyz because pyproj speaks lat/lon. The transform is one point per
cube and is `lru_cache`d. It costs nothing.

The real defect is a **layering inversion**: a file-organization artifact carries model
semantics.

### The fix

The raster already knows. Every cube carries `crs` + `transform`; the centre is derivable from
the file itself with no side channel:

```python
with rasterio.open(tif) as src:
    x, y = src.xy(src.height // 2, src.width // 2)
    lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(x, y)
```

The position is currently stored **twice** — once properly in the georeferencing, once lossily
in a string — and the model reads the lossy copy. This is the "native platform feature already
covers it" case: GeoTIFF georeferencing solves this and we re-solve it in the filename.

Target end state: **filename stays, semantics leave.** Keep the human-readable names — eyeballing
a cube directory is useful, and the name is a fine deterministic lookup key where a wrong guess
fails loudly with "file not found" — but have the loader take lat/lon from the transform. Then
rounding is harmless, renames are harmless, and the name is a label rather than a tensor. As a
bonus it deletes the `LC`/`LE` branch, since it works identically for GEE-era Landsat tifs.

### Alternatives considered and rejected

- **lat/lon in GeoTIFF tags** — strictly dominated. If the loader is being edited anyway, read
  the transform; do not add a second side channel.
- **Sidecar manifest (parquet/json) mapping cube → cell_id, lat, lon, day** — introduces a
  second source of truth that can drift from the directory; trades one consistency problem for
  another. Only worth it if a *catalog* is independently wanted (with ~84 GB of cubes and a
  per-day pre-flight in `PrebuiltCubeSource.missing`, plausibly — but that is a different
  motivation and should be argued on its own).
- **cell_id in the filename instead of lat/lon** — makes the cube non-self-describing, dependent
  on an external grid config matching. Worse than reading the transform.

### Blockers

1. **Bit-exact parity (the real one).** Dropping the 4 dp rounding shifts `static_x` by ~1.7e-6
   — physically meaningless (roughly 10× float32 epsilon, on a 1 km-resolution location prior)
   but enough to break bit-exactness against the GEE reference, which is what the AC-27 parity
   tests assert. Any attempt must re-baseline those, and should **measure** the embedding delta
   rather than assume it is negligible.
2. **"Downstream is sacred" (soft).** `_tif_to_array` is unmodified GEE-era code by policy. The
   change is ~6 lines and would improve both the GEE and direct-source paths, so this is a
   convention to renegotiate, not a technical barrier.

### Trigger

Do it when the loader is being modified for another reason **and** a parity re-baseline is
already on the table. Not worth a standalone change: post-consolidation this design is contained
and is not producing bugs.

______________________________________________________________________

## 2. Two band-description vocabularies for the same 308-band cube

Status: **Diagnosed. Step A (viewer reads both) is the agreed near-term unblocker; the `TS`
convention in `full_band_order()` is the agreed target end state, gated on an archive
migration.**

### Symptom

The same 308-band cube layout is described with two incompatible naming conventions, and a
third artifact carries no descriptions at all.

| Producer                                                           | Artifact                              | Convention                       | Example                                                    |
| ------------------------------------------------------------------ | ------------------------------------- | -------------------------------- | ---------------------------------------------------------- |
| `local_sources/exporter.py:391-406` (via `layout.full_band_order`) | hybrid cube `PR_*.tif`                | `<band>_t<idx>`, 0-based         | `VV_t0`, `B2_landsat_t3`, `QA60_t7`, `DEM`                 |
| `fsc/landsat_eval.py:1387-1701` (hardcoded 309-entry list)         | classic prediction `*_with_preds.tif` | `TS<n> <SENSOR> <band>`, 1-based | `TS1 S1 VV`, `TS4 LS B2`, `TS8 S2 cloud_flag`, `elevation` |
| `earthengine/eo_eval.py:485-488` (url-mode download)               | GEE cube `PR_*.tif_EPSG:*.tif`        | **none**                         | `(None, None, ...)`                                        |

The third row is the one that prompted this entry. `_export_for_polygon`'s `url` branch is a
raw `shutil.copyfileobj(r.raw, f)` of the GEE `GEO_TIFF` response. Earth Engine writes no band
descriptions and no `nodata` tag, and nothing reopens the file, so every cube produced by
`scripts/developer_scripts/fortress_mountain_basin/build_aoi_cubes_gee_url.py` lands bare.

Verified on disk (2026-09-02):

- `data/aoi_cubes/PR_20250401_*.tif` — 308 bands, float64, EPSG:32611, 100x100, all
  descriptions `None`, `nodata=None` **despite** `img.unmask(-9999)` at `eo_eval.py:423`.
- `data/fortress_validation_classic/raw/*.tif` — identical gap, so this is a url-mode exporter
  gap, not a hybrid-vs-classic difference.
- `data/output_tifs/fortress_validation_classic/*_with_preds.tif` — 309 descriptions present,
  plus dataset tags `description` / `prediction_model` / `processing_date` / `units`.

### Why it is not merely cosmetic

`local_sources/viewer/outputs.py` parses descriptions as its **only** band catalogue:

- `cube_variables()` (`:165`) — `raise ValueError(f"cube {path.name} has no band descriptions to catalogue")`
- `cube_availability()` (`:259`) — same guard
- `band_index()` (`:203`) — resolves `(var, timestep)` to a 1-based band **by description**;
  the docstring states this is deliberate, "never an arithmetic offset", so a future band-order
  change cannot silently mis-map a variable.
- parser: `_TIMESTEP_SUFFIX = re.compile(r"_t(\d+)$")` (`:39`), used at `:192` and `:290`.

Callers: `viewer/renderers.py:583-585` (`render_cube_band`) and
`scripts/developer_scripts/bow_valley_inference_local/data_viewer.py:93`.

Consequence: **GEE-URL cubes cannot be opened in the data viewer at all** — all three entry
points raise on descriptionless input. Descriptions are a functional dependency here, not
documentation.

### What is *not* the problem

**The classic hardcoded list is not out of order.** It was diffed against
`layout.full_band_order()` on 2026-09-02: all 308 input bands agree positionally. The only
differences are naming aliases, not sequence:

| `full_band_order()` | `landsat_eval.py` list |
| ------------------- | ---------------------- |
| `state_1km_t0`      | `TS1 MODIS cloud_flag` |
| `QA60_t0`           | `TS1 S2 cloud_flag`    |
| `QA_PIXEL_t0`       | `TS1 LS cloud_flag`    |
| `B2_landsat_t0`     | `TS1 LS B2`            |
| `DEM`               | `elevation`            |
| `Map`               | `ESA Worldcover Map`   |

It is nonetheless a **drift hazard**: 309 hand-typed strings duplicating a list that
`eo_eval.py:135` already derives from `MODALITIES`. It breaks silently if a modality is added,
removed, or reordered. Replace it with the derived list whenever `landsat_eval.py` is next
touched.

**The loader is indifferent.** `LandsatEvalDataset` keys off band *position*, never
descriptions (`rioxarray.open_rasterio` at `landsat_eval.py:252`). Naming is therefore free
from the model's point of view — the constraint comes entirely from the viewer and from human
readers of the deliverable.

### The fix

Two steps, deliberately separated because they have very different blast radii.

#### Step A — the viewer accepts both conventions (agreed, near-term)

One parse helper in `viewer/outputs.py` recognising both `_t(\d+)$` and `^TS(\d+)\s+`,
normalising to the viewer's existing **0-based** timestep axis (`TS<n>` maps to `n-1`), and a
`band_index()` that tries both spellings. Roughly 15 lines, one module, no data migration.

This is safe because the viewer holds **no hardcoded variable names**. Verified: var strings
flow out of the description parse into the `data_viewer.py` dropdowns (`:379`, `:386`, `:430`)
and straight back into `band_index()`. Only a docstring at `outputs.py:213` names `"VV"` /
`"DEM"`. Nothing else needs editing.

Off-by-one to respect: `TS` is 1-based; the viewer axis is 0-based throughout
(`_ts_label` renders `"t0"`, `data_viewer.py:426` parses back with `int(label[1:])`,
`render_cube_band` labels `@ t{timestep}`). Normalise at the parse boundary and nothing
downstream moves.

#### Step B — `TS` becomes the single convention (agreed target, gated)

`layout.full_band_order()` emits `TS<n> <SENSOR> <band>`; `exporter.py` and every new GEE cube
follow; `landsat_eval.py`'s hardcoded list is deleted in favour of the derived one. One
vocabulary across input cubes, prediction tifs, and the viewer.

`TS` is the better target for three reasons:

1. **It is the customer-facing vocabulary.** `*_with_preds.tif` is the deliverable; `TS1 S1 VV`
   is legible to a stakeholder opening it in QGIS, `VV_t0` is not.
2. **It carries the sensor**, which `_t<idx>` drops. That removes the need for the
   `B2_landsat` disambiguation suffix — `TS1 S2 B2` and `TS1 LS B2` are naturally distinct —
   and it groups the viewer's variable dropdown by sensor for free.
3. **It already exists and is shipped.** Adopting it is convergence, not a third invention.

Vocabulary decisions Step B must settle explicitly (these are *not* mechanical reformatting):

- **Cloud flags keep their EE band names — DECIDED 2026-09-02.** Use `TS1 S2 QA60`,
  `TS1 MODIS state_1km`, `TS1 LS QA_PIXEL`, and accept a documented divergence from the current
  `landsat_eval.py` strings (`MODIS cloud_flag` / `S2 cloud_flag` / `LS cloud_flag`).

  Three reasons, in increasing order of force:

  1. `QA60` **is** the GEE name, not a local invention. `earthengine/s2.py:44` sets
     `S2_CLOUD_FLAG_BANDS = ["QA60"]` and `get_s2_cloud_flag` (`s2.py:64-78`) selects it
     directly off `COPERNICUS/S2_HARMONIZED`. `state_1km` and `QA_PIXEL` are likewise genuine
     product band names. `<sensor> cloud_flag` is invented in exactly one place —
     `landsat_eval.py:1423-1425` — and nowhere else in the codebase.
  2. The reconstruction is the whole point of TASK-013c. `local_sources/s2.py:279-297` packs
     `MSK_CLASSI_B00.jp2` as `opaque<<10 | cirrus<<11` **to match that GEE band bit-for-bit**,
     and `test_s2_adapter.py:492-521` gates it at >=90 % pixel-exact agreement (AC-2). The name
     is what pins down what is being reproduced.
  3. **The rename is actively misleading, not merely lossy.** The three flags are three
     unrelated 16-bit layouts, not one schema across three sensors. Observed in one cube:
     `QA60` in `{0, 1024}` (bit 10 opaque, bit 11 cirrus); `state_1km` in
     `{1033, 1545, 1801, 32777, 36872}`; `QA_PIXEL` in `{22280, 23826, 23888, 24144}`. A common
     `cloud_flag` suffix invites decoding one with another's bit layout.

  **Caveat worth carrying into any use of QA60** (measured 2026-09-02, not a design issue —
  a property of the source): QA60 is present-but-identically-zero across the whole 2022-2023
  fortress archive. Scanned 7 dates from `20220309` to `20230524`: 264 QA60 bands carry data,
  **zero** of them non-zero. `data/aoi_cubes` `20250401`: 39 of 48 bands non-zero (`1024`).
  This is ESA Processing Baseline 04.00 (2022-01-25) ceasing QA60 production, reinstated
  ~PB 05.11 in early 2024. QA60 is a usable diagnostic for 2024+ cubes and carries no
  information for the 2022-2023 ones — do not read clear-sky from it there.

- **Statics**: `DEM` vs `elevation`, `Map` vs `ESA Worldcover Map`. Statics carry no `TS`
  prefix under either convention, so the viewer parses them identically; pick one spelling and
  state it in `layout.py`.

- The `MODALITIES` loop at `eo_eval.py:135` already knows each band's owning modality, so the
  `<SENSOR>` token is derivable — do **not** hand-type a second 308-entry list. A small
  modality-to-label map (`landsat` to `LS`, `viirs_fine` to `VIIRS`, ...) is the only literal
  needed.

#### Also in scope: metadata beyond descriptions

While `eo_eval.py:485-488` is open, three gaps close in the same edit:

1. **`nodata` is unset** on every GEE-URL cube. `img.unmask(-9999)` fills the gaps but the tif
   never declares the value, so any rasterio consumer masks nothing. This is arguably a more
   serious defect than the missing descriptions. Setting it is safe for the loader:
   `rioxarray.open_rasterio` defaults to `masked=False, mask_and_scale=False`, so raw values
   are preserved.
2. **Dataset tags** mirroring `landsat_eval.py`'s: `description`, `processing_date`,
   `source="Google Earth Engine"`, `window_start` / `window_end`, `nodata_value`.
3. **Per-band tags** — `sensor`, `acquisition_date` (the real calendar date, derivable from
   `window_end` and `DAYS_PER_TIMESTEP`), `native_resolution_m` (from
   `MODALITIES[...]["original_resolution"]`). Strictly richer than the classic artifact, which
   encodes the timestep only positionally in a prose blurb. GDAL band tags do not collide with
   the description parse.

#### Where the write belongs

Preferred: **`eo_eval._export_for_polygon`, `url` branch (`:485-488`)** — reopen the downloaded
file `r+` and stamp descriptions, `nodata`, and tags. It is the single point where the file is
created, so it covers `export_from_csv_utm_native` (the
`build_aoi_cubes_gee_url.py` path) *and* the classic `export_from_csv_utm`, for about 20 lines.

Guard it: only stamp when `src.count == len(names)`, otherwise log a warning and leave the file
alone. A silent mislabel is worse than no label.

Note `cloud` and `drive` modes write to GCS/Drive and cannot be post-processed this way; they
would keep landing bare until separately addressed.

Rejected alternative: a post-pass loop inside `build_aoi_cubes_gee_url.py`. It re-reads every
cube a second time, fixes only one of the two callers, and puts a data-format contract in an
operator script instead of in the exporter that owns it.

#### Second-order fix: `crop_classic_cubes.py` drops descriptions

`scripts/developer_scripts/fortress_mountain_basin/crop_classic_cubes.py:114-122` builds
`profile = src.profile | {...}` and writes. `src.profile` does **not** carry band descriptions,
so any metadata added upstream is silently lost on crop and never reaches
`data/fsc_inference_fortress_100m_cropped`. Needs `dst.descriptions = src.descriptions` (and
the dataset tags) forwarded — one line, but without it the whole change is invisible at the end
of the chain.

### Blockers

1. **5.9 TB of existing cubes, for Step B only.** `data/bow_valley_processing` is a symlink to
   `/archive/data/ai4snow/bow_valley_processing`: **476,135 cubes, 5.9 TB**, every one carrying
   `_t<idx>` descriptions written by `exporter.py:406`. Sample verified 2026-09-02:
   `('VV_t0', 'VH_t0', 'angle_t0', ...)` through `('QA60_t7', 'QA_PIXEL_t7', 'DEM', 'slope', 'aspect', 'Map')`, `nodata=-9999.0`. A `TS`-only viewer stops reading all of them.
   Rewriting is metadata-only (`rasterio.open(path, "r+")`, no pixel touch) so it is cheap per
   file, but it is 476k opens against archive storage and wants an idempotent, resumable,
   dry-runnable migration script. **Step A exists precisely so this migration is not on the
   critical path.**
2. **Test fixtures encode the convention.** `tests/test_local_sources/test_viewer_outputs.py`
   (`:223`, `:248`), `test_viewer_output_renderers.py:49`,
   `test_tracer_end_to_end.py:146`, `test_exporter_parity.py:84-88`. The last one derives its
   index via `full_band_order().index(f"{name}_t{ts}")`, so it follows automatically once the
   list changes; the viewer fixtures need explicit both-convention cases under Step A.
3. **Vocabulary decisions above are unresolved** and should not be made mid-implementation.

### Trigger

- **Step A**: as soon as anyone needs to inspect a GEE-URL cube in the data viewer. Currently
  blocking — `data/aoi_cubes` and `data/fortress_mountain_basin` (550 cubes) are unopenable.
  Independent of Step B; no migration.
- **Metadata stamping in `eo_eval.py`** (descriptions + `nodata` + tags): next time AOI cubes
  are generated. It is additive, and the missing `nodata` is a live correctness gap on every
  cube produced so far.
- **Step B**: when an archive re-generation or migration window is already open for another
  reason. Not worth standing up a 476k-file migration on its own; Step A removes the pressure.
