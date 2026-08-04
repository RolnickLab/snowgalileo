# Potential Future Work

Known-suboptimal designs that are **not** currently paying rent in bugs. Each entry records
the diagnosis, the fix, what blocks it, and the trigger that would make it worth doing — so a
future session can act without re-deriving the analysis.

Nothing here is scheduled. An entry earning a schedule should graduate to its own `PLAN.md`.

Last updated: 2026-07-16

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
