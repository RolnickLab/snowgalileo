# PLAN: Clipped-archive visual validation viewer

*Formerly `clip-viewer/PLAN.md`.*

## 1. Goal

A quick, reusable way to **visually sanity-check** the products in
`data/clipped_bow_valley_selection_raw/` after the TASK-002 clip stage — confirm
each product (a) is non-empty, (b) sits inside `data/bow_valley_inference_aoi.geojson`, and (c) looks
physically plausible per sensor. Covers all four content classes the user named:
NetCDF (ERA5), nested archives (S2/S3, Landsat), plain GeoTIFFs
(DEM/WorldCover/MODIS/VIIRS + the processed S1 SNAP tif), and an **AOI overlay on every
product**.

This is a **developer/QA tool**, deliberately separate from the TASK-001…016
ingestion adapters. It reads the clipped archive read-only; it writes nothing
into any `data/` tree.

## 2. Why build vs. reuse (decision record)

- `xarray.plot()` / QGIS / leafmap each solve *part* of this, but none gives a
  single "pick a clipped product → see it on a basemap with the AOI overlaid,
  archives cracked transparently, ERA5 stepped by date" view. See chat analysis.
- We build a thin **Solara + leafmap** app that orchestrates existing libs
  (rasterio, rioxarray, xarray, GDAL `/vsizip//vsitar/`). No new rendering engine
  — just glue + per-sensor quicklook routines.

## 3. Scope & non-goals

**In scope**

- Read `clip_manifest.csv`, list every product with its `action`/overlap/valid-px.
- Render selected product on a leafmap map with `data/bow_valley_inference_aoi.geojson` outlined.
- Per-sensor quicklook routines (see Contract §5).
- ERA5: variable picker + **date slider** over `valid_time`.
- Surface manifest metadata (overlap km², valid-pixel count, action) in the UI so
  `SKIP_*` products are visible as *why-skipped*, not just absent.

**Non-goals**

- No re-clipping, no reprojection-to-cell-grid, no band assembly (that's the
  adapter layer, TASK-006…014). The viewer renders what is on disk.
- No write-back, no caching layer, no auth, no deployment. Localhost dev tool.
- Not a replacement for the TASK-002 `clip_audit.py` numeric audit — this is the
  *visual* complement to it.

## 4. Known constraints / gotchas (verified on disk — Phase-0 spike confirmed)

**Phase-0 spike findings (revise §5):**

- **F1 — products are deeply nested.** DEM lives at
  `dem/DEM1_SAR_.../Copernicus_..._DEM.tif`, not flat. → `manifest.py` resolves
  every product through the manifest `output_path` column (single source of
  truth); no globbing fixed subdirs.
- **F2 — Landsat is genuinely MIXED-ZONE.** Path 042024 scenes = EPSG:32612,
  path 042025 = EPSG:32611 (verified per-scene). Not a clip bug; vindicates the
  dynamic-CRS rule. Renderer hardcodes no zone; leafmap reprojects each scene.
- **F3 (CORRECTED, then SUPERSEDED 2026-06-11) — S1 is read from the processed SNAP tif,
  not GCP-warped.** *Original finding:* the raw/clipped S1 measurement TIFF carries GCPs
  in EPSG:4326 and renders via on-the-fly GCP warp (`rasterio.warp` with `gcps=`). *Now:*
  S1 is no longer clipped — it is processed via ESA SNAP into a per-granule, map-projected
  `sentinel1_snap/s1_grd_*.tif` (EPSG:32611, terrain-corrected σ⁰). The viewer discovers S1
  from that cache (not the clip manifest) and the renderer reads band 2 (VV, linear σ⁰) as
  a plain `georef_raster` → dB — **no GCP warp, no zip**. See
  [`../raw-data-ingestion/PLAN-S1-PERGRANULE-SNAP.md`](../raw-data-ingestion/PLAN-S1-PERGRANULE-SNAP.md).
  Only **S3** remains non-georeferenced (its geolocation is a separate per-pixel
  `geo_coordinates.nc`, no GCPs in the radiance file).
- **F4 (RESOLVED in clip stage) — ERA5 manifest rows were not month-disambiguable.**
  Root cause: clippers recorded `output_path = dst_path.name`, dropping the
  `<YYYYMM>_ERA5LAND/` subdir, so the 3 monthly copies of each wind/temp var
  collapsed to identical rows. **Fixed** in `clip/orchestrator.py`: `output_path`
  is now the path RELATIVE TO THE SOURCE ROOT (the column's documented contract);
  the combined + era5 per-source manifests were regenerated (era5 re-clipped). All
  15 ERA5 rows now have unique `output_path`; viewer resolves them to 15 distinct
  files. Precip vars are flat (`<YYYYMM>_ERA5LAND_totalprecip.nc`); wind/temp nest
  under `<YYYYMM>_ERA5LAND/`. (No filesystem-walk workaround needed anymore.)

1. **S3 OLCI is not a regular grid.** `Oa*_radiance.nc` carry data; geolocation is
   a separate `geo_coordinates.nc` (per-pixel lat/lon), no affine/CRS. → S3 is
   rendered as a **non-georeferenced quicklook** (raw radiance array, no basemap
   overlay), with a UI note. We will NOT fabricate a CRS. (Optional later: scatter
   the lat/lon as points — out of scope v1.)
2. **Landsat CRS is per-scene EPSG:32612** (USGS WRS-2 zone, not AOI zone — see
   TASK-002 REVIEW_AUDIT verdict #2). Quicklook must read each band's CRS
   dynamically; leafmap reprojects to web-mercator for display.
3. **S2 is EPSG:32611 JP2** read via `/vsizip/` (GDAL needs the JP2 driver, present in
   the project's rasterio build — verify in subtask 0). **S1 is no longer an archive read**
   — it is the processed `sentinel1_snap/s1_grd_*.tif` (EPSG:32611, plain GeoTIFF; see F3).
4. **Archives are genuinely cropped** (clipped S2 88 MB vs raw 229 MB), so the
   viewer shows the clipped extent, which is the point.
5. **Manifest I/O = pandas** (user decision). `polars` is not installed; `pandas`
   1.5.3 is present transitively. Manifest is tiny (≈ a few hundred rows). Pin
   pandas explicitly in the dev group so the viewer doesn't lean on a transitive.
   This is a deliberate, user-approved deviation from the CLAUDE.md polars default
   (scoped to this dev tool only).
6. ERA5 NetCDFs are tiny regular grids (16×20 × 31 days), lat/lon coords, EPSG:4326
   — trivially georeferenceable via rioxarray `write_crs(4326)`.
7. **Big-file guard:** the largest quicklook sources (raw S1 measurement TIFFs were
   ~146 MB; processed S1 SNAP tifs are AOI-cropped and smaller, but the guard still
   applies generally). Quicklooks MUST use rasterio `overview`/decimated reads
   (`out_shape`), never full-res loads
   (geospatial skill: no eager multi-GB loads).

## 5. Contract — `quicklook(product) -> QuicklookResult` (define BEFORE impl)

Single interface every modality implements. Drafted in `CONTRACT.md`, stubbed in
`viewer/quicklook.py` as an ABC/protocol before any renderer is written.

```
QuicklookResult:
  kind: Literal["georef_raster", "plain_image"]
  # georef_raster: a COG/array leaflet can place via bounds+CRS (-> leafmap layer)
  # plain_image:   a PNG/ndarray with no map placement (S3, corrupt fallback)
  array_or_path: np.ndarray | Path     # rendered RGB/single-band, decimated
  bounds_4326: tuple[float,float,float,float] | None   # None => plain_image
  src_crs: str | None
  label: str                            # e.g. "S2 TCI (B4/B3/B2)"
  note: str | None                      # e.g. "S3 OLCI: non-georeferenced"
```

Per-modality renderers (each ≤ ~40 lines, dispatched by `source`):

| source        | quicklook                                                                      | bands / var                                                  |
| ------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| dem           | single-band, terrain cmap                                                      | elevation                                                    |
| worldcover    | single-band, class palette                                                     | Map                                                          |
| modis / viirs | RGB or single SR band                                                          | sur_refl b1/b4/b3 (MODIS), I1 (VIIRS)                        |
| landsat8/9    | georef RGB true-color (decimated, per-scene CRS)                               | B4/B3/B2 from tar via /vsitar/                               |
| sentinel2     | georef RGB true-color (decimated)                                              | B04/B03/B02 jp2 via /vsizip/                                 |
| sentinel1     | **georef** dB-stretched (SNAP terrain-corrected, plain GeoTIFF; F3 superseded) | VV (band 2, linear σ⁰→dB) from `sentinel1_snap/s1_grd_*.tif` |
| sentinel3     | plain_image, single radiance (NO geo)                                          | Oa08_radiance                                                |
| era5          | georef single-band + date slider                                               | tp / t2m / winds, valid_time idx                             |

Failure contract: any unreadable/corrupt product → `plain_image` placeholder +
`note` with the exception class; never crash the app, never a silent blank.

## 6. Architecture

**Decisions (locked):** module home = `src/data/local_sources/viewer/` (moved
from `src/viewer/` post-Phase-5 — viewer scopes to local clipped sources); deps =
`solara`+`leafmap` (dev group); manifest I/O = `pandas`.

```
scripts/developer_scripts/bow_valley_inference_local/data_viewer.py   # Solara entrypoint (`solara run`)
src/data/local_sources/viewer/
  manifest.py     # load clip_manifest.csv -> list[ProductRow] (pandas)
  quicklook.py    # QuicklookResult, Protocol, dispatch-by-source
  renderers.py    # per-modality renderers (§5 table)
  archives.py     # /vsizip/ /vsitar/ path builders + member discovery
  aoi.py          # load data/bow_valley_inference_aoi.geojson -> geojson layer + bounds
```

- Solara reactive state: selected `source`, selected `product_id`, ERA5
  `(variable, date_idx)`. Map rebuilds layer on change.
- Pydantic-settings `ViewerSettings`: clipped root, aoi path, decimation target
  (default 1024 px long edge), default basemap. No magic numbers.

## 7. Dependencies to add (dev group) — APPROVED

`solara`, `leafmap`, and pin `pandas` explicitly. Already present: rasterio (via
rioxarray), xarray, rioxarray, numpy, matplotlib, geopandas, h5netcdf. Add under a
`dev`/`viz` group in `pyproject.toml`, NOT core runtime deps (keeps the ingestion
pipeline lean). User approved the add.

## 8. Phased delivery (incremental review per CLAUDE.md — STOP after each)

- **Phase 0 — spike (no app):** verify GDAL can read one JP2 via /vsizip/, one
  Landsat band via /vsitar/, one ERA5 var, decimated. Confirm leafmap+solara
  import. ~30 lines throwaway. → STOP, report what reads / what doesn't.
- **Phase 1 — contract:** write `CONTRACT.md` + `quicklook.py` stubs + `manifest.py`
  - `aoi.py`. No rendering. → STOP for approval.
- **Phase 2 — GeoTIFF renderers + map shell:** dem/worldcover/modis/viirs +
  Solara map with AOI overlay + product picker. The "GeoTIFFs already easy, but
  one-stop with AOI" win lands here. → STOP.
- **Phase 3 — archive renderers:** landsat, S2 via vsi paths, decimated RGB; S1 from the
  processed SNAP GeoTIFF (F3 superseded — no longer an archive read). → STOP.
- **Phase 4 — ERA5 + S3:** ERA5 date slider; S3 plain_image with note. → STOP.
- **Phase 5 — polish + docs:** README run instructions, ruff/mypy clean, a couple
  of unit tests on `manifest.py`/`archives.py` path building (pure funcs, no I/O).

## 9. Acceptance

- `solara run scripts/developer_scripts/bow_valley_inference_local/data_viewer.py` opens; every manifest row
  selectable; CLIP rows render, SKIP rows show their skip reason.
- AOI outline visible on every georef_raster product.
- ERA5 date slider steps through valid_time; values change.
- S3 renders without crashing, labeled non-georeferenced.
- No full-res >100 MB read (decimated reads only).
- ruff + mypy clean on `src/data/local_sources/viewer/` + the entrypoint; manifest/archive unit tests green.
- Suite introduces zero new failures vs TEST_BASELINE.md.

## 10. Decisions (resolved)

1. ✅ Add `solara`+`leafmap` to a dev dep group.
2. ✅ Module home: `src/data/local_sources/viewer/` (importable/testable; moved
   from `src/viewer/` — the viewer validates only local clipped sources, so it
   belongs under that package).
3. ✅ Manifest I/O: `pandas` (present transitively; pin in dev group).
