# Clip Viewer

A developer / QA tool for **visual validation of the AOI clip stage**
(TASK-002). Pick a clipped product from `clip_manifest.csv`, see its quicklook
rendered on a basemap with the `data/bow_valley_inference_aoi.geojson` outline overlaid, and read the
clip-stage metadata (overlap km², valid-pixel count, action) beside it.

It exists because the clip-stage unit tests gate on a *valid-pixel count*, which
can stay high while the science array is wrong (a sheared window or an empty
`(0,0)` clip padded by full-copied non-science datasets). The eye catches what
the count cannot — this viewer surfaced both the MODIS/VIIRS sinusoidal-shear
bug and the S3 `scale_factor` bug. See `docs/agents/KNOWLEDGE.md`.

## Run

```bash
uv run solara run scripts/developer_scripts/data_viewer.py
```

Opens at <http://localhost:8765>. The page loads every manifest row; select a
**source**, then a **product**. Single-band rasters (DEM, WorldCover) and RGB
composites (MODIS, VIIRS, Landsat, S2) render georeferenced on the map. ERA5
adds a **date slider** (one frame per available time step). Sentinel-3 OLCI is
swath geometry with no map CRS, so it renders as a plain grayscale image in a
side panel rather than on the basemap.

Reads the clipped archive **read-only**; writes only transient decimated
GeoTIFFs to a temp dir (`clip_viewer_*`), cleaned on exit.

## Configuration

All via `VIEWER_*` env vars (pydantic-settings, see `settings.py`):

| Setting | Env var | Default |
|---|---|---|
| Clipped root | `VIEWER_CLIPPED_ROOT` | `data/clipped_bow_valley_selection_raw` |
| AOI GeoJSON | `VIEWER_AOI_PATH` | `data/bow_valley_inference_aoi.geojson` |
| Manifest name | `VIEWER_MANIFEST_NAME` | `clip_manifest.csv` |
| Decimation long edge (px) | `VIEWER_LONG_EDGE` | `1024` |
| Basemap | `VIEWER_DEFAULT_BASEMAP` | `Esri.WorldImagery` |

## How it works

* **`manifest.py`** — loads `clip_manifest.csv` into `ProductRow`s and resolves
  each `output_path` to a real file/dir (flat file, nested DEM basename via
  `rglob`, or a MODIS/VIIRS per-grid directory).
* **`archives.py`** — GDAL `/vsizip/` + `/vsitar/` path builders so Landsat tar
  and S2/S1 zip bands are read *inside* the archive, decimated, without
  extraction (never materialises the ~146 MB S1 measurement TIFF).
* **`renderers.py`** — one renderer per modality. All reads are **decimated**
  (`out_shape`/overview). Native CRS is honoured per source (MODIS Sinusoidal,
  Landsat EPSG:32612, S2 EPSG:32611, S1 GCPs in 4326) and reprojected to a true
  4326 grid for display. `result_to_geotiff` writes a **finite** nodata
  (`-9999.0`, never `NaN`) plus an alpha band — `localtileserver`'s
  `/api/metadata` 500s on non-finite nodata.
* **`quicklook.py`** — `render_product` dispatches to the registered renderer
  and returns a `QuicklookResult` (with a failure contract, never a raw
  traceback to the UI).

## Caveats

* **Do not trust the manifest `valid_pixel_count` as a science-pixel metric.** It
  counts every grid-matching 2D dataset plus full-copied non-science datasets, so
  it is inflated (≈100× for S3). Judge clip correctness by the rendered science
  array, not the count.
* Display GeoTIFFs need a finite nodata. If `/api/metadata` 500s, suspect a
  `NaN` nodata leaking back in (memory: `clip-viewer-tileserver-nan-nodata`).
* This is a QA tool, not a pipeline stage — it writes nothing under
  `data/clipped_bow_valley_selection_raw`.
