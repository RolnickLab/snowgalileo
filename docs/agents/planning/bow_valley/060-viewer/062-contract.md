# CONTRACT: clip-viewer rendering interface

*Formerly `clip-viewer/CONTRACT.md`.*

Contract-first per CLAUDE.md. Every modality renderer implements ONE function
shape; the Solara app and the map layer depend only on this contract, never on a
renderer's internals.

## Data types

### `ProductRow` (from `manifest.py`)

One row of `clip_manifest.csv`, resolved to an on-disk path.

```
product_id:        str
source:            str   # dem|worldcover|modis|viirs|landsat8|landsat9
                         # |sentinel1|sentinel2|sentinel3|era5
footprint_bbox:    tuple[float,float,float,float]  # minx,miny,maxx,maxy (4326)
intersects:        bool
aoi_overlap_km2:   float
valid_pixel_count: int
action:            Literal["CLIP","SKIP_NO_OVERLAP","SKIP_DEGENERATE_OVERLAP"]
path:              Path | None   # resolved from output_path; None for SKIP rows
```

Rule (F1): `path` is derived from the manifest `output_path` column joined to the
clipped root — the manifest is the single source of truth for product location.
SKIP rows have `path=None` and are listed but not renderable (UI shows skip reason).

### `QuicklookResult` (from `quicklook.py`)

```
kind:           Literal["georef_raster","plain_image"]
image:          np.ndarray            # HxW (gray) or HxWx3 (RGB), uint8 or float
bounds_4326:    tuple|None            # (minx,miny,maxx,maxy); None => plain_image
src_crs:        str|None              # native CRS of the rendered raster
label:          str                   # e.g. "S2 true-color (B04/B03/B02)"
note:           str|None              # e.g. "S3 OLCI: non-georeferenced (no CRS)"
```

- `georef_raster` → leafmap places it via `bounds_4326` (reprojected from `src_crs`).
- `plain_image` → shown in a side panel, NOT on the map (S3, error fallback).

## Renderer protocol

```python
class Renderer(Protocol):
    source: str
    def render(self, row: ProductRow, *, long_edge: int) -> QuicklookResult: ...
```

- Dispatch by `row.source` via a registry `RENDERERS: dict[str, Renderer]`.
- `long_edge` is the decimation target (ViewerSettings, default 1024). Renderers
  MUST use decimated/overview reads (`out_shape`) — never full-res (large sources, e.g.
  raw S1 was ~146 MB; processed S1 SNAP tifs are AOI-cropped but the rule still applies).

## Failure contract

Any exception inside a renderer is caught by the dispatcher and converted to a
`QuicklookResult(kind="plain_image", image=<1px placeholder>, note=f"{type}: {msg}")`.
The app never crashes on a bad product; the reason is always shown.

## CRS contract (geospatial skill: CRS is law)

- Renderers return the NATIVE `src_crs`; reprojection to web-mercator is the map
  layer's job (leafmap), done per-product (F2: Landsat is mixed-zone).
- `bounds_4326` is computed by transforming the raster's native bounds with
  `pyproj`/`rasterio.warp.transform_bounds(always_xy=True)` — never assumed.

## AOI contract

`aoi.py` loads `data/bow_valley_inference_aoi.geojson` once → a GeoJSON layer added on top of every
`georef_raster` product so clip alignment is visually verifiable (the user's
explicit ask). AOI bounds also seed the initial map view.

## Non-georeferenced set (v1)

`{sentinel3}` → always `plain_image` (its geolocation is a separate per-pixel
`geo_coordinates.nc`, no GCPs in the radiance file). **S1 is NOT in this set** — it is
processed via SNAP into a map-projected EPSG:32611 GeoTIFF and renders as a
`georef_raster` (see below). Documented, deliberate. Not a gap to be silently filled.

## S1 render path (processed SNAP tif — supersedes the GCP-warp path)

> **SUPERSEDED (2026-06-11).** S1 is no longer clipped/GCP-warped. It is processed via
> ESA SNAP into a per-granule `sentinel1_snap/s1_grd_*.tif` (EPSG:32611, terrain-
> corrected). The viewer discovers S1 from that cache (not the clip manifest;
> `manifest._discover_s1_products`) and the renderer reads **band 2 (VV, linear σ⁰)**
> decimated, converts to dB, and reprojects to 4326 via the standard georeferenced-raster
> path (`_read_band_decimated` + `_to_4326`) — no GCPs. Returns `kind="georef_raster"`.
> See [`../raw-data-ingestion/PLAN-S1-PERGRANULE-SNAP.md`](../raw-data-ingestion/PLAN-S1-PERGRANULE-SNAP.md).
>
> *Original (historical) GCP-warp path:* S1 had `crs=None` + identity transform but real
> GCPs; the renderer warped the decimated VV band via `rasterio.warp.reproject(..., gcps=...)`. Removed along with `clip_sentinel1`.
