# CONTRACT: clip-viewer rendering interface

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
note:           str|None              # e.g. "S1 GRD: non-georeferenced (no CRS)"
```
- `georef_raster` → leafmap places it via `bounds_4326` (reprojected from `src_crs`).
- `plain_image`   → shown in a side panel, NOT on the map (S1, S3, error fallback).

## Renderer protocol
```python
class Renderer(Protocol):
    source: str
    def render(self, row: ProductRow, *, long_edge: int) -> QuicklookResult: ...
```
- Dispatch by `row.source` via a registry `RENDERERS: dict[str, Renderer]`.
- `long_edge` is the decimation target (ViewerSettings, default 1024). Renderers
  MUST use decimated/overview reads (`out_shape`) — never full-res (F3: S1 ~146 MB).

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
`aoi.py` loads `data/aoi.geojson` once → a GeoJSON layer added on top of every
`georef_raster` product so clip alignment is visually verifiable (the user's
explicit ask). AOI bounds also seed the initial map view.

## Non-georeferenced set (v1)
`{sentinel3}` → always `plain_image` (its geolocation is a separate per-pixel
`geo_coordinates.nc`, no GCPs in the radiance file). **S1 is NOT in this set** — it
carries EPSG:4326 GCPs and renders as a `georef_raster` via GCP warp (F3 corrected).
Documented, deliberate. Not a gap to be silently filled.

## GCP-warp path (S1)
S1 has `crs=None` + identity transform but real GCPs. The S1 renderer reads the
decimated VV band, then `rasterio.warp.reproject(..., gcps=src.gcps[0],
src_crs=src.gcps[1])` to web-mercator/4326 to obtain `image` + `bounds_4326`.
Returns `kind="georef_raster"`.
