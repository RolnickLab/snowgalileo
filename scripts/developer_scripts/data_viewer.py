"""Solara entrypoint for the clipped-archive visual validation viewer.

Run with::

    uv run solara run scripts/developer_scripts/data_viewer.py

A developer/QA tool: pick a clipped product from the manifest, see its quicklook
placed on a basemap with the ``data/aoi.geojson`` outline overlaid, and read the
clip-stage metadata (overlap km², valid-pixel count, action). Reads the clipped
archive read-only; writes only transient decimated GeoTIFFs to a temp dir.

See ``docs/agents/planning/clip-viewer/PLAN.md`` and ``CONTRACT.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import solara

import src.data.local_sources.viewer.renderers  # noqa: F401  -- registers all renderers on import
from src.data.local_sources.viewer.aoi import aoi_bounds_4326, load_aoi_geojson
from src.data.local_sources.viewer.manifest import ProductRow, load_products
from src.data.local_sources.viewer.quicklook import QuicklookResult, render_product
from src.data.local_sources.viewer.renderers import era5_time_steps, result_to_geotiff
from src.data.local_sources.viewer.settings import ViewerSettings

import leafmap  # isort: skip  (heavy import, kept after local modules)

_SETTINGS = ViewerSettings()
_TMPDIR = Path(tempfile.mkdtemp(prefix="clip_viewer_"))

# Loaded once at import (cheap: a few hundred manifest rows).
_PRODUCTS: list[ProductRow] = load_products(_SETTINGS)
_AOI_GEOJSON = load_aoi_geojson(_SETTINGS)
_AOI_BOUNDS = aoi_bounds_4326(_AOI_GEOJSON)

# Sources with a registered renderer (Phases 2-4 → all ten). Kept explicit so a
# future source without a renderer is flagged rather than silently blank.
_RENDERABLE_SOURCES = {
    "dem",
    "worldcover",
    "modis",
    "viirs",
    "landsat8",
    "landsat9",
    "sentinel2",
    "sentinel1",
    "era5",
    "sentinel3",
}

_SOURCES: list[str] = sorted({p.source for p in _PRODUCTS})


def _products_for(source: str) -> list[ProductRow]:
    return [p for p in _PRODUCTS if p.source == source]


def _add_aoi_overlay(m: leafmap.Map) -> None:
    m.add_geojson(
        _AOI_GEOJSON,
        layer_name="AOI",
        style={"color": "#ff1744", "weight": 2, "fillOpacity": 0.0},
    )


@solara.component
def MetadataPanel(row: ProductRow, result: QuicklookResult | None) -> None:
    """Show clip-stage manifest metadata + render note for the selected product."""
    with solara.Card("Product metadata"):
        solara.Markdown(
            f"**id:** `{row.product_id}`  \n"
            f"**source:** `{row.source}`  \n"
            f"**action:** `{row.action}`  \n"
            f"**intersects:** `{row.intersects}`  \n"
            f"**AOI overlap:** {row.aoi_overlap_km2:.3f} km²  \n"
            f"**valid pixels:** {row.valid_pixel_count:,}  \n"
            f"**footprint bbox (4326):** `{row.footprint_bbox}`  \n"
            f"**path:** `{row.path}`"
        )
        if result is not None and result.note is not None:
            solara.Warning(f"{result.label}: {result.note}")
        elif result is not None:
            solara.Info(result.label)


@solara.component
def PlainImagePanel(result: QuicklookResult) -> None:
    """Render a non-georeferenced quicklook (S3, error fallback) in a side panel.

    ``plain_image`` results have no map placement; show the array as a PNG with a
    grayscale colormap so the radiance shape is still inspectable.
    """
    import io

    import matplotlib.cm as cm
    import PIL.Image

    with solara.Card(result.label):
        image = result.image
        if image.size <= 1:
            solara.Warning(result.note or "no image")
            return
        gray = image if image.ndim == 2 else image[..., 0]
        rgba = (cm.get_cmap("gray")(gray / 255.0) * 255).astype("uint8")
        buf = io.BytesIO()
        PIL.Image.fromarray(rgba).save(buf, format="PNG")
        solara.Image(buf.getvalue())


@solara.component
def Page() -> None:
    source, set_source = solara.use_state(_SOURCES[0] if _SOURCES else "")
    products = _products_for(source)
    ids = [p.product_id for p in products]
    product_id, set_product_id = solara.use_state(ids[0] if ids else "")
    date_idx, set_date_idx = solara.use_state(0)

    # Keep product selection valid when the source changes.
    if product_id not in ids:
        product_id = ids[0] if ids else ""

    row = next(
        (p for p in products if p.product_id == product_id),
        products[0] if products else None,
    )

    # ERA5 is time-stepped: expose a date slider over its valid_time axis.
    era5_steps: list[str] = []
    if source == "era5" and row is not None and row.path is not None:
        try:
            era5_steps = era5_time_steps(row.path)
        except Exception:  # noqa: BLE001 — slider is best-effort
            era5_steps = []
    safe_date_idx = min(date_idx, len(era5_steps) - 1) if era5_steps else 0

    solara.Title("Clip viewer")
    with solara.Sidebar():
        solara.Select(
            label="Source",
            value=source,
            values=_SOURCES,
            on_value=set_source,
        )
        if source not in _RENDERABLE_SOURCES:
            solara.Warning(
                f"`{source}` has no renderer yet (later phase). "
                "Metadata only; the map shows the AOI."
            )
        solara.Select(
            label="Product",
            value=product_id,
            values=ids,
            on_value=set_product_id,
        )
        if era5_steps:
            solara.SliderInt(
                label=f"Date: {era5_steps[safe_date_idx]}",
                value=safe_date_idx,
                min=0,
                max=len(era5_steps) - 1,
                on_value=set_date_idx,
            )
        if row is not None:
            result = (
                render_product(row, long_edge=_SETTINGS.long_edge, date_idx=safe_date_idx)
                if source in _RENDERABLE_SOURCES
                else None
            )
            MetadataPanel(row, result)
            if result is not None and result.kind == "plain_image":
                PlainImagePanel(result)
        else:
            result = None

    def make_map() -> leafmap.Map:
        m = leafmap.Map(
            center=[
                (_AOI_BOUNDS[1] + _AOI_BOUNDS[3]) / 2,
                (_AOI_BOUNDS[0] + _AOI_BOUNDS[2]) / 2,
            ],
            zoom=8,
        )
        m.add_basemap(_SETTINGS.default_basemap)
        if (
            result is not None
            and result.kind == "georef_raster"
            and result.bounds_4326 is not None
        ):
            safe = "".join(c if c.isalnum() else "_" for c in product_id)
            tif = result_to_geotiff(result, _TMPDIR / f"{source}_{safe}.tif")
            # Pass the colour-band indexes explicitly. result_to_geotiff appends an
            # alpha band to uint8 outputs (RGB→RGBA, gray→gray+alpha); the tile
            # server applies that alpha via colorinterp on its own, so indexes must
            # point only at the colour bands, never the alpha. Relying on a None
            # default makes leafmap assume 3 RGB bands and index past a 2-band
            # (gray+alpha) raster → IndexError.
            indexes = [1, 2, 3] if result.image.ndim == 3 else [1]
            m.add_raster(
                str(tif),
                indexes=indexes,
                layer_name=result.label,
                zoom_to_layer=True,
            )
        _add_aoi_overlay(m)
        m.fit_bounds([[_AOI_BOUNDS[1], _AOI_BOUNDS[0]], [_AOI_BOUNDS[3], _AOI_BOUNDS[2]]])
        return m

    # Re-create the map when source/product changes (key forces remount).
    leafmap_widget = make_map()
    solara.display(leafmap_widget)


# Solara discovers `Page` as the app component.
