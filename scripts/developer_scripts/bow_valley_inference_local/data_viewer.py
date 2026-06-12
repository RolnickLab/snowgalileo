"""Solara entrypoint for the Bow Valley data viewer.

Run with::

    uv run solara run scripts/developer_scripts/bow_valley_inference_local/data_viewer.py

A developer/QA tool with three tabs, each a leafmap map with the
``data/bow_valley_inference_aoi.geojson`` outline overlaid:

* **Clip** — pick a clipped product from the manifest; see its quicklook on a basemap
  with the clip-stage metadata (overlap km², valid-pixel count, action).
* **Cube** — inspect an assembled per-cell cube (Stage-2 output) layer by layer and date
  by date: pick prediction date → cell → variable, then step the timestep slider.
* **Daily FSC** — step a date slider through the daily fractional-snow-cover COGs; the
  selected day renders colormapped (0–1) on the map.

Reads the clipped archive and ``processing_root`` outputs read-only; writes only transient
decimated GeoTIFFs to a temp dir.

See ``docs/agents/planning/clip-viewer/PLAN.md``, ``CONTRACT.md`` and
``PLAN-V2-CUBE-FSC-TABS.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import solara
import solara.lab

import src.data.local_sources.viewer.renderers  # noqa: F401  -- registers all renderers on import
from src.data.local_sources.viewer.aoi import aoi_bounds_4326, load_aoi_geojson
from src.data.local_sources.viewer.manifest import ProductRow, load_products
from src.data.local_sources.viewer.outputs import (
    CubeAvailability,
    CubeRow,
    cube_availability,
    cubes_for_date,
    dates_for_cubes,
    list_cubes,
    list_fsc,
    timesteps_for_var,
    vars_at_timestep,
)
from src.data.local_sources.viewer.quicklook import QuicklookResult, render_product
from src.data.local_sources.viewer.renderers import (
    era5_time_steps,
    fsc_colorbar,
    render_cube_band,
    render_fsc,
    result_to_geotiff,
)
from src.data.local_sources.viewer.settings import ViewerSettings

import leafmap  # isort: skip  (heavy import, kept after local modules)

_SETTINGS = ViewerSettings()
_TMPDIR = Path(tempfile.mkdtemp(prefix="data_viewer_"))

# Loaded once at import (cheap: a few hundred manifest rows + a directory scan).
_PRODUCTS: list[ProductRow] = load_products(_SETTINGS)
_CUBES: list[CubeRow] = list_cubes(_SETTINGS)
_FSC = list_fsc(_SETTINGS)
_AOI_GEOJSON = load_aoi_geojson(_SETTINGS)
_AOI_BOUNDS = aoi_bounds_4326(_AOI_GEOJSON)

# Sources with a registered clip renderer (Phases 2-4 → all ten). Kept explicit so a
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

# Cube availability is a per-cube full-band read (~0.5s on the ~100×100 cubes). Memoise it
# by path so re-renders (a slider/dropdown nudge) don't re-scan the same cube each time.
_CUBE_AVAIL_CACHE: dict[Path, CubeAvailability] = {}


def _availability_for(path: Path) -> CubeAvailability:
    cached = _CUBE_AVAIL_CACHE.get(path)
    if cached is None:
        cached = cube_availability(path)
        _CUBE_AVAIL_CACHE[path] = cached
    return cached


# Cube selector modes: which axis is picked first and filters the other.
_CUBE_MODE_TIMESTEP = "Select by timestep"
_CUBE_MODE_VARIABLE = "Select by variable"
_CUBE_MODES = [_CUBE_MODE_TIMESTEP, _CUBE_MODE_VARIABLE]


def _products_for(source: str) -> list[ProductRow]:
    return [p for p in _PRODUCTS if p.source == source]


def _add_aoi_overlay(m: leafmap.Map) -> None:
    m.add_geojson(
        _AOI_GEOJSON,
        layer_name="AOI",
        style={"color": "#ff1744", "weight": 2, "fillOpacity": 0.0},
    )


def _opaque_data_bounds_4326(
    image: "object", bounds_4326: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """Bbox (in 4326) of the non-transparent pixels of a decimated RGB/gray image.

    Sparse fields (e.g. FSC covers ~2 % of its raster extent) frame poorly when zoomed to
    the *full* layer bounds — the data is a speck in the AOI bbox. This tightens the zoom
    target to just the pixels that actually carry colour (any colour band > 0), so the map
    frames the data, not the empty extent. Returns ``None`` if every pixel is empty.
    """
    import numpy as np

    arr = np.asarray(image)
    opaque = arr > 0 if arr.ndim == 2 else np.any(arr > 0, axis=2)
    rows = np.any(opaque, axis=1)
    cols = np.any(opaque, axis=0)
    if not rows.any() or not cols.any():
        return None
    h, w = opaque.shape
    west, south, east, north = bounds_4326
    r0, r1 = int(np.argmax(rows)), h - int(np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), w - int(np.argmax(cols[::-1]))
    # Row 0 is the north edge (top); map row index → latitude accordingly.
    lon0 = west + (east - west) * c0 / w
    lon1 = west + (east - west) * c1 / w
    lat1 = north - (north - south) * r0 / h
    lat0 = north - (north - south) * r1 / h
    return (lon0, lat0, lon1, lat1)


def _render_on_map(
    result: QuicklookResult | None,
    *,
    key: str,
    zoom_to_data: bool = False,
    colorbar: tuple[list[str], float, float, str] | None = None,
) -> leafmap.Map:
    """Build a leafmap map centred on the AOI with ``result`` placed on it (if georef).

    Shared by all three tabs: the AOI outline is always drawn; a ``georef_raster``
    result is written to a transient GeoTIFF and added as a raster layer. ``key`` makes
    the transient filename unique per selection so layers don't collide.

    Args:
        result: The quicklook to place (or ``None`` for AOI-only).
        key: Unique transient-filename seed for this selection.
        zoom_to_data: If ``True``, frame the map on the non-transparent data footprint
            rather than the full layer extent — for sparse fields (FSC) that would
            otherwise be a speck inside the AOI bbox.
        colorbar: Optional ``(hex_colours, vmin, vmax, caption)`` for a continuous on-map
            colour scale (e.g. the FSC 0–1 legend). Drawn bottom-right.
    """
    center = [
        (_AOI_BOUNDS[1] + _AOI_BOUNDS[3]) / 2,
        (_AOI_BOUNDS[0] + _AOI_BOUNDS[2]) / 2,
    ]
    zoom = 8

    has_raster = (
        result is not None
        and result.kind == "georef_raster"
        and result.bounds_4326 is not None
    )
    tif: Path | None = None
    if has_raster:
        assert result is not None and result.bounds_4326 is not None  # narrows for mypy
        safe = "".join(c if c.isalnum() else "_" for c in key)
        tif = result_to_geotiff(result, _TMPDIR / f"{safe}.tif")
        if zoom_to_data:
            data_box = _opaque_data_bounds_4326(result.image, result.bounds_4326)
            if data_box is not None:
                center = [(data_box[1] + data_box[3]) / 2, (data_box[0] + data_box[2]) / 2]
                # Pick a zoom from the data-box span so a wide-but-sparse field (FSC
                # scatters across ~the whole AOI) isn't over-zoomed and clipped, while a
                # genuinely small footprint still fills the view. ~1.4° span → zoom 8.
                span_deg = max(data_box[2] - data_box[0], data_box[3] - data_box[1])
                zoom = 8 if span_deg > 0.7 else 9 if span_deg > 0.3 else 11

    m = leafmap.Map(center=center, zoom=zoom)
    m.add_basemap(_SETTINGS.default_basemap)
    if has_raster and tif is not None:
        assert result is not None  # narrows for mypy (guarded by has_raster)
        # Pass the colour-band indexes explicitly. result_to_geotiff appends an alpha
        # band to uint8 outputs (RGB→RGBA, gray→gray+alpha); the tile server applies that
        # alpha via colorinterp on its own, so indexes must point only at the colour
        # bands, never the alpha. Relying on a None default makes leafmap assume 3 RGB
        # bands and index past a 2-band (gray+alpha) raster → IndexError.
        indexes = [1, 2, 3] if result.image.ndim == 3 else [1]
        m.add_raster(
            str(tif),
            indexes=indexes,
            layer_name=result.label,
            opacity=1.0,
            # When zooming to the data footprint we have already framed the view; letting
            # zoom_to_layer re-frame would snap back to the full (mostly-empty) extent.
            zoom_to_layer=not zoom_to_data,
        )
        if colorbar is not None:
            colors, vmin, vmax, caption = colorbar
            m.add_colorbar(
                colors=colors,
                vmin=vmin,
                vmax=vmax,
                caption=caption,
                position="bottomright",
            )
    _add_aoi_overlay(m)
    # NOTE: deliberately no ``m.fit_bounds(...)`` here. fit_bounds triggers a
    # viewport bounds round-trip on a map that may not be attached to the front-end
    # yet; the returned east/west traits come back ``None`` and the Map trait
    # validator raises ``TraitError: 'east' ... expected a float, not NoneType`` on
    # re-render. The constructor ``center``/``zoom`` already frames the AOI, and
    # ``zoom_to_layer`` frames the active layer, so fit_bounds was redundant anyway.
    return m


# --------------------------------------------------------------------------- #
# Tab: Clip (the original manifest viewer)
# --------------------------------------------------------------------------- #


@solara.component
def _MetadataPanel(row: ProductRow, result: QuicklookResult | None) -> None:
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
def _PlainImagePanel(result: QuicklookResult) -> None:
    """Render a non-georeferenced quicklook (S3, error fallback) in a side panel."""
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
def ClipTab() -> None:
    """The original clip-manifest viewer (source → product → quicklook on the map)."""
    source, set_source = solara.use_state(_SOURCES[0] if _SOURCES else "")
    products = _products_for(source)
    ids = [p.product_id for p in products]
    product_id, set_product_id = solara.use_state(ids[0] if ids else "")
    date_idx, set_date_idx = solara.use_state(0)

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

    result: QuicklookResult | None = None
    with solara.Columns([4, 8]):
        with solara.Column():
            solara.Select(label="Source", value=source, values=_SOURCES, on_value=set_source)
            if source not in _RENDERABLE_SOURCES:
                solara.Warning(
                    f"`{source}` has no renderer yet (later phase). "
                    "Metadata only; the map shows the AOI."
                )
            solara.Select(
                label="Product", value=product_id, values=ids, on_value=set_product_id
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
                    render_product(
                        row, long_edge=_SETTINGS.long_edge, date_idx=safe_date_idx
                    )
                    if source in _RENDERABLE_SOURCES
                    else None
                )
                _MetadataPanel(row, result)
                if result is not None and result.kind == "plain_image":
                    _PlainImagePanel(result)
        with solara.Column():
            solara.display(
                _render_on_map(result, key=f"clip_{source}_{product_id}_{safe_date_idx}")
            )


# --------------------------------------------------------------------------- #
# Tab: Cube (assembled per-cell cubes — layer by layer, date by date)
# --------------------------------------------------------------------------- #


def _ts_label(timestep: int) -> str:
    """Dropdown label for a timestep index."""
    return f"t{timestep}"


@solara.component
def CubeTab() -> None:
    """Inspect a cube band, date → cell → (timestep ⇄ variable) by the chosen mode.

    Two selection modes drive which axis is picked first and filters the other, where
    "available" means the band carries real (non-nodata) data — most dynamic bands are
    all-nodata at most timesteps (e.g. Landsat is real at only one or two timesteps):

    * **Select by timestep** — pick a timestep first; the variable dropdown then lists only
      the variables that are real at that timestep (plus the statics, which ignore the
      timestep axis).
    * **Select by variable** — pick a variable first; the timestep dropdown then lists only
      the timesteps at which that variable is real. Statics have no timestep.
    """
    if not _CUBES:
        solara.Info(
            "No cubes found under "
            f"`{_SETTINGS.cubes_dir}`. Run `export_bow_valley_cube.py` first."
        )
        return

    dates = dates_for_cubes(_CUBES)
    date_labels = [d.isoformat() for d in dates]
    date_label, set_date_label = solara.use_state(date_labels[0])
    if date_label not in date_labels:
        date_label = date_labels[0]
    sel_date = dates[date_labels.index(date_label)]

    cells = cubes_for_date(_CUBES, sel_date)
    cell_labels = [c.cell_label for c in cells]
    cell_label, set_cell_label = solara.use_state(cell_labels[0])
    if cell_label not in cell_labels:
        cell_label = cell_labels[0]
    cell = cells[cell_labels.index(cell_label)]

    avail = _availability_for(cell.path)
    all_vars = [*avail.dynamic_order, *avail.statics]
    all_ts = list(range(avail.n_timesteps))

    mode, set_mode = solara.use_state(_CUBE_MODE_TIMESTEP)
    # Raw selections persist across re-renders; each axis is clamped below to whatever the
    # *other* axis (and the active cube) makes valid, so a stale pick falls back cleanly
    # when the date/cell/mode changes rather than rendering an all-nodata band.
    var, set_var = solara.use_state(all_vars[0])
    timestep, set_timestep = solara.use_state(0)

    if mode == _CUBE_MODE_TIMESTEP:
        # Timestep is the free axis; the variable list is filtered to it.
        sel_ts = timestep if timestep in all_ts else (all_ts[0] if all_ts else 0)
        var_choices = vars_at_timestep(avail, sel_ts)
        sel_var = var if var in var_choices else var_choices[0]
        ts_choices = all_ts
    else:
        # Variable is the free axis; the timestep list is filtered to it.
        sel_var = var if var in all_vars else all_vars[0]
        ts_choices = timesteps_for_var(avail, sel_var)
        sel_ts = timestep if timestep in ts_choices else (ts_choices[0] if ts_choices else 0)
        var_choices = all_vars

    is_static = sel_var in avail.statics

    result = render_cube_band(
        path=cell.path,
        var=sel_var,
        timestep=sel_ts,
        is_static=is_static,
        long_edge=_SETTINGS.long_edge,
    )

    def _timestep_select() -> None:
        # The timestep dropdown over ``ts_choices``. In timestep mode this is the free axis
        # (always the full t0..tN), so it must always render. In variable mode it is the
        # dependent axis: a static has no timestep, and an all-nodata var has none either.
        if mode == _CUBE_MODE_VARIABLE and is_static:
            solara.Info("Static band — no timestep.")
            return
        if not ts_choices:
            solara.Warning(f"`{sel_var}` is all-nodata in this cube — nothing to show.")
            return
        solara.Select(
            label="Timestep",
            value=_ts_label(sel_ts),
            values=[_ts_label(t) for t in ts_choices],
            on_value=lambda label: set_timestep(int(label[1:])),
        )

    def _variable_select() -> None:
        solara.Select(
            label="Variable", value=sel_var, values=var_choices, on_value=set_var
        )

    with solara.Columns([4, 8]):
        with solara.Column():
            solara.Select(
                label="Prediction date",
                value=date_label,
                values=date_labels,
                on_value=set_date_label,
            )
            solara.Select(
                label="Cell (lat, lon)",
                value=cell_label,
                values=cell_labels,
                on_value=set_cell_label,
            )
            solara.ToggleButtonsSingle(value=mode, values=_CUBE_MODES, on_value=set_mode)
            # Selector order follows the mode: the free axis is picked first and filters
            # the second.
            if mode == _CUBE_MODE_TIMESTEP:
                _timestep_select()
                _variable_select()
            else:
                _variable_select()
                _timestep_select()
            with solara.Card("Cube band"):
                solara.Markdown(
                    f"**file:** `{cell.path.name}`  \n"
                    f"**band:** `{result.label}`  \n"
                    f"**CRS:** `{result.src_crs}`"
                )
        with solara.Column():
            solara.display(
                _render_on_map(result, key=f"cube_{cell.path.stem}_{sel_var}_{sel_ts}")
            )


# --------------------------------------------------------------------------- #
# Tab: Daily FSC (date slider over the daily-FSC COGs)
# --------------------------------------------------------------------------- #


@solara.component
def FscTab() -> None:
    """Step a date slider through the daily-FSC COGs; render colormapped (0–1)."""
    if not _FSC:
        solara.Info(
            "No daily-FSC COGs found under "
            f"`{_SETTINGS.daily_fsc_dir}`. Run `infer_bow_valley_daily_fsc.py` first."
        )
        return

    idx, set_idx = solara.use_state(0)
    safe_idx = min(idx, len(_FSC) - 1)
    fsc_row = _FSC[safe_idx]

    result = render_fsc(path=fsc_row.path, long_edge=_SETTINGS.long_edge)

    with solara.Columns([4, 8]):
        with solara.Column():
            if len(_FSC) > 1:
                solara.SliderInt(
                    label=f"Date: {fsc_row.pred_date.isoformat()} ({safe_idx + 1}/{len(_FSC)})",
                    value=safe_idx,
                    min=0,
                    max=len(_FSC) - 1,
                    on_value=set_idx,
                )
            else:
                solara.Info(f"Only one date on disk: {fsc_row.pred_date.isoformat()}.")
            with solara.Card("Daily FSC"):
                solara.Markdown(
                    f"**date:** `{fsc_row.pred_date.isoformat()}`  \n"
                    f"**file:** `{fsc_row.path.name}`  \n"
                    f"**scale:** fixed 0–1 (turbo)  \n"
                    f"**CRS:** `{result.src_crs}`"
                )
        with solara.Column():
            fsc_colors, fsc_vmin, fsc_vmax = fsc_colorbar()
            solara.display(
                _render_on_map(
                    result,
                    key=f"fsc_{fsc_row.pred_date.isoformat()}",
                    zoom_to_data=True,
                    colorbar=(fsc_colors, fsc_vmin, fsc_vmax, "FSC (0–1)"),
                )
            )


# --------------------------------------------------------------------------- #
# Page: three tabs
# --------------------------------------------------------------------------- #


@solara.component
def Page() -> None:
    solara.Title("Bow Valley data viewer")
    with solara.lab.Tabs():
        with solara.lab.Tab("Clip"):
            ClipTab()
        with solara.lab.Tab("Cube"):
            CubeTab()
        with solara.lab.Tab("Daily FSC"):
            FscTab()
