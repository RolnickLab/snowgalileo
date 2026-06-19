"""Rendering contract: ``QuicklookResult``, the ``Renderer`` protocol, and the
dispatcher. Renderers themselves live in ``renderers.py`` (Phase 2+).

See ``docs/agents/planning/bow_valley/060-viewer/062-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import structlog

from src.data.local_sources.viewer.manifest import ProductRow

logger = structlog.get_logger(__name__)

Kind = Literal["georef_raster", "plain_image"]

# Modalities with no usable georeferencing on disk, always rendered as plain
# (non-map) images. Only S3 OLCI qualifies: its geolocation lives in a separate
# per-pixel ``geo_coordinates.nc`` with no GCPs in the radiance file. S1 is NOT
# here — it carries EPSG:4326 GCPs and is GCP-warped onto the map (Phase-0 F3
# corrected).
NON_GEOREF_SOURCES: frozenset[str] = frozenset({"sentinel3"})


@dataclass(frozen=True)
class QuicklookResult:
    """A renderable quicklook for one product.

    Attributes:
        kind: ``georef_raster`` (placeable on the map via ``bounds_4326``) or
            ``plain_image`` (shown in a side panel, never on the map).
        image: ``HxW`` (single-band) or ``HxWx3`` (RGB) array, decimated.
        bounds_4326: ``(minx, miny, maxx, maxy)`` for map placement, else ``None``.
        src_crs: Native CRS string of the rendered raster, else ``None``.
        label: Human-readable description (e.g. band combination).
        note: Optional caveat (non-georeferenced, error fallback, ...).
        alpha_mask: Optional ``HxW`` boolean validity mask (``True`` = real data,
            ``False`` = nodata → transparent). When set, ``result_to_geotiff`` builds
            the alpha band from this *explicit* mask instead of inferring transparency
            from stretched value (the all-zero-RGB heuristic). Required for sources
            whose valid data can legitimately stretch to 0 — e.g. cube bands, where a
            dark-but-valid or uniform field would otherwise be falsely dropped as
            nodata. Must match ``image`` height/width.
    """

    kind: Kind
    image: npt.NDArray[np.floating | np.integer]
    bounds_4326: tuple[float, float, float, float] | None
    src_crs: str | None
    label: str
    note: str | None = None
    alpha_mask: npt.NDArray[np.bool_] | None = None


@runtime_checkable
class Renderer(Protocol):
    """Per-modality quicklook renderer."""

    source: str

    def render(self, row: ProductRow, *, long_edge: int, date_idx: int = 0) -> QuicklookResult:
        """Render ``row`` to a ``QuicklookResult`` using decimated reads.

        ``date_idx`` selects a time slice for time-stepped sources (ERA5); other
        renderers ignore it.
        """
        ...


# Populated by renderers.py via ``register()`` (Phase 2+).
RENDERERS: dict[str, Renderer] = {}


def register(renderer: Renderer) -> Renderer:
    """Register a renderer under its ``source`` key (decorator-friendly)."""
    RENDERERS[renderer.source] = renderer
    return renderer


def _error_result(label: str, exc: Exception) -> QuicklookResult:
    return QuicklookResult(
        kind="plain_image",
        image=np.zeros((1, 1), dtype=np.uint8),
        bounds_4326=None,
        src_crs=None,
        label=label,
        note=f"{type(exc).__name__}: {exc}",
    )


def render_product(
    row: ProductRow, *, long_edge: int = 1024, date_idx: int = 0
) -> QuicklookResult:
    """Dispatch a product to its renderer, enforcing the failure contract.

    Any renderer exception (or missing renderer / unresolved path) is converted to
    a ``plain_image`` placeholder carrying the reason — the app never crashes on a
    bad product.

    Args:
        row: The product to render.
        long_edge: Decimation target passed to the renderer.
        date_idx: Time-slice index for time-stepped sources (ERA5); ignored elsewhere.

    Returns:
        A ``QuicklookResult``; ``plain_image`` with a ``note`` on any failure.
    """
    if row.path is None:
        return _error_result(
            label=f"{row.source}: {row.product_id}",
            exc=ValueError(f"no output ({row.action})"),
        )
    renderer = RENDERERS.get(row.source)
    if renderer is None:
        return _error_result(
            label=f"{row.source}: {row.product_id}",
            exc=NotImplementedError(f"no renderer for source {row.source!r}"),
        )
    try:
        return renderer.render(row, long_edge=long_edge, date_idx=date_idx)
    except Exception as exc:  # failure contract: never propagate
        logger.warning("render_failed", source=row.source, product=row.product_id)
        return _error_result(label=f"{row.source}: {row.product_id}", exc=exc)
