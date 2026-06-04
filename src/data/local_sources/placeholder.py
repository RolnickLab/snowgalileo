"""Placeholder adapters — the all-``-9999`` stand-ins that plumb the pipeline.

TASK-004 wires the whole pipeline end to end *before* any real source read
exists. Every modality is represented by a :class:`PlaceholderAdapter` that
satisfies the :class:`~src.data.local_sources.base.LocalSourceAdapter` port but
returns :func:`~src.data.local_sources.base.create_placeholder` (an all-``-9999``
``(len(bands_out), H, W)`` array) for **every** ``(cell, day)``. The exporter
treats these exactly as it will treat the real adapters in TASK-006…TASK-014, so
the band order, tensor shapes, mask paths, and filename are proven correct while
the FSC output is (correctly) degenerate.

**One adapter per modality (subtask 3).** The 38-band dynamic block partitions
into the same band groups the downstream loader slices
(``landsat_eval.py:281-313``): S1+S2+Landsat (high-res, 15), S3 (med-res, 2),
MODIS+VIIRS-fine (low-res, 9), VIIRS-coarse+ERA5 (time, 9), and the three cloud
flags (3). Each group is one placeholder adapter. The static stack
(DEM/slope/aspect/Map, 4) is a sixth, ``day``-independent adapter. The exporter
concatenates them in this exact order per timestep — the band-order contract
lives in :mod:`src.data.local_sources.layout`, never retyped here.

When a real adapter lands (e.g. TASK-009 MODIS), it replaces the matching
placeholder in :func:`dynamic_adapters` / :func:`static_adapter` and the exporter
is unchanged.
"""

from __future__ import annotations

import datetime

import numpy.typing as npt

from src.data.earthengine import eo_eval as _eo_eval
from src.data.local_sources.base import (
    GridCell,
    LocalSourceAdapter,
    SpatialKind,
    create_placeholder,
)
from src.data.local_sources.layout import DYNAMIC_BANDS, STATIC_BANDS

# Re-export the loader's own band-group lists so each placeholder declares the
# exact slice it owns. Importing (not retyping) keeps the partition byte-true to
# the downstream loader's slicing (SPEC AC-26).
_HIGH_RES_BANDS: list[str] = list(_eo_eval.SPACE_TIME_HIGH_RES_BANDS)  # 15
_MED_RES_BANDS: list[str] = list(_eo_eval.SPACE_TIME_MED_RES_BANDS)  # 2
_LOW_RES_BANDS: list[str] = list(_eo_eval.EO_SPACE_TIME_LOW_RES_BANDS)  # 9
_TIME_BANDS: list[str] = list(_eo_eval.TIME_BANDS)  # 9
_CLOUD_BANDS: list[str] = list(_eo_eval.CLOUD_BANDS)  # 3


class PlaceholderAdapter(LocalSourceAdapter):
    """A modality adapter that always returns the all-``-9999`` placeholder.

    Declares a contiguous slice of the canonical band order and returns
    :func:`create_placeholder` of the matching shape for any ``(cell, day)`` —
    the *normal* missing-acquisition return value, not an error (AC-4 / FR-13).

    Args:
        bands_out: The exact band names this adapter owns, in canonical order.
        spatial_kind: The band group's resolution tier (informational; the
            exporter places bands by position, the loader downsamples by group).
    """

    def __init__(self, *, bands_out: list[str], spatial_kind: SpatialKind) -> None:
        self.bands_out = bands_out
        self.spatial_kind = spatial_kind
        self.native_fill = None

    def fetch(
        self,
        cell: GridCell,
        day: datetime.date | None,
    ) -> npt.NDArray:
        """Return an all-``-9999`` ``(len(bands_out), *cell.shape)`` array.

        Args:
            cell: Target grid cell (supplies the output ``shape``).
            day: Ignored — the placeholder is missing on every day.

        Returns:
            The placeholder band stack on the cell grid.
        """
        return create_placeholder(n_bands=len(self.bands_out), shape=cell.shape)


def dynamic_adapters() -> list[PlaceholderAdapter]:
    """Return one placeholder per dynamic band group, in canonical order.

    The concatenation of the groups' ``bands_out`` is exactly
    :data:`~src.data.local_sources.layout.DYNAMIC_BANDS` (asserted below), so the
    exporter writing each adapter's output in this list order reproduces the
    per-timestep dynamic block ``create_ee_image`` emits.

    Returns:
        The five dynamic-modality placeholder adapters (high, med, low, time, cloud).
    """
    adapters = [
        PlaceholderAdapter(bands_out=_HIGH_RES_BANDS, spatial_kind="high"),
        PlaceholderAdapter(bands_out=_MED_RES_BANDS, spatial_kind="med"),
        PlaceholderAdapter(bands_out=_LOW_RES_BANDS, spatial_kind="low"),
        PlaceholderAdapter(bands_out=_TIME_BANDS, spatial_kind="time"),
        PlaceholderAdapter(bands_out=_CLOUD_BANDS, spatial_kind="time"),
    ]
    flattened = [band for adapter in adapters for band in adapter.bands_out]
    assert flattened == DYNAMIC_BANDS, (
        "Placeholder dynamic band groups do not reproduce DYNAMIC_BANDS — "
        "band-layout contract broken."
    )
    return adapters


def static_adapter() -> PlaceholderAdapter:
    """Return the single static (DEM/slope/aspect/Map) placeholder adapter.

    Its ``bands_out`` equals :data:`~src.data.local_sources.layout.STATIC_BANDS`
    and it ignores ``day`` (static layers are time-invariant).

    Returns:
        The static-stack placeholder adapter.
    """
    adapter = PlaceholderAdapter(bands_out=list(STATIC_BANDS), spatial_kind="static")
    assert adapter.bands_out == STATIC_BANDS, (
        "Placeholder static band group does not match STATIC_BANDS."
    )
    return adapter
