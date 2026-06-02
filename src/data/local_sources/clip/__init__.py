"""AOI clip stage (Phase 0.5) for the Bow Valley direct-source pipeline.

This package crops every raw dataset in ``data/bow_valley_selection_raw`` to the
authoritative AOI polygon (``data/aoi.geojson``), non-destructively, into
``data/clipped_bow_valley_selection_raw`` — the single archive root every
``LocalSource*`` adapter reads.

Modules
-------
settings
    Pydantic-settings config (``MIN_AOI_OVERLAP_AREA_KM2`` etc.) and the AOI
    loader. No magic numbers in the clip routines.
gate
    The two-stage intersect gate (§2.0 of ``CLIPPING_PLAN.md``): footprint-vs-AOI
    polygon intersection, then minimum-useful-overlap. The one place
    footprint filtering happens — adapters never re-implement it.
footprints
    Metadata-only footprint readers per modality (Landsat ``MTL.json``, Sentinel
    ``manifest.safe`` GML, MODIS/VIIRS sinusoidal tile bounds, NetCDF coord
    ranges) — decode no pixels.
manifest
    The per-source clip manifest: one row per input product with its action.
"""
