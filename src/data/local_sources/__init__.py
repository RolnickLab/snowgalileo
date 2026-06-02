"""Direct-source local ingestion package for the Bow Valley pipeline.

See ``docs/agents/planning/raw-data-ingestion/PLAN_BOW_VALLEY_DATA.md`` for the
architecture. This package replaces Google Earth Engine ingestion with adapters
that read a locally clipped archive and produce GeoTIFFs byte-compatible with
``create_ee_image``.

Phase 0 ships only the geometry half of :mod:`src.data.local_sources.grid`.
"""
