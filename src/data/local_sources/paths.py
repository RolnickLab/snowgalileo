"""Centralized data-root paths for the Bow Valley local-source pipeline.

Every stage (clip, audit, viewer, STAC, future cube assembly) resolves its data
roots from :class:`LocalPaths` rather than hardcoding them, so the pipeline can
be repointed at a different region or storage layout without editing code — set
the ``LOCAL_*`` environment variables (or a repo-root ``.env``) and/or repoint
the ``data/`` symlinks.

See ``data/BOW_VALLEY_DATA_LAYOUT.md`` for the full path contract and the symlink/portability
notes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LocalPaths(BaseSettings):
    """Resolvable data roots for the local-source pipeline.

    All paths default to the repo's standard ``data/`` layout (relative to the
    process CWD, i.e. the repo root). Override any of them via the ``LOCAL_``
    environment prefix, e.g. ``LOCAL_RAW_ROOT=/mnt/region2/raw``.

    Attributes:
        raw_root: Untouched per-modality download archive (read-only input).
        clipped_root: AOI-clipped archive; the single root every downstream
            adapter reads.
        processing_root: Stage-2 cube-assembly tree (intermediate cache +
            assembled cubes under ``cubes/`` + daily FSC COGs). The cubes are the
            durable end-product; there is no separate cube-archive root.
        aoi_path: Authoritative AOI polygon (EPSG:4326, single-Polygon GeoJSON).
    """

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_", env_file=".env", extra="ignore", frozen=True
    )

    raw_root: Path = Path("data/bow_valley_selection_raw")
    clipped_root: Path = Path("data/clipped_bow_valley_selection_raw")
    processing_root: Path = Path("data/bow_valley_processing")
    aoi_path: Path = Path("data/bow_valley_inference_aoi.geojson")
