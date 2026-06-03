"""Viewer configuration (pydantic-settings, no magic numbers)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ViewerSettings(BaseSettings):
    """Runtime configuration for the clip viewer.

    All paths default to the repo's standard layout; override via ``VIEWER_*``
    environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="VIEWER_", extra="ignore")

    clipped_root: Path = Path("data/clipped_bow_valley_selection_raw")
    aoi_path: Path = Path("data/aoi.geojson")
    manifest_name: str = "clip_manifest.csv"

    # Decimation target for quicklook reads (long edge, px). Guards against the
    # ~146 MB S1 full-res loads (geospatial skill: no eager multi-GB reads).
    long_edge: int = 1024

    default_basemap: str = "Esri.WorldImagery"

    @property
    def manifest_path(self) -> Path:
        return self.clipped_root / self.manifest_name
