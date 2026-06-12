"""Viewer configuration (pydantic-settings, no magic numbers)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.local_sources.paths import LocalPaths

_PATHS = LocalPaths()


class ViewerSettings(BaseSettings):
    """Runtime configuration for the clip viewer.

    Path defaults inherit from :class:`~src.data.local_sources.paths.LocalPaths`
    (so ``LOCAL_*`` region overrides flow through), and may be further overridden
    with the viewer-specific ``VIEWER_*`` environment prefix.
    """

    model_config = SettingsConfigDict(env_prefix="VIEWER_", extra="ignore")

    clipped_root: Path = Field(default_factory=lambda: _PATHS.clipped_root)
    aoi_path: Path = Field(default_factory=lambda: _PATHS.aoi_path)
    manifest_name: str = "clip_manifest.csv"

    # Stage-2 output roots for the cube + daily-FSC tabs (pipeline outputs, not the
    # clipped archive). Default from LocalPaths.processing_root; VIEWER_* overridable.
    processing_root: Path = Field(default_factory=lambda: _PATHS.processing_root)

    # Decimation target for quicklook reads (long edge, px). Guards against the
    # ~146 MB S1 full-res loads (geospatial skill: no eager multi-GB reads).
    long_edge: int = 1024

    default_basemap: str = "Esri.WorldImagery"

    @property
    def manifest_path(self) -> Path:
        return self.clipped_root / self.manifest_name

    @property
    def cubes_dir(self) -> Path:
        """Directory of assembled per-cell cubes (``PR_*.tif``)."""
        return self.processing_root / "cubes"

    @property
    def daily_fsc_dir(self) -> Path:
        """Directory of daily fractional-snow-cover COGs (``fsc_*.tif``)."""
        return self.processing_root / "daily_fsc"
