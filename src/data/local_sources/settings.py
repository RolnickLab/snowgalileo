"""Configuration for the Stage-2 cube-assembly pipeline (``cube.yaml``).

``CubeSettings`` is the single typed entry point for the exporter/driver: it
carries the archive read-root, the processing write-root, the sweep mode, the
inference window, the per-cell target CRS, and the cache cap. Stage-2 subdirs
(``cube_cache/``, ``cubes/``, ``daily_fsc/``, ``manifests/``, ``scratch/``) are
**derived** from ``processing_root`` so no process can write outside its own
subdir (PLAN §3 Directory layout, SPEC FR-20b).

Precedence (highest first): ``CUBE_``-prefixed environment variables → values in
``cube.yaml`` → field defaults (which themselves resolve from
:class:`~src.data.local_sources.paths.LocalPaths`, so the cube config and the
clip stage share one path source of truth).

**Path contract (FR-6).** ``archive_root`` is the **clipped** archive
(``data/clipped_bow_valley_selection_raw``); the raw archive path appears only in
the clip stage's config, never here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from src.data.local_sources.paths import LocalPaths

_PATHS = LocalPaths()

SweepMode = Literal["A", "B"]

#: Default inference window (PLAN §3 Temporal window), inclusive of both ends.
DEFAULT_WINDOW_START = date(2025, 4, 6)
DEFAULT_WINDOW_END = date(2025, 5, 28)


class CubeSettings(BaseSettings):
    """Typed Stage-2 cube-assembly configuration.

    Attributes:
        archive_root: The **clipped** archive every adapter reads (FR-6).
        processing_root: Stage-2 write-root; all subdirs derive from it.
        mode: Sweep mode — ``"A"`` (in-AOI sample cells) or ``"B"`` (tile the AOI).
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).
        cell_crs: Per-cell target CRS — UTM 11N (matches the GEE reference
            patches; see ``docs/agents/KNOWLEDGE.md``).
        cache_max_entries: FIFO cap on the per-(modality, cell, day) ``.npz`` cache.
    """

    model_config = SettingsConfigDict(
        env_prefix="CUBE_", extra="ignore", frozen=True
    )

    archive_root: Path = _PATHS.clipped_root
    processing_root: Path = _PATHS.processing_root
    mode: SweepMode = "A"
    window_start: date = DEFAULT_WINDOW_START
    window_end: date = DEFAULT_WINDOW_END
    cell_crs: str = "EPSG:32611"
    cache_max_entries: Annotated[int, Field(gt=0)] = 200_000

    # --- derived Stage-2 subdirs (each process writes only its own) --------- #

    @property
    def cube_cache_dir(self) -> Path:
        """Intermediate per-(modality, cell, day) ``.npz`` cache (cleanable)."""
        return self.processing_root / "cube_cache"

    @property
    def cubes_dir(self) -> Path:
        """Assembled 8-day multiband cube tifs (kept deliverable)."""
        return self.processing_root / "cubes"

    @property
    def daily_fsc_dir(self) -> Path:
        """Daily FSC COGs over the AOI (kept inference deliverable)."""
        return self.processing_root / "daily_fsc"

    @property
    def manifests_dir(self) -> Path:
        """Clip-manifest copy + coverage profiles + kept/dropped cell manifest."""
        return self.processing_root / "manifests"

    @property
    def scratch_dir(self) -> Path:
        """Transient per-worker temp (cleanable mid-run)."""
        return self.processing_root / "scratch"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML source below env so ``CUBE_`` env vars win over the file.

        The YAML path is supplied per-call via ``init_settings`` (``_yaml_file``);
        see :meth:`from_yaml`.
        """
        yaml_path = getattr(settings_cls, "_yaml_file", None)
        sources: tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
        )
        if yaml_path is not None:
            sources = (*sources, YamlConfigSettingsSource(settings_cls, yaml_path))
        return sources

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> CubeSettings:
        """Load settings from a ``cube.yaml`` (env vars still take precedence).

        Args:
            yaml_path: Path to the cube config YAML.

        Returns:
            The validated :class:`CubeSettings`.
        """
        # Bind the YAML path for settings_customise_sources, then construct.
        cls._yaml_file = yaml_path  # type: ignore[attr-defined]
        try:
            return cls()
        finally:
            cls._yaml_file = None  # type: ignore[attr-defined]


#: Default finetuned ``EncoderWithHead`` checkpoint (Q6 — real weights on disk; see
#: ``docs/agents/KNOWLEDGE.md`` / memory ``snowgalileo-checkpoints-available``). Overridable
#: via ``inference.yaml`` / ``INFER_CHECKPOINT``; existence is validated at run time, not import.
DEFAULT_CHECKPOINT = Path(
    "logging_checkpoints/snowgalileo_finetune/clear_pretrained_20_216tcxve.pth"
)

#: Default eval config (``configs/eval/``) — drives encoder size token, ``sigmoid_slope``,
#: and the per-mode head ``eval_config`` for the model build (mirrors ``eval_only.py``).
DEFAULT_EVAL_CONFIG_NAME = "fsc_inference_bow_river_tiny.json"


class InferenceSettings(BaseSettings):
    """Typed Stage-2 *inference-run* configuration (``inference.yaml``).

    Sibling to :class:`CubeSettings`, with the same YAML+env precedence. It carries
    only *how to run the model* — checkpoint, eval config, decoder mode, batch size,
    device, and the daily-FSC output dir (Q5 override). The *sweep definition* (window,
    mode, archive/processing roots) stays in ``cube.yaml`` so there is a single source
    of truth for it; the inference script reads both configs and passes the window from
    ``CubeSettings``.

    Precedence (highest first): ``INFER_``-prefixed env vars → ``inference.yaml`` → field
    defaults.

    Attributes:
        checkpoint: Finetuned ``EncoderWithHead`` ``.pth``. **Required at run time** — the
            inference script asserts it exists and fails loudly rather than silently
            initializing random weights (an all-random sweep yields a meaningless COG).
        eval_config_name: Eval JSON under ``configs/eval/`` (model-size token + head config).
        decoder_mode: Head decoding strategy (``finetune`` / ``linear_probe`` /
            ``attention_probe``).
        batch_size: Cells per encoder forward pass.
        device: Torch device string (``cpu`` / ``cuda``).
        out_dir: Daily-FSC COG output dir. ``None`` → the script defaults it to
            :pyattr:`CubeSettings.daily_fsc_dir` (Q5 override point).
        export_workers: Process-pool workers for the driver's per-day parallel cube
            pre-export (each worker holds ~600 MB after the windowed-read fix); clamped
            to ``min(cpu_count, cells)`` at run time.
    """

    model_config = SettingsConfigDict(
        env_prefix="INFER_", extra="ignore", frozen=True
    )

    checkpoint: Path = DEFAULT_CHECKPOINT
    eval_config_name: str = DEFAULT_EVAL_CONFIG_NAME
    decoder_mode: str = "finetune"
    batch_size: Annotated[int, Field(gt=0)] = 8
    device: str = "cpu"
    out_dir: Path | None = None
    export_workers: Annotated[int, Field(gt=0)] = 8

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the YAML source below env so ``INFER_`` env vars win over the file."""
        yaml_path = getattr(settings_cls, "_yaml_file", None)
        sources: tuple[PydanticBaseSettingsSource, ...] = (
            init_settings,
            env_settings,
            dotenv_settings,
        )
        if yaml_path is not None:
            sources = (*sources, YamlConfigSettingsSource(settings_cls, yaml_path))
        return sources

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> InferenceSettings:
        """Load settings from an ``inference.yaml`` (env vars still take precedence).

        Args:
            yaml_path: Path to the inference config YAML.

        Returns:
            The validated :class:`InferenceSettings`.
        """
        cls._yaml_file = yaml_path  # type: ignore[attr-defined]
        try:
            return cls()
        finally:
            cls._yaml_file = None  # type: ignore[attr-defined]
