"""Canonical band order + filename contract for the local-source pipeline.

This module is the **single source of truth** for two things the
``LocalSourceExporter`` (TASK-004) and every adapter must agree on with the
**unchanged** downstream loader (``LandsatEvalDataset``):

1. **Band order.** The dynamic (per-timestep) and static band-name lists are
   **re-exported verbatim from the Earth Engine module** — they are never retyped
   here. Retyping would let the two drift; importing keeps them byte-identical
   (SPEC AC-5 / AC-26).

   *Source of truth = ``src.data.earthengine.eo_eval``*, **not** ``eo``. Both
   modules derive the same lists from the shared ``MODALITIES`` config (so they
   are value-equal), but the downstream consumer that actually reads our exported
   tifs — ``src/fsc/landsat_eval.py`` — imports its band lists from ``eo_eval``
   (see ``landsat_eval.py:40-57``). Mirroring the consumer's source keeps the
   contract honest. We additionally assert equality against ``eo`` so the two
   duplicated derivations can never silently diverge without a test failing.

2. **Filename.** :func:`build_cube_filename` produces the one filename the loader's
   ``prediction_month_from_file`` (``landsat_eval.py:171-181``) parses correctly:
   ``PR_{YYYYMMDD}_{LAT}_{LON}_SC00.tif`` (SPEC FR-18 / AC-9).

What the exporter writes per cell (the tif's band sequence), matching
``create_ee_image`` (``eo.py:411-464``):

    [ DYNAMIC_BANDS for timestep 0,  DYNAMIC_BANDS for timestep 1, ...,
      DYNAMIC_BANDS for timestep NUM_TIMESTEPS-1 ]  +  STATIC_BANDS

i.e. ``len(DYNAMIC_BANDS) * NUM_TIMESTEPS + len(STATIC_BANDS)`` bands total.

**Name-coincidence warning.** ``eo.py`` also defines a list literally named
``STATIC_BANDS = ["x", "y", "z"]`` — those are the *location* channels
(``static_x``) the loader synthesises from lat/lon; they are a **different
concept** and are **never written to the tif**. Our :data:`STATIC_BANDS` here is
the exporter's *static spatial stack* (``EE_SPACE_BANDS`` = DEM/slope/aspect/Map).
Do not conflate them.
"""

from __future__ import annotations

import re
from datetime import date

from src.data.config import NUM_TIMESTEPS
from src.data.earthengine import eo as _eo
from src.data.earthengine import eo_eval as _eo_eval

# --- Band order (re-exported, never retyped) ------------------------------- #

#: Per-timestep dynamic band order the exporter writes (38 bands):
#: S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5 + 3 cloud flags.
DYNAMIC_BANDS: list[str] = list(_eo_eval.EO_ALL_DYNAMIC_IN_TIME_BANDS)

#: Static spatial stack the exporter appends once after the dynamic block:
#: ``["DEM", "slope", "aspect", "Map"]`` (WorldCover stays a single ``Map`` band;
#: the loader one-hot-encodes it internally — we do NOT emit the one-hot form).
STATIC_BANDS: list[str] = list(_eo_eval.EE_SPACE_BANDS)

#: Total band count of one exported cube tif (used by the exporter and the
#: loader's channel arithmetic at ``landsat_eval.py:269``).
TOTAL_BANDS: int = len(DYNAMIC_BANDS) * NUM_TIMESTEPS + len(STATIC_BANDS)

# Guard: the two EE modules derive these lists independently from MODALITIES.
# If they ever diverge, fail loudly at import rather than ship a mismatched tif.
assert DYNAMIC_BANDS == list(_eo.EO_ALL_DYNAMIC_IN_TIME_BANDS), (
    "eo_eval and eo dynamic band orders diverged — band-layout contract broken."
)
assert STATIC_BANDS == list(_eo.EE_SPACE_BANDS), (
    "eo_eval and eo static band orders diverged — band-layout contract broken."
)


def full_band_order() -> list[str]:
    """Return the exporter's full per-tif band sequence (dynamic×T then static).

    Timestep-``t`` dynamic bands are suffixed ``_t{t}`` (``t`` 0-based) so the
    flattened list is unambiguous; the static bands keep their bare names. This
    matches ``create_ee_image``'s interleave (band ``b`` at timestep ``t`` is the
    ``t``-th repeat of the dynamic block) for naming/debug purposes — the loader
    keys off **position**, not these names, so the suffix scheme is internal.

    Returns:
        A list of ``TOTAL_BANDS`` band-name strings.
    """
    bands: list[str] = []
    for t in range(NUM_TIMESTEPS):
        bands.extend(f"{name}_t{t}" for name in DYNAMIC_BANDS)
    bands.extend(STATIC_BANDS)
    return bands


# --- Filename contract ------------------------------------------------------ #

#: Regex every exporter filename must match (SPEC FR-18 / AC-9). ``LAT``/``LON``
#: are signed decimal degrees; ``SC`` carries a (synthetic) cloud-score suffix.
CUBE_FILENAME_REGEX: str = r"^PR_\d{8}_-?\d+\.\d+_-?\d+\.\d+_SC\d+\.tif$"

#: Compiled form for callers that prefer a pattern object.
CUBE_FILENAME_PATTERN: re.Pattern[str] = re.compile(CUBE_FILENAME_REGEX)


def build_cube_filename(
    *,
    window_end: date,
    lat: float,
    lon: float,
    cloud_score: int = 0,
) -> str:
    """Build the exporter's per-cell tif filename (the loader's ``PR`` branch).

    The downstream parser ``prediction_month_from_file`` reads the month from
    ``name.split("_")[1][4:6]`` and ``_tif_to_array`` reads ``lat = parts[2]``,
    ``lon = parts[3]`` (the non-Landsat else-branch, since ``PR`` does not start
    with ``LC``/``LE``). The format below places each field accordingly.

    Args:
        window_end: The 8-day window's end day; its ``YYYYMMDD`` form is field 1
            and its ``.month`` is what the loader recovers.
        lat: Cell-centre latitude in signed decimal degrees (EPSG:4326).
        lon: Cell-centre longitude in signed decimal degrees (EPSG:4326).
        cloud_score: Synthetic cloud-score suffix (``SC{cloud_score:02d}``);
            defaults to ``0`` → ``SC00``.

    Returns:
        A filename of the form ``PR_{YYYYMMDD}_{LAT}_{LON}_SC{cc}.tif`` that
        matches :data:`CUBE_FILENAME_REGEX`.
    """
    return f"PR_{window_end.strftime('%Y%m%d')}_{lat}_{lon}_SC{cloud_score:02d}.tif"
