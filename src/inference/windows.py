"""Temporal-window helpers for the inference sweep (TASK-015, PLAN §5).

Two pure date helpers drive the driver loop:

- :func:`inference_days` — every day in the configured ``[window_start, window_end]``
  inference range (the outer sweep; one daily FSC mosaic per element).
- :func:`eight_day_window` — the ``NUM_TIMESTEPS``-day model input window
  ``[window_end - 7 … window_end]`` ascending, ending at a single inference day.

These mirror the exporter's own window derivation (``exporter._window_days``) and
the grid generator's day enumeration (``grid._window_days``) but are the public,
inference-side contract so the driver does not reach into another module's
private helper. The CSV ``date`` column is **never** read here (PLAN §8 Q4 /
SPEC AC-31): the days come only from the configured window.
"""

from __future__ import annotations

import datetime

from src.data.config import DAYS_PER_TIMESTEP, NUM_TIMESTEPS


def inference_days(
    window_start: datetime.date,
    window_end: datetime.date,
) -> list[datetime.date]:
    """Return every inference day in ``[window_start, window_end]`` inclusive.

    Args:
        window_start: First inference day (inclusive).
        window_end: Last inference day (inclusive).

    Returns:
        Ascending list of dates; one daily FSC mosaic is produced per element.

    Raises:
        ValueError: If ``window_end`` precedes ``window_start``.
    """
    if window_end < window_start:
        raise ValueError(f"window_end {window_end} precedes window_start {window_start}.")
    span = (window_end - window_start).days
    return [window_start + datetime.timedelta(days=offset) for offset in range(span + 1)]


def eight_day_window(window_end: datetime.date) -> list[datetime.date]:
    """Return the model input window ``[window_end - 7 … window_end]`` ascending.

    The window holds :data:`~src.data.config.NUM_TIMESTEPS` days at a
    :data:`~src.data.config.DAYS_PER_TIMESTEP`-day stride, ending at the
    prediction day ``window_end`` (the last element). This is the same window the
    exporter assembles a cube for.

    Args:
        window_end: The window-end (prediction) day.

    Returns:
        The ``NUM_TIMESTEPS`` window days in ascending order; ``days[-1]`` is
        ``window_end``.
    """
    return [
        window_end - datetime.timedelta(days=DAYS_PER_TIMESTEP * offset)
        for offset in reversed(range(NUM_TIMESTEPS))
    ]
