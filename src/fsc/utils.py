from datetime import datetime
from pathlib import Path

from src.data.config import NORTH_HEM_SEASONS


# for season analysis
def extract_season_from_filename(filename: str) -> str:
    """Extract season from filename assuming format: <prefix>_<YYYYMMDD>_<lat>_<lon>.tif.

    Disclaimer: This function was created with the assistance of ChatGPT.
    While thoroughly reviewed and tested by the author, AI-generated code may contain errors.
    """
    parts = Path(filename).stem.split("_")
    try:
        date_str = parts[1]
        month_day = date_str[4:8]  # MMDD
        # convert into MM-DD format
        month_day = month_day[:2] + "-" + month_day[2:]

        for season, (start, end) in NORTH_HEM_SEASONS.items():
            start_date = datetime.strptime(start, "%m-%d")
            end_date = datetime.strptime(end, "%m-%d")
            month_day_date = datetime.strptime(month_day, "%m-%d")

            # normal season (no year wrap)
            if start_date <= end_date:
                if start_date <= month_day_date <= end_date:
                    return season
            else:
                # Season wraps around the year (e.g., winter)
                if month_day_date >= start_date or month_day_date <= end_date:
                    return season
        # If no season matched, return summer season
        return "summer"
    except (ValueError, IndexError):
        raise ValueError(f"Filename '{filename}' does not match expected format.")


class SigmoidSlopeScheduler:
    """Exponential decay."""

    def __init__(self, model, start, end, total_steps):
        self.model = model
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.step_idx = 0

    def step(self):
        t = min(self.step_idx / self.total_steps, 1.0)
        value = self.start * (self.end / self.start) ** t
        self.model.sigmoid_slope.fill_(value)
        self.step_idx += 1
