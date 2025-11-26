from datetime import datetime
from pathlib import Path

from src.data.config import NORTH_HEM_SEASONS


def extract_season_from_filename(filename: str) -> str:
    """Extract season from filename assuming format: <prefix>_<YYYYMMDD>_<lat>_<lon>.tif"""

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
        raise ValueError(
            f"Date {month_day} in filename {filename} does not fall within any defined season"
        )
    except (ValueError, IndexError):
        raise ValueError(f"Filename {filename} does not contain a valid season segment")
