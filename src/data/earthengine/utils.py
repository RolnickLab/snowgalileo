import json
import os
import random
from datetime import date, datetime, timedelta
from typing import Union

import ee

from ..config import NO_DATA_VALUE


def get_ee_credentials():
    gcp_sa_key = os.environ.get("GCP_SA_KEY")
    if gcp_sa_key is not None:
        gcp_sa_email = json.loads(gcp_sa_key)["client_email"]
        print(f"Logging into EarthEngine with {gcp_sa_email}")
        return ee.ServiceAccountCredentials(gcp_sa_email, key_data=gcp_sa_key)
    else:
        print("Logging into EarthEngine with default credentials")
        return "persistent"


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


def create_placeholder(region: ee.Geometry, selected_bands, fill_value=NO_DATA_VALUE):
    """
    Creates a placeholder image for a region with constant values for each band in selected_bands.
    """
    constant_bands = [ee.Image.constant(fill_value).rename(band) for band in selected_bands]

    placeholder_image = ee.Image.cat(constant_bands).clip(region)
    return placeholder_image


def sample_time_window(start_date: str, end_date: str, window_size: int):
    """
    Sample random time window within a specified date range.

    Args:
        start_date: Start of the timeframe in 'YYYY-MM-DD' format.
        end_date: End of the timeframe in 'YYYY-MM-DD' format.
        window_size: Length of each time window in days.

    Returns:
        list of tuples: Each tuple contains the start and end dates of a sampled time window.
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_date - start_date).days + 1

    # ensure the window fits in the range
    max_start_day = total_days - window_size
    if max_start_day < 0:
        raise ValueError("Window size is larger than the total date range.")

    random_start = random.randint(0, max_start_day + 1)

    window_start = start_date + timedelta(days=random_start)
    window_end = window_start + timedelta(days=window_size - 1)
    time_window = (window_start.date(), window_end.date())

    return time_window
