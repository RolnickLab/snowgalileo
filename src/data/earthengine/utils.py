import json
import os
import random
import shutil
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
    constant_bands = [ee.Image(ee.constant(fill_value)).rename(band) for band in selected_bands]

    placeholder_image = ee.Image.cat(constant_bands).clip(region)
    return placeholder_image


def sample_time_window(start_date: str, end_date: str, window_size: int, seed=None):
    """
    Sample random time window within a specified date range.

    Args:
        start_date: Start of the timeframe in 'YYYY-MM-DD' format.
        end_date: End of the timeframe in 'YYYY-MM-DD' format.
        window_size: Length of each time window in days.

    Returns:
        list of tuples: Each tuple contains the start and end dates of a sampled time window.
    """
    if seed is not None:
        random.seed(seed)

    start_date_tp = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_tp = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_date_tp - start_date_tp).days + 1

    # ensure the window fits in the range
    max_start_day = total_days - window_size
    if max_start_day < 0:
        raise ValueError("Window size is larger than the total date range.")

    random_start = random.randint(0, max_start_day + 1)

    window_start = start_date_tp + timedelta(days=random_start)
    window_end = window_start + timedelta(days=window_size - 1)
    time_window = (window_start.date(), window_end.date())

    return time_window


def sample_season_year(season, start_year, end_year, seed=None):
    """
    Randomly samples a year between start_year and end_year and assigns it to the season.

    Args:
        season: Tuple with season name as first and date ranges as second item.
        start_year (int): Start year for random sampling.
        end_year (int): End year for random sampling.

    Returns:
        dict: A dictionary with seasons as keys and randomly sampled year-specific date ranges.
    """
    if seed is not None:
        random.seed(seed)

    season, (start_date, end_date) = season

    if end_date.startswith("02"):
        # We can sample from the previous year if we have the mid season
        # TODO: This is hacky, we should handle this better
        assert start_date.startswith("12")
        start_year = start_year - 1
        sampled_year = random.randint(start_year, end_year)
        # If the season spans two years (e.g., mid: "12-15" to "02-28"), handle it
        season_with_year = (
            f"{sampled_year}-{start_date}",  # Start year remains the sampled year
            f"{sampled_year + 1}-{end_date}",  # End year goes into the next year
        )
    else:
        sampled_year = random.randint(start_year, end_year)
        season_with_year = (
            f"{sampled_year}-{start_date}",
            f"{sampled_year}-{end_date}",
        )

    return season_with_year


def get_location_season_identifier(filename) -> str:
    return filename.split("_dates=")[0] + ".tif"


def copy_files_with_partial_check(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    dest_files = os.listdir(dest_folder)
    dest_location_season = {get_location_season_identifier(f) for f in dest_files}

    for file_name in os.listdir(src_folder):
        src_file = os.path.join(src_folder, file_name)
        if os.path.isfile(src_file):
            src_location_season = get_location_season_identifier(file_name)
            if src_location_season in dest_location_season:
                print(f"Duplicate found, skipping: {src_location_season}")
            else:
                dest_file = os.path.join(dest_folder, file_name)
                shutil.copy2(src_file, dest_file)  # Copy the file
                print(f"Copied: {src_file} to {dest_file}")
