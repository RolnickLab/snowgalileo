from datetime import date

import ee

from .utils import get_monthly_data

image_collection = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG"
VIIRS_BANDS = ["avg_rad"]
VIIRS_SHIFT_VALUES = [0.0]
# visually checked - this seems much more reasonable than
# the GEE estimate
VIIRS_DIV_VALUES = [100]


def get_single_viirs_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    if (start_date.year == 2023) & (start_date.month == 10):
        # for some reason, VIIRS data for October 2023 is missing
        # so we replace it with November 2023 data
        start_date = date(start_date.year, 11, 1)

    return get_monthly_data(image_collection, VIIRS_BANDS, region, start_date, unmask=True)
