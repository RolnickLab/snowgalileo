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
    return get_monthly_data(image_collection, VIIRS_BANDS, region, start_date)
