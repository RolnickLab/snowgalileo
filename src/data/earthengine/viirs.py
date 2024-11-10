from datetime import date
from typing import Tuple

import ee

from .utils import date_to_string, get_closest_dates

image_collection = "NASA/VIIRS/002/VNP09GA"
VIIRS_BANDS_500m = ["I1", "I3"]
VIIRS_BANDS_1000m = ["M5", "M7", "M10", "M11"]
VIIRS_500m_SHIFT_VALUES = [-0.795, -0.795]
VIIRS_500m_DIV_VALUES = [0.805, 0.805]
VIIRS_1000m_SHIFT_VALUES = [-0.795, -0.795, -0.795, -0.795]
VIIRS_1000m_DIV_VALUES = [0.805, 0.805, 0.805, 0.805]

def get_single_viirs_500m_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(VIIRS_BANDS_500m_BANDS)
    ).first().toDouble()

    return image

def get_single_viirs_1000m_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(VIIRS_BANDS_1000m_BANDS)
    ).first().toDouble()

    return image