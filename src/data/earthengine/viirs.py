from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "NASA/VIIRS/002/VNP09GA"
VIIRS_FINE_BANDS = ["I1", "I3"]
VIIRS_COARSE_BANDS = ["M5", "M7", "M10", "M11"]
VIIRS_FINE_SHIFT_VALUES = [-0.795, -0.795]
VIIRS_FINE_DIV_VALUES = [0.805, 0.805]
VIIRS_COARSE_SHIFT_VALUES = [-0.795, -0.795, -0.795, -0.795]
VIIRS_COARSE_DIV_VALUES = [0.805, 0.805, 0.805, 0.805]

ALL_VIIRS_BANDS = VIIRS_FINE_BANDS + VIIRS_COARSE_BANDS

def get_single_viirs_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
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
        .select(ALL_VIIRS_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, ALL_VIIRS_BANDS).toDouble()

    return image