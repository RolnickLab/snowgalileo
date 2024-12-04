from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "LANDSAT/LC09/C02/T1_TOA"

# Snow-specific Landsat bands
LANDSAT_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]
LANDSAT_SHIFT_VALUES = [float(0.0)] * len(LANDSAT_BANDS)
LANDSAT_DIV_VALUES = [float(1e4)] * len(LANDSAT_BANDS)


def get_single_landsat_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
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
        .select(LANDSAT_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, LANDSAT_BANDS).toDouble()

    # all imagery has to have the same data type to be compatible
    return image.toDouble()
