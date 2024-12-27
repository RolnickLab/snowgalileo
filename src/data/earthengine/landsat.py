from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "LANDSAT/LC08/C02/T1_TOA"

# Snow-specific Landsat bands
ORIG_LANDSAT_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]
LANDSAT_BANDS = [
    "B2_landsat",
    "B3_landsat",
    "B4_landsat",
    "B5_landsat",
    "B6_landsat",
    "B7_landsat",
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
        .select(ORIG_LANDSAT_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, LANDSAT_BANDS).toDouble()

    # Rename the bands to be unique
    renamed_image = image.select(
        ORIG_LANDSAT_BANDS,
        LANDSAT_BANDS
    )

    return renamed_image