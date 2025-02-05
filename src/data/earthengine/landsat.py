from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection_l08 = "LANDSAT/LC08/C02/T1_TOA"
image_collection_l09 = "LANDSAT/LC09/C02/T1_TOA"

# Snow-specific Landsat bands
ORIG_LANDSAT_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]
LANDSAT08_BANDS = [
    "B2_landsat08",
    "B3_landsat08",
    "B4_landsat08",
    "B5_landsat08",
    "B6_landsat08",
    "B7_landsat08",
]
LANDSAT09_BANDS = [
    "B2_landsat09",
    "B3_landsat09",
    "B4_landsat09",
    "B5_landsat09",
    "B6_landsat09",
    "B7_landsat09",
]
LANDSAT08_SHIFT_VALUES = [float(0.0)] * len(LANDSAT08_BANDS)
LANDSAT08_DIV_VALUES = [float(1e4)] * len(LANDSAT08_BANDS)

LANDSAT09_SHIFT_VALUES = [float(0.0)] * len(LANDSAT09_BANDS)
LANDSAT09_DIV_VALUES = [float(1e4)] * len(LANDSAT09_BANDS)


def get_single_landsat08_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    image = (
        ee.ImageCollection(image_collection_l08)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(ORIG_LANDSAT_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, LANDSAT08_BANDS).toDouble()

    # Rename the bands to be unique
    renamed_image = image.select(ORIG_LANDSAT_BANDS, LANDSAT08_BANDS)

    return renamed_image

def get_single_landsat09_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    image = (
        ee.ImageCollection(image_collection_l09)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(ORIG_LANDSAT_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, LANDSAT09_BANDS).toDouble()

    # Rename the bands to be unique
    renamed_image = image.select(ORIG_LANDSAT_BANDS, LANDSAT09_BANDS)

    return renamed_image