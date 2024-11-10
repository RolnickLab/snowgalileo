import math
from datetime import date

import ee

from .utils import date_to_string

# After 2022-01-25, Sentinel-2 scenes with PROCESSING_BASELINE '04.00' or
# above have their DN (value) range shifted by 1000. The HARMONIZED
# collection shifts data in newer scenes to be in the same range as in older scenes.
image_collection = "COPERNICUS/S2_HARMONIZED"

# removed B1, B9, B10
ALL_S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]
REMOVED_BANDS = [item for item in ALL_S2_BANDS if item not in S2_BANDS]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)


def get_single_s2_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

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
        .select(S2_BANDS)
    ).first()

    if image.getInfo() is None:
        print("No S2 Image on date: {}".format(start_date))
        return np.nan

    return image.toDouble()