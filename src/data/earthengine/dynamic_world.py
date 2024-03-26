from datetime import date

import ee

from .utils import get_closest_dates

ORIGINAL_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_BANDS = [f"DW_{band}" for band in ORIGINAL_BANDS]
DW_SHIFT_VALUES = [0] * len(DW_BANDS)
DW_DIV_VALUES = [1] * len(DW_BANDS)


def get_dw_image_collection(
    region: ee.Geometry, start_date: date, end_date: date
) -> ee.ImageCollection:
    # we start by getting all the data for the range
    dw_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(ee.DateRange(str(start_date), str(end_date)))
        .select(ORIGINAL_BANDS, DW_BANDS)
    )

    return dw_collection


def get_single_dw_image(
    region: ee.Geometry, start_date: date, end_date: date, dw_imcol: ee.ImageCollection
) -> ee.Image:
    mid_date = start_date + ((end_date - start_date) / 2)
    return ee.Image(get_closest_dates(mid_date, dw_imcol).select(DW_BANDS)).clip(region).mean()
