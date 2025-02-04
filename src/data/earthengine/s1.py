from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "COPERNICUS/S1_GRD"
S1_BANDS = ["VV", "VH", "angle"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_SHIFT_VALUES = [25.0, 25.0, 0.0]
S1_DIV_VALUES = [25.0, 25.0, 90.0]


def get_single_s1_image(
    region: ee.Geometry,
    start_date: date,
    end_date: date,
) -> ee.Image:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    s1 = (
        ee.ImageCollection(image_collection)
        .filterDate(startDate, endDate)
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    if s1.size().getInfo() == 0:
        return create_placeholder(region, S1_BANDS).toDouble()

    image = s1.select(S1_BANDS).first()

    return image
