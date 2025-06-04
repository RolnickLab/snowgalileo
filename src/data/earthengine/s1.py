from datetime import date

import ee

from src.data.earthengine.utils import create_placeholder, date_to_string

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
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    s1 = (
        ee.ImageCollection(image_collection)
        .filterDate(startDate, endDate)
        .filterBounds(region)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    if s1.size().getInfo() == 0:
        return create_placeholder(region, S1_BANDS).toDouble()

    #return create_placeholder(region, S1_BANDS).toDouble()
    image = s1.select(S1_BANDS).first()

    return image
