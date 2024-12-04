from datetime import date

import ee
import numpy as np

from .utils import date_to_string, create_placeholder

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

    # there is max 1 image per date
    image = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region).select(S1_BANDS).first()

    if image.getInfo() is None:
        return create_placeholder(region, S1_BANDS).toDouble()

    # print the orbit properties to see if we have any data
    print(image.get("orbitProperties_pass").filter(ee.Filter.eq("instrumentMode", "IW")).getInfo())

    # different areas have either ascending, descending coverage or both.
    # https://sentinel.esa.int/web/sentinel/missions/sentinel-1/observation-scenario
    # we want the coverage to be consistent (so don't want to take both) but also want to
    # take whatever is available
    orbit = image.filter(
        ee.Filter.eq("orbitProperties_pass", image.first().get("orbitProperties_pass"))
    ).filter(ee.Filter.eq("instrumentMode", "IW"))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(S1_BANDS)
    ).first()

    # all imagery has to have the same data type to be compatible
    return image.toDouble()
