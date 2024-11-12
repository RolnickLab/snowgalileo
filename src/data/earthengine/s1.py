from datetime import date
from typing import Tuple
import numpy as np

import ee

from .utils import date_to_string

image_collection = "COPERNICUS/S1_GRD"
S1_BANDS = ["VV", "VH"]
# EarthEngine estimates Sentinel-1 values range from -50 to 1
S1_SHIFT_VALUES = [25.0, 25.0]
S1_DIV_VALUES = [25.0, 25.0]


# TODO: check if we are OK in using both orbit passes or should constrain on one
# (would leave us with less frequent images)


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

    s1 = ee.ImageCollection(image_collection).filterDate(startDate, endDate).filterBounds(region)

    if (s1.size().getInfo()== 0):
        print("No S1 Image on date: {}".format(start_date))
        return np.nan

    # different areas have either ascending, descending coverage or both.
    # https://sentinel.esa.int/web/sentinel/missions/sentinel-1/observation-scenario
    # we want the coverage to be consistent (so don't want to take both) but also want to
    # take whatever is available
    orbit = s1.filter(
        ee.Filter.eq("orbitProperties_pass", s1.first().get("orbitProperties_pass"))
    ).filter(ee.Filter.eq("instrumentMode", "IW"))

    vv = orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    vh = orbit.filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))

    composite = ee.Image.cat(
        [
            (vv.select("VV")).first(),
            (vh.select("VH")).first(),
        ]
    ).clip(region)

    # rename to the bands
    image = composite.select(S1_BANDS)

    return image

