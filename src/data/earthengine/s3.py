from datetime import date
from typing import Tuple
import numpy as np

import ee

from .utils import date_to_string, create_placeholder

image_collection = "COPERNICUS/S3/OLCI"
S3_BANDS = ["Oa17_radiance", "Oa21_radiance"]
S3_SHIFT_VALUES = []
S3_DIV_VALUES = []

def get_single_s3_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:

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
        .select(S3_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, S3_BANDS).toDouble()

    # all imagery has to have the same data type to be compatible
    return image.toDouble()