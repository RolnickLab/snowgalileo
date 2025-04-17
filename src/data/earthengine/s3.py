from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "COPERNICUS/S3/OLCI"
S3_BANDS = ["Oa17_radiance", "Oa21_radiance"]

# TODO: change these values
S3_SHIFT_VALUES = [float(0.0)] * len(S3_BANDS)
S3_DIV_VALUES = [float(1.0)] * len(S3_BANDS)


def get_single_s3_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(S3_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, S3_BANDS).toDouble()

    return image
