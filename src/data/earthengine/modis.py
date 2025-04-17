from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection_terra = "MODIS/061/MOD09GA"

MODIS_BANDS = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_b03",
    "sur_refl_b04",
    "sur_refl_b05",
    "sur_refl_b06",
    "sur_refl_b07",
]
MODIS_SHIFT_VALUES = [-7950.0, -7950.0, -7950.0, -7950.0, -7950.0, -7950.0, -7950.0]
MODIS_DIV_VALUES = [8050, 8050, 8050, 8050, 8050, 8050, 8050]


def get_single_modis_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection_terra)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(MODIS_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, MODIS_BANDS).toDouble()

    return image
