from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection_terra = "MODIS/061/MOD09GA"

# TODO (optional): include these products or remove

image_collection_aqua = "MODIS/061/MYD09GA"
image_collection_albedo = "MODIS/061/MCD43A1"
image_collection_terra_snow_cover = "MODIS/061/MOD10A1"

MODIS_BANDS = ["sur_refl_b03", "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
MODIS_SHIFT_VALUES = [-7950.0, -7950.0, -7950.0, -7950.0, -7950.0]
MODIS_DIV_VALUES = [8050, 8050, 8050, 8050, 8050]


def get_single_modis_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    dates = ee.DateRange(
        date_to_string(start_date),
        date_to_string(end_date),
    )

    startDate = ee.DateRange(dates).start()
    endDate = ee.DateRange(dates).end()

    image = (
        ee.ImageCollection(image_collection_terra)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(MODIS_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, MODIS_BANDS).toDouble()

    # all imagery has to have the same data type to be compatible
    #return image.toDouble()
    return image
