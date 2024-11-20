from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BANDS = ["skin_temperature", "total_precipitation_sum"]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, 0.0]
ERA5_DIV_VALUES = [35.0, 0.03]

# TODO: add more compatible era5 bands


def get_single_era5_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
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
        .select(ERA5_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, ERA5_BANDS).toDouble()

    # all imagery has to have the same data type to be compatible
    return image.toDouble()
