from datetime import date

import ee

from snow_galileo.data.earthengine.utils import create_placeholder, date_to_string

image_collection = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_BANDS = [
    "skin_temperature",
    "temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]
# for temperature, shift to celcius and then divide by 35 based on notebook (ranges from)
# 37 to -22 degrees celcius
# For rainfall, based on
# https://github.com/nasaharvest/lem/blob/main/notebooks/exploratory_data_analysis.ipynb
ERA5_SHIFT_VALUES = [-272.15, -272.15, 0.0, 0.0, 0.0]
ERA5_DIV_VALUES = [35.0, 35.0, 0.03, float(1e4), float(1e4)]


def get_single_era5_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(ERA5_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, ERA5_BANDS).toDouble()

    return image
