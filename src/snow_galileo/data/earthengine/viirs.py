from datetime import date

import ee

from snow_galileo.data.earthengine.utils import create_placeholder, date_to_string

image_collection = "NASA/VIIRS/002/VNP09GA"
VIIRS_FINE_BANDS = ["I1", "I3"]
VIIRS_COARSE_BANDS = ["M5", "M7", "M10", "M11"]
VIIRS_FINE_SHIFT_VALUES = [-0.795, -0.795]
VIIRS_FINE_DIV_VALUES = [0.805, 0.805]
VIIRS_COARSE_SHIFT_VALUES = [-0.795, -0.795, -0.795, -0.795]
VIIRS_COARSE_DIV_VALUES = [0.805, 0.805, 0.805, 0.805]

VIIRS_CLOUD_FLAG_BANDS = ["QF1"]


def get_single_viirs_fine_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(VIIRS_FINE_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, VIIRS_FINE_BANDS).toDouble()

    return image


def get_single_viirs_coarse_image(
    region: ee.Geometry, start_date: date, end_date: date
) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(VIIRS_COARSE_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, VIIRS_COARSE_BANDS).toDouble()

    return image


def get_viirs_cloud_flag(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    cloud_bitflag = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(VIIRS_CLOUD_FLAG_BANDS)
    ).first()

    if cloud_bitflag.getInfo() is None:
        return create_placeholder(region, VIIRS_CLOUD_FLAG_BANDS).toDouble()

    return cloud_bitflag
