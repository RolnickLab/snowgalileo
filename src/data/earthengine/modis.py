from datetime import date

import ee

from src.data.earthengine.utils import create_placeholder, date_to_string

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

MODIS_CLOUD_BAND = ["state_1km"]


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


def get_modis_cloud_flag(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    cloud_bitflag = (
        ee.ImageCollection(image_collection_terra)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(MODIS_CLOUD_BAND)
    ).first()

    if cloud_bitflag.getInfo() is None:
        return create_placeholder(region, MODIS_CLOUD_BAND).toDouble()

    return cloud_bitflag

    """
    cloud_bitflag = ee.Number(cloud_bitflag)
    cloud_state = bitwiseExtract(cloud_bitflag, 0, 1)

    # mapping from https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA#bands
    if cloud_state.eq(0):
        state = "clear"
    elif cloud_state.eq(1):
        state = "cloudy"
    elif cloud_state.eq(2):
        state = "mixed"
    elif cloud_state.eq(3):
        state = "clear"
    else:
        raise ValueError("Invalid cloud state value")

    return state
    """


def bitwiseExtract(value, fromBit, toBit):
    """
    Modified from https://gis.stackexchange.com/questions/349371/creating-cloud-free-images-out-of-a-mod09a1-modis-image-in-gee/349401#349401

    Utility to extract bitmask values.
    Look up the bit-ranges in the catalog.
    """
    maskSize = ee.Number(1).add(toBit).subtract(fromBit)
    mask = ee.Number(1).leftShift(maskSize).subtract(1)
    return value.rightShift(fromBit).bitwiseAnd(mask)
