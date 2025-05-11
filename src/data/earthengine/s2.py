from datetime import date

import ee

from src.data.earthengine.utils import create_placeholder, date_to_string

# TODO: check if we have to convert no data values to double


# After 2022-01-25, Sentinel-2 scenes with PROCESSING_BASELINE '04.00' or
# above have their DN (value) range shifted by 1000. The HARMONIZED
# collection shifts data in newer scenes to be in the same range as in older scenes.
image_collection = "COPERNICUS/S2_HARMONIZED"

ALL_S2_BANDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
# Snow-specific bands
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

REMOVED_BANDS = [item for item in ALL_S2_BANDS if item not in S2_BANDS]
S2_SHIFT_VALUES = [float(0.0)] * len(S2_BANDS)
S2_DIV_VALUES = [float(1e4)] * len(S2_BANDS)

S2_CLOUD_BAND = ["QA60"]


def get_single_s2_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(S2_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, S2_BANDS).toDouble()

    return image


def get_s2_cloud_flag(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    cloud_bitflag = (
        ee.ImageCollection(image_collection)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(S2_CLOUD_BAND)
    ).first()

    if cloud_bitflag.getInfo() is None:
        return create_placeholder(region, S2_CLOUD_BAND).toDouble()

    return cloud_bitflag
