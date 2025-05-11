from datetime import date

import ee

from src.data.earthengine.utils import create_placeholder, date_to_string

image_collection_l08 = "LANDSAT/LC08/C02/T1_TOA"
image_collection_l09 = "LANDSAT/LC09/C02/T1_TOA"

# Snow-specific Landsat bands
ORIG_LANDSAT_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
]
LANDSAT_BANDS = [
    "B2_landsat",
    "B3_landsat",
    "B4_landsat",
    "B5_landsat",
    "B6_landsat",
    "B7_landsat",
]

LANDSAT_SHIFT_VALUES = [float(0.0)] * len(LANDSAT_BANDS)
LANDSAT_DIV_VALUES = [float(1e4)] * len(LANDSAT_BANDS)

LANDSAT_CLOUD_BANDS = ["QA_PIXEL"]


# first checks if Landsat 9 is available, if not, it uses Landsat 8
def get_single_landsat_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    image = (
        ee.ImageCollection(image_collection_l09)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(ORIG_LANDSAT_BANDS)
    ).first()

    if image.getInfo() is None:
        image = (
            ee.ImageCollection(image_collection_l08)
            .filterBounds(region)
            .filterDate(startDate, endDate)
            .select(ORIG_LANDSAT_BANDS)
        ).first()

        if image.getInfo() is None:
            # If no image is found, create a placeholder image with the same bands
            # and the specified region
            return create_placeholder(region, LANDSAT_BANDS).toDouble()

    # Rename the bands to be unique
    renamed_image = image.select(ORIG_LANDSAT_BANDS, LANDSAT_BANDS)

    return renamed_image


def get_landsat_cloud_flag(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    startDate = ee.Date(date_to_string(start_date))
    endDate = ee.Date(date_to_string(end_date))

    cloud_bitflag = (
        ee.ImageCollection(image_collection_l09)
        .filterBounds(region)
        .filterDate(startDate, endDate)
        .select(LANDSAT_CLOUD_BAND)
    ).first()

    if cloud_bitflag.getInfo() is None:
        cloud_bitflag = (
            ee.ImageCollection(image_collection_l08)
            .filterBounds(region)
            .filterDate(startDate, endDate)
            .select(LANDSAT_CLOUD_BAND)
        ).first()

        if cloud_bitflag.getInfo() is None:
            # If no image is found, create a placeholder image with the same bands
            # and the specified region
            return create_placeholder(region, LANDSAT_CLOUD_BAND).toDouble()

    return cloud_bitflag
