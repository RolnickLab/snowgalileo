from datetime import date

import ee

EE_WC_BANDS = ["Map"]
# from https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200#bands
NUM_WC_CLASSES = 11
WC_CLASS_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
WC_BANDS_NAMES = [
    "WC_tree_cover",
    "WC_shrubland",
    "WC_grassland",
    "WC_cropland",
    "WC_built_up",
    "WC_bare_sparse_vegetation",
    "WC_snow_and_ice",
    "WC_permanent_water_bodies",
    "WC_herbaceous_wetland",
    "WC_mangroves",
    "WC_moss_and_lichen",
]
WC_SHIFT_VALUES = [0] * NUM_WC_CLASSES
WC_DIV_VALUES = [1] * NUM_WC_CLASSES


def get_single_ee_wc_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    wc = ee.ImageCollection("ESA/WorldCover/v200").filterBounds(region).select(EE_WC_BANDS).first()

    return wc
