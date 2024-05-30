from datetime import date

import ee

from .utils import get_monthly_data

image_collection = "IDAHO_EPSCOR/TERRACLIMATE"
TC_BANDS = ["def", "soil", "aet"]
TC_SHIFT_VALUES = [0.0, 0.0, 0.0]
TC_DIV_VALUES = [4548, 8882, 2000]


def get_single_terraclimate_image(
    region: ee.Geometry, start_date: date, end_date: date
) -> ee.Image:
    return get_monthly_data(image_collection, TC_BANDS, region, start_date, unmask=True)
