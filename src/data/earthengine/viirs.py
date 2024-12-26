from datetime import date

import ee

from .utils import create_placeholder, date_to_string

image_collection = "NASA/VIIRS/002/VNP09GA"
ALL_VIIRS_BANDS = ["I1", "I3", "M5", "M7", "M10", "M11"]
VIIRS_BANDS_500m = ["I1", "I3"]
VIIRS_BANDS_1000m = ["M5", "M7", "M10", "M11"]
VIIRS_SHIFT_VALUES_500m = [-0.795, -0.795]
VIIRS_DIV_VALUES_500m = [0.805, 0.805]
VIIRS_SHIFT_VALUES_1000m = [-0.795, -0.795, -0.795, -0.795]
VIIRS_DIV_VALUES_1000m = [0.805, 0.805, 0.805, 0.805]

def get_single_viirs_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
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
        .select(ALL_VIIRS_BANDS)
    ).first()

    if image.getInfo() is None:
        return create_placeholder(region, ALL_VIIRS_BANDS).toDouble()

    # has to be double to be compatible with the sentinel 1 imagery, which is in
    # float64
    #return image.toDouble()
    return image