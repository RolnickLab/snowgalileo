from datetime import date

import ee

WC_BANDS = ["Map"]
WC_SHIFT_VALUES = [0] * len(WC_BANDS)
WC_DIV_VALUES = [1] * len(WC_BANDS)


def get_single_wc_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    wc_collection = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .filterBounds(region)
        .filterDate(ee.DateRange(str(start_date), str(end_date)))
        .select(WC_BANDS)
        .mean()
    )

    return wc_collection
