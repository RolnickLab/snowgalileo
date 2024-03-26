from datetime import date

import ee

ORIGINAL_BANDS = [
    "water",
    "trees",
    "grass",
    "flooded_vegetation",
    "crops",
    "shrub_and_scrub",
    "built",
    "bare",
    "snow_and_ice",
]

DW_BANDS = [f"DW_{band}" for band in ORIGINAL_BANDS]
DW_SHIFT_VALUES = [0] * len(DW_BANDS)
DW_DIV_VALUES = [1] * len(DW_BANDS)


def get_single_dw_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    start_date = date(start_date.year, start_date.month, start_date.day)
    end_date = date(end_date.year, end_date.month, end_date.day)

    # we start by getting all the data for the range
    dw_collection = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(ee.DateRange(str(start_date), str(end_date)))
        .select(ORIGINAL_BANDS, DW_BANDS)
    )

    fifteen_days_in_ms = 1296000000
    output_images = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.month < 12:
            next_date = date(current_date.year, current_date.month + 1, 1)
        else:
            next_date = date(current_date.year + 1, 1, 1)
        mid_date = current_date + (next_date - current_date) / 2
        mid_date_ee = ee.Date(str(date(mid_date.year, mid_date.month, mid_date.day)))

        # first, order by distance from mid_date
        from_mid_date = dw_collection.map(
            lambda image: image.set(
                "dateDist",
                ee.Number(image.get("system:time_start")).subtract(mid_date_ee.millis()).abs(),
            )
        )
        from_mid_date = from_mid_date.sort("dateDist", opt_ascending=True)

        # no matter what, we take the first element in the image collection
        # and we add 1 to ensure the less_than condition triggers
        max_diff = ee.Number(from_mid_date.first().get("dateDist")).max(
            ee.Number(fifteen_days_in_ms)
        )

        kept_images = from_mid_date.filterMetadata("dateDist", "not_greater_than", max_diff)
        output_images.append(kept_images.mean())

        current_date = next_date

    return ee.Image.cat(output_images)
