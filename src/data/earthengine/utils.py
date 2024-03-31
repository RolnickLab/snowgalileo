from datetime import date
from typing import Union

import ee


def date_to_string(input_date: Union[date, str]) -> str:
    if isinstance(input_date, str):
        return input_date
    else:
        assert isinstance(input_date, date)
        return input_date.strftime("%Y-%m-%d")


def get_closest_dates(mid_date: date, imcol: ee.ImageCollection) -> ee.ImageCollection:
    fifteen_days_in_ms = 1296000000

    mid_date_ee = ee.Date(date_to_string(mid_date))
    # first, order by distance from mid_date
    from_mid_date = imcol.map(
        lambda image: image.set(
            "dateDist",
            ee.Number(image.get("system:time_start"))
            .subtract(mid_date_ee.millis())  # type: ignore
            .abs(),
        )
    )
    from_mid_date = from_mid_date.sort("dateDist", opt_ascending=True)

    # no matter what, we take the first element in the image collection
    # and we add 1 to ensure the less_than condition triggers
    max_diff = ee.Number(from_mid_date.first().get("dateDist")).max(  # type: ignore
        ee.Number(fifteen_days_in_ms)
    )

    kept_images = from_mid_date.filterMetadata("dateDist", "not_greater_than", max_diff)
    return kept_images
