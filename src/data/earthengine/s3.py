from datetime import date
from typing import Tuple

import ee

from .utils import date_to_string, get_closest_dates

image_collection = "COPERNICUS/S3/OLCI"
S3_BANDS = ["Oa17_radiance", "Oa21_radiance"]
S3_SHIFT_VALUES = []
S3_DIV_VALUES = []