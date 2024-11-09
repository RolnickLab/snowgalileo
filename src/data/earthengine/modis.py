from datetime import date
from typing import Tuple

import ee

from .utils import date_to_string, get_closest_dates

image_collection_terra = "MODIS/061/MOD09GA"

# TODO (optional): include these products
image_collection_aqua = "MODIS/061/MYD09GA"
image_collection_albedo = "MODIS/061/MCD43A1"
image_collection_terra_snow_cover = "MODIS/061/MOD10A1"

MODIS_BANDS = ["sur_refl_b03", "sur_refl_b04", "sur_refl_b05", "sur_refl_b06", "sur_refl_b07"]
MODIS_SHIFT_VALUES = [-7950.0, -7950.0, -7950.0, -7950.0, -7950.0]
MODIS_DIV_VALUES = [8050, 8050, 8050, 8050, 8050]

# TODO (optional): cloud handling