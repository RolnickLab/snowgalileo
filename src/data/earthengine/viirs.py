from datetime import date
from typing import Tuple

import ee

from .utils import date_to_string, get_closest_dates

image_collection = "NASA/VIIRS/002/VNP09GA"
VIIRS_BANDS_500m = ["I1", "I3"]
VIIRS_BANDS_1000m = ["M5", "M7", "M10", "M11"]
VIIRS_SHIFT_VALUES = [-0.795, -0.795, -0.795, -0.795, -0.795, -0.795]
VIIRS_DIV_VALUES = [0.805, 0.805, 0.805, 0.805, 0.805, 0.805]