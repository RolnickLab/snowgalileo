from .bigearthnet_eval import BigEarthNetEval
from .brick_kiln_eval import BrickKilnEval
from .cashew_plant_eval import CashewPlantEval
from .cropharvest_eval import BinaryCropHarvestEval, MultiClassCropHarvestEval
from .eurosat_eval import EuroSatEval
from .sa_crop_type_eval import SACropEval
from .pastis_patch_eval import PastisPatchEval
from .pastis_pixel_eval import PastisPixelEval
from .so2sat_eval import So2SatEval
from .treesat_eval import TreeSatEval

__all__ = [
    "TreeSatEval",
    "EuroSatEval",
    "So2SatEval",
    "BinaryCropHarvestEval",
    "MultiClassCropHarvestEval",
    "PastisPixelEval",
    "PastisPatchEval",
    "CashewPlantEval",
    "SACropEval",
    "BrickKilnEval",
    "BigEarthNetEval",
]
