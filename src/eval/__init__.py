from .cropharvest_eval import BinaryCropHarvestEval, MultiClassCropHarvestEval
from .eurosat_eval import EuroSatEval
from .pastis_pixel_eval import PastisEval
from .so2sat_eval import So2SatEval
from .treesat_eval import TreeSatEval

__all__ = [
    "TreeSatEval",
    "EuroSatEval",
    "So2SatEval",
    "BinaryCropHarvestEval",
    "MultiClassCropHarvestEval",
    "PastisEval",
]
