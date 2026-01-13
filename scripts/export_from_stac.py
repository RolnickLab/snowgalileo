"""
Script to export data from pystac
"""

import argparse

from src.data import EarthEngineExporterEval
from src.eval.globalsnowpack import export_from_filename_for_folder

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--tifs_folder", type=str, default="landsat_eval_tifs/patches_UTM_5_95")
argparser.add_argument("--mask_folder", type=str, default="landsat_eval_masks/patches_UTM_5_95")
argparser.add_argument("--start_idx", type=int, default=0)
args = argparser.parse_args().__dict__

export_from_filename_for_folder(folder=args["mask_folder"], start_idx=args["start_idx"])
