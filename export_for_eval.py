"""
Script to export data to a Google Cloud Bucket
"""

import argparse

from src.data import EarthEngineExporterEval

# os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", type=str, default="url")
argparser.add_argument("--check_gcp", type=bool, default=False)
argparser.add_argument("--tifs_folder", type=str, default="landsat_eval_tifs/100m_tif_global")
argparser.add_argument("--mask_folder", type=str, default="landsat_eval_masks/all/100m_mask_global")
args = argparser.parse_args().__dict__

exporter = EarthEngineExporterEval(
    check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"]
)
exporter.export_from_filename_for_folder(
    folder=args["mask_folder"],
)
