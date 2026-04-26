"""
Script to export data to a Google Cloud Bucket
"""

import argparse

from src.data import EarthEngineExporterEval

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--mode", type=str, default="url")
argparser.add_argument("--check_gcp", type=bool, default=False)
argparser.add_argument("--tifs_folder", type=str, default="rockies_march")
argparser.add_argument("--path_to_csv", type=str, default="rockies_march.csv")
args = argparser.parse_args().__dict__

exporter = EarthEngineExporterEval(
    check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"]
)
exporter.export_from_csv_utm(csv_file=args["path_to_csv"])
