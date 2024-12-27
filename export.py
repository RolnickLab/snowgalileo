"""
Script to export data to a Google Cloud Bucket
"""

import argparse

import geopandas

from src.data import EarthEngineExporter
from src.data.config import DATA_FOLDER
from src.data.earthengine.eo import LAT, LON

# os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--start_export_from_idx", type=int, default=0)
argparser.add_argument("--num_exports", type=int, default=3000)
argparser.add_argument("--filename", type=str, default="sampling_points_mountains_lat_42-60.csv")
argparser.add_argument("--mode", type=str, default="drive")
argparser.add_argument("--check_gcp", type=bool, default=False)
argparser.add_argument("--export_all_bands", action="store_true", help="Workaround to deal with URL download limit - if false, exclude VIIRS and ERA5")
argparser.add_argument("--tifs_folder", type=str, default="tifs")
args = argparser.parse_args().__dict__

filepath = DATA_FOLDER / "pretraining_points" / args["filename"]
assert filepath.exists()
latlons = geopandas.read_file(filepath)

if LAT not in latlons.columns:
    latlons[LON] = latlons.geometry.centroid.x.values
    latlons[LAT] = latlons.geometry.centroid.y.values

exporter = EarthEngineExporter(check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"])
exporter.export_for_latlons(latlons=latlons[args["start_export_from_idx"] :], num_exports_to_start=args["num_exports"], all_bands=args["export_all_bands"])
