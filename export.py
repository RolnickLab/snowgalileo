"""
Script to export data to a Google Cloud Bucket
"""

import argparse
import os

import geopandas

from src.data import EarthEngineExporter
from src.data.config import DATA_FOLDER, EE_PROJECT
from src.data.earthengine.eo import LAT, LON

#os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--start_export_from_idx", type=int, default=0)
argparser.add_argument("--num_exports", type=int, default=3000)
argparser.add_argument("--filename", type=str, default="combined_points.csv")
argparser.add_argument("--mode", type=str, default="drive")
args = argparser.parse_args().__dict__

filepath = DATA_FOLDER / "pretraining_points" / args["filename"]
assert filepath.exists()
latlons = geopandas.read_file(filepath)

if LAT not in latlons.columns:
    latlons[LON] = latlons.geometry.centroid.x.values
    latlons[LAT] = latlons.geometry.centroid.y.values

exporter = EarthEngineExporter(check_gcp=True, mode=args["mode"])
exporter.export_for_latlons(latlons[args["start_export_from_idx"] :], args["num_exports"])
