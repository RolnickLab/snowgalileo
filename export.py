"""
Script to export data to a Google Cloud Bucket
"""
import argparse
import os

import geopandas

from src.data import EarthEngineExporter
from src.data.config import DATA_FOLDER, EE_PROJECT

os.environ["GOOGLE_CLOUD_PROJECT"] = EE_PROJECT

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--num_exports", type=int, default=3000)
args = argparser.parse_args().__dict__

labels = geopandas.read_file(DATA_FOLDER / "dynamic_world_samples.geojson")
exporter = EarthEngineExporter(check_gcp=True)
exporter.export_for_labels(labels, args["num_exports"])
