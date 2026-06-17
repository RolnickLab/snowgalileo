import argparse

import geopandas

from src.data import EarthEngineExporter
from src.data.config import DATA_FOLDER
from src.data.earthengine.eo import LAT, LON

argparser = argparse.ArgumentParser(
    description="Starter script for exporting pre-training data from Google Earth Engine."
)
argparser.add_argument(
    "--start_export_from_idx",
    type=int,
    default=0,
    help="In the case of exporting in batches: Set to a higher index to start exporting points lower in the script.",
)
argparser.add_argument(
    "--num_exports",
    type=int,
    default=3000,
    help="There is a limitation of files that can be exported at once. 3000 should be a good choice.",
)
argparser.add_argument(
    "--filename",
    type=str,
    default="sampling_points_mountains_lat_42-60.csv",
    help="Filename to file that stores dates and locations to be exported. The file must be stored in data/pretraining_points/",
)
argparser.add_argument(
    "--mode",
    type=str,
    default="url",
    choices=["cloud", "drive", "url"],
    help="We can export data from GEE using different modes. For SnowGalileo, we have solely used the URL mode because it is fastest and for free, although the other might be more accurate and can export more data at once.",
)
argparser.add_argument(
    "--check_gcp",
    type=bool,
    default=False,
    help="Whether to check Google Cloud Storage before exporting.",
)
argparser.add_argument(
    "--tifs_folder",
    type=str,
    default="tifs",
    help="Folder name of the folder where the exported tifs should be stored.",
)
args = argparser.parse_args().__dict__

filepath = DATA_FOLDER / "pretraining_points" / args["filename"]
assert filepath.exists()
latlons = geopandas.read_file(filepath)

if LAT not in latlons.columns:
    latlons[LON] = latlons.geometry.centroid.x.values
    latlons[LAT] = latlons.geometry.centroid.y.values

exporter = EarthEngineExporter(
    check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"]
)
exporter.export_for_latlons(
    latlons=latlons[args["start_export_from_idx"] :],
    num_exports_to_start=args["num_exports"],
)
