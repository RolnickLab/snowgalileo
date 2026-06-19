import argparse

from src.data import EarthEngineExporterEval

argparser = argparse.ArgumentParser(
    description="Starter script for exporting input data to be used for inference from Google Earth Engine."
)
argparser.add_argument(
    "--mode",
    type=str,
    default="url",
    choices=["cloud", "drive", "url"],
    help="We can export data from GEE using different modes. For SnowGalileo, we have solely used the URL mode because it is fastest and for free, although the other might be more accurate and can export more data at once. Could be helpful when wanting to predict larger map areas.",
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
    default="rockies_march",
    help="Folder name of the folder where the exported tifs should be stored.",
)
argparser.add_argument(
    "--path_to_csv",
    type=str,
    default="rockies_march.csv",
    help="Filename of the csv file that stores locations and dates to be exported. Here, the file must specify bounding box bounds in UTM format.",
)
args = argparser.parse_args().__dict__

exporter = EarthEngineExporterEval(
    check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"]
)
exporter.export_from_csv_utm(csv_file=args["path_to_csv"])
