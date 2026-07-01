import argparse

from snow_galileo.data import EarthEngineExporterEval

argparser = argparse.ArgumentParser(
    description="Starter script for exporting input data to be used for fine-tuning and evaluation from Google Earth Engine."
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
    default="landsat_eval_tifs/patches_UTM_5_95",
    help="Path where the exported files should be stored.",
)
argparser.add_argument(
    "--mask_folder",
    type=str,
    default="landsat_eval_masks/patches_UTM_5_95",
    help="Path that stores the label GeoTIFF. These will be used to derive the location bounds and dates for the input data to be exported. The input will be from the same ground area as the label, and covering the same day including 7 days before the label day.",
)
argparser.add_argument("--start_idx", type=int, default=0)
args = argparser.parse_args().__dict__

exporter = EarthEngineExporterEval(
    check_gcp=args["check_gcp"], mode=args["mode"], tifs_folder=args["tifs_folder"]
)
exporter.export_from_filename_for_folder(folder=args["mask_folder"], start_idx=args["start_idx"])
