from pathlib import Path
from datetime import date

DAYS_PER_TIMESTEP = 1
NUM_TIMESTEPS = 10
# this is the maximum patch_size * num_patches.
# we will need to change this if that assumption changes
DATASET_OUTPUT_HW = 96

# TODO: Remove start_year and end_year in case we don't need them (defined in seasons instead)
START_YEAR = 2022
END_YEAR = 2023
EXPORTED_HEIGHT_WIDTH_METRES = 1000

# TODO: adjust seasons based on domain knowledge
SEASONS = {
    "early": ("2023-10-01", "2023-12-31"),
    "mid": ("2023-01-01", "2023-3-31"),
    "late": ("2023-04-01", "2023-6-30"),
}

NO_DATA_VALUE = -9999

USE_INDECES = False

EE_PROJECT = None
EE_BUCKET_TIFS = None
EE_DRIVE_FOLDER_TIFS = "snow_ee_exports"
EE_FOLDER_TIFS = "tifs4"
EE_FOLDER_H5PYS = "h5pys"

DATA_FOLDER = Path(__file__).parents[2] / "data"
TIFS_FOLDER = DATA_FOLDER / "tifs"
NORMALIZATION_DICT_FILENAME = "normalization.json"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
ENCODER_FILENAME = "encoder.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
TARGET_ENCODER_FILENAME = "target_encoder.pt"
DECODER_FILENAME = "decoder.pt"
CONFIG_FILENAME = "config.json"