from pathlib import Path

DAYS_PER_TIMESTEP = 30
NUM_TIMESTEPS = 12
DATASET_OUTPUT_HW = 128

# TODO: Update when ERA5 gets updated
START_YEAR = 2022
END_YEAR = 2023
EXPORTED_HEIGHT_WIDTH_METRES = 1400

EE_PROJECT = "large-earth-model"
EE_BUCKET_TIFS = "presto-tifs"
EE_FOLDER_TIFS = "tifs2"

DATA_FOLDER = Path(__file__).parents[2] / "data"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
