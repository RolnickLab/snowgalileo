from pathlib import Path

DAYS_PER_TIMESTEP = 30
NUM_TIMESTEPS = 12
# this is the maximum patch_size * num_patches.
# we will need to change this if that assumption changes
DATASET_OUTPUT_HW = 96

# TODO: Update when ERA5 gets updated
START_YEAR = 2022
END_YEAR = 2023
EXPORTED_HEIGHT_WIDTH_METRES = 1400

EE_PROJECT = "large-earth-model"
EE_BUCKET_TIFS = "presto-tifs"
EE_FOLDER_TIFS = "tifs4"

DATA_FOLDER = Path(__file__).parents[2] / "data"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
