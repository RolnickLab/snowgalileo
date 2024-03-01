from pathlib import Path

DAYS_PER_TIMESTEP = 30
DEFAULT_NUM_TIMESTEPS = 12

# TODO: Update when ERA5 gets updated
START_YEAR = 2022
END_YEAR = 2023
EXPORTED_HEIGHT_WIDTH_METRES = 1400

EE_PROJECT = "large-earth-model"
EE_BUCKET_TIFS = "presto-tifs"

DATA_FOLDER = Path(__file__).parents[2] / "data"

# These are model configurations, and should
# probably live somewhere else
VIT_PATCH_SIZE = 16
PRESTO_INPUT_SIZE = 32
CROMA_INPUT_SIZE = 128

assert CROMA_INPUT_SIZE % VIT_PATCH_SIZE == 0
assert PRESTO_INPUT_SIZE % VIT_PATCH_SIZE == 0
