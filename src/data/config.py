from pathlib import Path

DAYS_PER_TIMESTEP = 1

NUM_TIMESTEPS = 16

# time range to sample a random time window from. End year is inclusive (START_YEAR <= N <= END_YEAR)
# if the season spans two years, the end year will be the following year
# (i.e., if the end year is 2019, it is possible to get data from early 2020)
# for the start year, we are limited by Sentinel-3 data availability (starting 2016-10-18)
# This means effectively, we can sample from (START_YEAR - 1)-12-16 to (END_YEAR + 1)-02-28
START_YEAR = 2017
END_YEAR = 2020

EXPORTED_HEIGHT_WIDTH_METRES = 500
# this is the maximum patch_size * num_patches.
# we will need to change this if that assumption changes
# Note: 96 for 1km x 1km
DATASET_OUTPUT_HW = 48

# the idea is that for exporting different data, we will only have to change this dictionary in the end
# i.e., sort the modalities into different shape_types, add / remove satellite modalities
# for using / not using modalities, the "active" flag should get used
MODALITIES = {
    "s1": {
        "original_resolution": 10,
        "shape_type": "s_t_h_x",
        "active": True,
        "export": True,
    },
    "s2": {
        "original_resolution": 10,
        "shape_type": "s_t_h_x",
        "active": True,
        "export": True,
    },
    "landsat": {
        "original_resolution": 30,
        "shape_type": "s_t_h_x",
        "active": True,
        "export": True,
    },
    "s3": {
        "original_resolution": 300,
        "shape_type": "t_x",
        "active": True,
        "export": True,
    },
    "modis": {
        "original_resolution": 500,
        "shape_type": "t_x",
        "active": True,
        "export": True,
    },
    "viirs_fine": {
        "original_resolution": 500,
        "shape_type": "t_x",
        "active": True,
        "export": True,
    },
    "viirs_coarse": {
        "original_resolution": 1000,
        "shape_type": "t_x",
        "active": True,
        "export": True,
    },
    "era5": {
        "original_resolution": 11132,
        "shape_type": "t_x",
        "active": True,
        "export": True,
    },
    "srtm": {
        "original_resolution": 30,
        "shape_type": "sp_x",
        "active": True,
        "export": True,
    },
    "location": {
        "original_resolution": None,
        "shape_type": "st_x",
        "active": True,
        "export": False,
    },
    "ndsi": {
        "original_resolution": 500,
        "shape_type": "t_x",
        "active": True,
        "export": False,
    },
    "ndvi": {
        "original_resolution": 500,
        "shape_type": "t_x",
        "active": False,
        "export": False,
    },
}

# inclusive (i.e., the end date of a season is included in the season)
SEASONS = {
    "early": ("10-01", "12-15"),
    "mid": ("12-16", "02-28"),
    "late": ("03-01", "06-30"),
}

NO_DATA_VALUE = -9999

USE_INDECES = False

EE_PROJECT = "ee-marlena"
EE_BUCKET_TIFS = None
EE_DRIVE_FOLDER_NAME = "snow_ee_exports_20241226"
EE_DRIVE_FOLDER_ID = "1cL7tEHhC92UHmuwdEgH0ero6aXQqhBcb"
EE_FOLDER_TIFS = "tifs4"
EE_FOLDER_H5PYS = "h5pys"

DATA_FOLDER = Path(__file__).parents[2] / "data"
TIFS_FOLDER = DATA_FOLDER / "tifs3000"
NORMALIZATION_DICT_FILENAME = "normalizing_dict.json"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
ENCODER_FILENAME = "encoder.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
TARGET_ENCODER_FILENAME = "target_encoder.pt"
DECODER_FILENAME = "decoder.pt"
CONFIG_FILENAME = "config.json"
