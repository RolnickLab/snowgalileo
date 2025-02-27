from pathlib import Path

DAYS_PER_TIMESTEP = 1

# we use the max repeat cycle of the modalities used, which is 8-day in the case of combined Landsat 8 + 9
NUM_TIMESTEPS = 8

# time range to sample a random time window from. End year is inclusive (START_YEAR <= N <= END_YEAR)
# if the season spans two years, the end year will be the following year
# (i.e., if the end year is 2019, it is possible to get data from early 2020)
# for the start year, we are limited by Sentinel-3 data availability (starting 2016-10-18)
# This means effectively, we can sample from (START_YEAR - 1)-12-16 to (END_YEAR + 1)-02-28
#START_YEAR = 2017
#END_YEAR = 2020

# Landsat 9 restricts to 2022 - 2023
START_YEAR = 2022
END_YEAR = 2023

EXPORTED_HEIGHT_WIDTH_METRES = 1000
# this is the maximum patch_size * num_patches.
# we will need to change this if that assumption changes
# Note: 96 for 1km x 1km and 48 for 500m x 500m
DATASET_OUTPUT_HW_HIGH_RES = 96
DATASET_OUTPUT_HW_MED_RES = 96
DATASET_OUTPUT_HW_LOW_RES = 96

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
    "landsat08": {
        "original_resolution": 30,
        "shape_type": "s_t_h_x",
        "active": True,
        "export": True,
    },
    "landsat09": {
        "original_resolution": 30,
        "shape_type": "s_t_h_x",
        "active": True,
        "export": True,
    },
    "s3": {
        "original_resolution": 300,
        "shape_type": "s_t_m_x",
        "active": True,
        "export": True,
    },
    "modis": {
        "original_resolution": 500,
        "shape_type": "s_t_l_x",
        "active": True,
        "export": True,
    },
    "viirs_fine": {
        "original_resolution": 500,
        "shape_type": "s_t_l_x",
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
        "shape_type": "s_t_l_x",
        "active": True,
        "export": False,
    },
    "ndvi": {
        "original_resolution": 500,
        "shape_type": "s_t_l_x",
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

# TODO: this is a bit hard-coded; empirically identified lower bound thresholds (inclusive) to avoid outliers in the input data
CHANNEL_WISE_INVALID_DATA_THRESHOLDS = {
    "s_t_h_x": {
        0: -50,  # S1 VV
        1: -50,  # S1 VH
        2: 0,  # S1 angle
        3: -2000,  # S2 B2
        4: -2000,  # S2 B3
        5: -2000,  # S2 B4
        6: -2000,  # S2 B8
        7: -2000,  # S2 B11
        8: -2000,  # S2 B12
        9: -2000,  # Landsat 8 B2
        10: -2000,  # Landsat 8 B3
        11: -2000,  # Landsat 8 B4
        12: -2000,  # Landsat 8 B5
        13: -2000,  # Landsat 8 B6
        14: -2000,  # Landsat 8 B7
        15: -2000,  # Landsat 9 B2
        16: -2000,  # Landsat 9 B3
        17: -2000,  # Landsat 9 B4
        18: -2000,  # Landsat 9 B5
        19: -2000,  # Landsat 9 B6
        20: -2000,  # Landsat 9 B7
    },
    "s_t_m_x": {
        0: -1000,  # S3
        1: -1000,  # S3
    },
    "s_t_l_x": {
        0: -100,  # MODIS
        1: -100,  # MODIS
        2: -100,  # MODIS
        3: -100,  # MODIS
        4: -100,  # MODIS
        5: -0.01,  # VIIRS
        6: -0.01,  # VIIRS
        7: -5,  # NDSI
    },
    "sp_x": {
        0: -10,  # SRTM elevation
        1: -10,  # SRTM slope
    },
    "t_x": {
        0: -0.01,  # VIIRS
        1: -0.01,  # VIIRS
        2: -0.01,  # VIIRS
        3: -0.01,  # VIIRS
        4: -10,  # ERA5
        5: -10,  # ERA5
        6: -10,  # ERA5
        7: -10,  # ERA5
        8: -10,  # ERA5
    },
    "st_x": {
        0: -1,  # x
        1: -1,  # y
        2: -1,  # z
    }
}
USE_INDECES = False

EE_PROJECT = "ee-marlena"
EE_BUCKET_TIFS = None
EE_DRIVE_FOLDER_NAME = "snow_ee_exports_20241226"
EE_DRIVE_FOLDER_ID = "1cL7tEHhC92UHmuwdEgH0ero6aXQqhBcb"
EE_FOLDER_TIFS = "tifs4"
EE_FOLDER_H5PYS = "h5pys_full"

DATA_FOLDER = Path(__file__).parents[2] / "data"
TIFS_FOLDER = DATA_FOLDER / "tifs_all_bands_1km"
NORMALIZATION_DICT_FILENAME = "normalizing_dict_1km.json"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
ENCODER_FILENAME = "encoder.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
TARGET_ENCODER_FILENAME = "target_encoder.pt"
DECODER_FILENAME = "decoder.pt"
CONFIG_FILENAME = "config.json"