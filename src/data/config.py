import os
from pathlib import Path
from typing import Any, Dict

try:  # Optional: load a repo-root .env so EE_PROJECT et al. can live outside VCS.
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # python-dotenv not installed — env vars still work.
    pass

DAYS_PER_TIMESTEP = 1

# we use the max repeat cycle of the modalities used, which is 8-day in the case of combined Landsat 8 + 9
NUM_TIMESTEPS = 8

# time range to sample a random time window from. End year is inclusive (START_YEAR <= N <= END_YEAR)
# if the season spans two years, the end year will be the following year
# (i.e., if the end year is 2019, it is possible to get data from early 2020)
# This means effectively, we can sample from (START_YEAR - 1)-12-16 to (END_YEAR + 1)-02-28
# Landsat 9 restricts to 2022 - 2023
START_YEAR = 2022
END_YEAR = 2023

# inclusive (i.e., the end date of a season is included in the season)
NORTH_HEM_SEASONS = {
    "early": ("10-01", "12-15"),
    "mid": ("12-16", "02-28"),
    "late": ("03-01", "06-30"),
}

SOUTH_HEM_SEASONS = {
    "early": ("04-01", "06-15"),
    "mid": ("06-16", "08-28"),
    "late": ("09-01", "12-30"),
}

NO_DATA_VALUE = -9999
MODIS_FILL_VALUE = -28672.0

# TODO: the naming here is confusing
EXPORTED_HEIGHT_WIDTH_METRES = 1000
DATASET_OUTPUT_HW_HIGH_RES = 100
DATASET_OUTPUT_HW_MED_RES = 200
DATASET_OUTPUT_HW_LOW_RES = 500

NUM_HIGH_RES_PIXELS_PER_DIM = EXPORTED_HEIGHT_WIDTH_METRES // DATASET_OUTPUT_HW_HIGH_RES
NUM_MED_RES_PIXELS_PER_DIM = EXPORTED_HEIGHT_WIDTH_METRES // DATASET_OUTPUT_HW_MED_RES
NUM_LOW_RES_PIXELS_PER_DIM = EXPORTED_HEIGHT_WIDTH_METRES // DATASET_OUTPUT_HW_LOW_RES

NDI_VALID_DATA_BOUNDS = (-1, 1)

# the idea is that for exporting different data, we will only have to change this dictionary in the end
# i.e., sort the modalities into different shape_types, add / remove satellite modalities
# for using / not using modalities, the "active" flag should get used
MODALITIES: Dict[str, Dict[str, Any]] = {
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
    "modis_cloud_flag": {
        "original_resolution": 1000,
        "shape_type": "clouds",
        "active": True,
        "export": True,
    },
    "s2_cloud_flag": {
        "original_resolution": 60,
        "shape_type": "clouds",
        "active": True,
        "export": True,
    },
    "landsat_cloud_flag": {
        "original_resolution": 30,
        "shape_type": "clouds",
        "active": True,
        "export": True,
    },
    "dem": {
        "original_resolution": 30,
        "shape_type": "sp_x",
        "active": True,
        "export": True,
    },
    "ee_wc": {
        "original_resolution": 10,
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
        "active": True,
        "export": False,
    },
}

# Empirically identified lower bound thresholds (inclusive) to avoid outliers and fill values in the input data
# Identified based on expected value ranges as stated by documentations, or by visual inspection of histograms of the data
CHANNEL_WISE_INVALID_DATA_THRESHOLDS: Dict[str, Dict] = {
    "s_t_h_x": {
        0: -50,  # S1 VV: expected min as specific by GEE band ranges
        1: -50,  # S1 VH: expected min as specific by GEE band ranges
        2: 0,  # S1 angle: expected min as specific by GEE band ranges
        3: -1,  # S2 B2: Sentinel-2 lower bound is 0 according to https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/#units
        4: -1,  # S2 B3
        5: -1,  # S2 B4
        6: -1,  # S2 B8
        7: -1,  # S2 B11
        8: -1,  # S2 B12
        9: 0.0000001,  # Landsat B2: Landsat Collection 2 uses 0 for no data
        10: 0.0000001,  # Landsat B3
        11: 0.0000001,  # Landsat B4
        12: 0.0000001,  # Landsat B5
        13: 0.0000001,  # Landsat B6
        14: 0.0000001,  # Landsat B7
    },
    "s_t_m_x": {
        0: -1,  # S3 (empirically identified)
        1: -1,  # S3 (empirically identified)
    },
    "s_t_l_x": {
        0: -100,  # MODIS: min as specific by GEE band ranges, masks out fill value -28672
        1: -100,  # MODIS
        2: -100,  # MODIS
        3: -100,  # MODIS
        4: -100,  # MODIS
        5: -100,  # MODIS
        6: -100,  # MODIS
        7: -0.01,  # VIIRS: min as specific by GEE band ranges, masks out fill value -28672
        8: -0.01,  # VIIRS
        9: -1,  # NDSI lower bound
        10: -1,  # NDVI lower bound
    },
    "sp_x": {
        0: 0.0000001,  # elevation: invalid values are set to zero according to https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM.html
        1: 0,  # slope
        2: 0,  # aspect
        3: 0,  # WC Map (one-hot encoded channels below)
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
    },
    "t_x": {
        0: -0.01,  # VIIRS: min as specific by GEE band ranges, masks out fill value -28672
        1: -0.01,  # VIIRS
        2: -0.01,  # VIIRS
        3: -0.01,  # VIIRS
        4: 184,  # ERA5 temperature in Kelvin: lower bound is min temperature recorded on Earth
        5: 184,  # ERA5 temperature in Kelvin
        6: -1,  # ERA5 precipitation
        7: -53,  # ERA5 wind component u https://confluence.ecmwf.int/display/CKB/ERA5%3A+large+10m+winds
        8: -53,  # ERA5 wind component v
    },
    "st_x": {
        0: -1,  # x
        1: -1,  # y
        2: -1,  # z
    },
}
USE_INDECES = False

# Earth Engine Cloud project. Overridable via the EE_PROJECT env var (or a
# repo-root .env) so each user can bill API calls against their own registered
# project without editing this file. Defaults to the original author's project.
EE_PROJECT = os.environ.get("EE_PROJECT", "ee-marlena")
EE_BUCKET_TIFS = None
EE_DRIVE_FOLDER_NAME = "snow_ee_exports_20260111"
EE_DRIVE_FOLDER_ID = "1cL7tEHhC92UHmuwdEgH0ero6aXQqhBcb"
EE_FOLDER_TIFS = "tifs4"
EE_FOLDER_H5PYS = "h5pys"

# TODO: this should be defined somewhere else
RESULTS_FOLDER = Path(__file__).parents[2] / "results"

# when in this repo, uncomment the following line
DATA_FOLDER = Path(__file__).parents[2] / "data"
TIFS_FOLDER = DATA_FOLDER / "tifs_all_bands"
NORMALIZATION_DICT_FILENAME = "normalizing_dict_december.json"
OUTPUT_FOLDER = DATA_FOLDER / "outputs"
ENCODER_FILENAME = "encoder"
OPTIMIZER_FILENAME = "optimizer"
DECODER_FILENAME = "decoder"
CONFIG_FILENAME = "config"
