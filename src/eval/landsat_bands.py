S1_BANDS = [
    "VV",
    "VH",
    "angle",
]

S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
]

LANDSAT_BANDS = [
    "B2_landsat",
    "B3_landsat",
    "B4_landsat",
    "B5_landsat",
    "B6_landsat",
    "B7_landsat",
]

NDVI_BANDS = ["NDVI"]

S3_BANDS = ["Oa17_radiance", "Oa21_radiance"]

MODIS_BANDS = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_b03",
    "sur_refl_b04",
    "sur_refl_b05",
    "sur_refl_b06",
    "sur_refl_b07",
]

VIIRS_FINE_BANDS = ["I1", "I3"]
VIIRS_COARSE_BANDS = ["M5", "M7", "M10", "M11"]

ERA5_BANDS = [
    "skin_temperature",
    "temperature_2m",
    "total_precipitation_sum",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
]

MODIS_CLOUD_FLAG_BANDS = ["state_1km"]
VIIRS_CLOUD_FLAG_BANDS = ["QF1"]
S2_CLOUD_FLAG_BANDS = ["QA60"]
LANDSAT_CLOUD_FLAG_BANDS = ["QA_PIXEL"]
DEM_BANDS = ["DEM", "slope", "aspect"]
WC_BANDS = ["Map"]

LOCATION_BANDS = ["x", "y", "z"]

LANDSAT_SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + LANDSAT_BANDS + NDVI_BANDS
SPACE_TIME_BANDS_MED_RES = S3_BANDS
SPACE_TIME_BANDS_LOW_RES = MODIS_BANDS + VIIRS_FINE_BANDS
LANDSAT_SPACE_BANDS = DEM_BANDS + WC_BANDS
LANDSAT_TIME_BANDS = VIIRS_COARSE_BANDS + ERA5_BANDS
LANDSAT_STATIC_BANDS = LOCATION_BANDS
LANDSAT_CLOUD_FLAG_BANDS = (
    MODIS_CLOUD_FLAG_BANDS
    + VIIRS_CLOUD_FLAG_BANDS
    + S2_CLOUD_FLAG_BANDS
    + LANDSAT_CLOUD_FLAG_BANDS
)

LANDSAT_BANDS = (
    LANDSAT_SPACE_TIME_BANDS
    + SPACE_TIME_BANDS_MED_RES
    + SPACE_TIME_BANDS_LOW_RES
    + LANDSAT_SPACE_BANDS
    + LANDSAT_TIME_BANDS
    + LANDSAT_STATIC_BANDS
)
