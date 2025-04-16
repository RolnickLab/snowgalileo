import ee

image_collection = "COPERNICUS/DEM/GLO30"
DEM_BANDS = ["DEM"]
DEM_SHIFT_VALUES = [float(0.0)] * len(DEM_BANDS)
DEM_DIV_VALUES = [float(1.0)] * len(DEM_BANDS)


def get_single_dem_image(region: ee.Geometry) -> ee.Image:
    dem = ee.Image(image_collection).clip(region).select(DEM_BANDS[0])
    return dem
