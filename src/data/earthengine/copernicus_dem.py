from datetime import date

import ee

image_collection = "COPERNICUS/DEM/GLO30"
DEM_BANDS = ["DEM", "slope", "aspect"]
DEM_SHIFT_VALUES = [float(0.0), float(0.0), float(0.0)]
DEM_DIV_VALUES = [float(1.0), float(1.0), float(1.0)]


def get_single_dem_image(region: ee.Geometry, start_date: date, end_date: date) -> ee.Image:
    elevation = ee.Image(image_collection).clip(region).select(DEM_BANDS[0])
    # Calculate slope. Units are degrees, range is [0,90]
    slope = ee.Terrain.slope(elevation)
    # Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W.
    aspect = ee.Terrain.aspect(elevation)
    return ee.Image.cat([elevation, slope, aspect])
