import datetime
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pyproj
import pystac
import rasterio
from pystac import Asset, Catalog, Collection, Extent, Item, SpatialExtent, TemporalExtent

from src.data.local_sources.paths import LocalPaths

# Catalog the AOI-clipped archive (the single downstream root). Repointable to
# another region via the LOCAL_CLIPPED_ROOT env var — see data/BOW_VALLEY_DATA_LAYOUT.md.
# The clipped archive preserves the raw per-modality subdir + archive layout, so
# the directory walk below is unchanged.
_PATHS = LocalPaths()
base_dir = _PATHS.clipped_root
# Catalog output lives in the repo data/ tree (not inside the clipped archive,
# which is typically a symlink to external storage). Override with STAC_OUTPUT_DIR.
output_dir = Path(os.environ.get("STAC_OUTPUT_DIR", "data/stac_catalog"))

# Standard Sinusoidal projection transformer for MODIS/VIIRS
sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
wgs84 = pyproj.Proj("epsg:4326")
sinu_to_wgs84 = pyproj.Transformer.from_proj(sinu, wgs84, always_xy=True)

# Projected coordinate transformer for Landsat (UTM 12N) to WGS84
utm12 = pyproj.Proj("epsg:32612")
utm12_to_wgs84 = pyproj.Transformer.from_proj(utm12, wgs84, always_xy=True)

# Caches to avoid redundant file IO
landsat_bounds_cache: dict[str, list[float]] = {}
sentinel2_bounds_cache: dict[str, list[float]] = {}
modis_bounds_cache: dict[str, list[float]] = {}


def get_modis_bounds():
    # MODIS/VIIRS tile h10v03 sinusoidal bounds projected to WGS84
    if "h10v03" not in modis_bounds_cache:
        # upperLeft = [-8895604.157, 6671703.118], lowerRight = [-7783653.638, 5559752.598]
        lon_min, lat_min = sinu_to_wgs84.transform(-8895604.157, 5559752.598)
        lon_max, lat_max = sinu_to_wgs84.transform(-7783653.638, 6671703.118)
        modis_bounds_cache["h10v03"] = [lon_min, lat_min, lon_max, lat_max]
    return modis_bounds_cache["h10v03"]


def get_landsat_bounds(tar_path, path_row):
    if path_row not in landsat_bounds_cache:
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getnames()
                tif_members = [m for m in members if m.endswith(".TIF") or m.endswith(".tif")]
                if tif_members:
                    first_tif = tif_members[0]
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tar.extract(first_tif, path=tmpdir)
                        tif_file = Path(tmpdir) / first_tif
                        with rasterio.open(tif_file) as src:
                            bounds = src.bounds
                            lon_min, lat_min = utm12_to_wgs84.transform(bounds.left, bounds.bottom)
                            lon_max, lat_max = utm12_to_wgs84.transform(bounds.right, bounds.top)
                            landsat_bounds_cache[path_row] = [lon_min, lat_min, lon_max, lat_max]
        except Exception as e:
            print(f"Error reading Landsat bounds for {path_row}: {e}")
            # Fallback to general Rockies extent
            landsat_bounds_cache[path_row] = [-118.0, 49.0, -114.0, 52.0]
    return landsat_bounds_cache.get(path_row, [-118.0, 49.0, -114.0, 52.0])


def get_sentinel2_bounds(zip_path, tile_name):
    if tile_name not in sentinel2_bounds_cache:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                mtd_name = [
                    x for x in z.namelist() if "MTD_MSIL1C.xml" in x or "MTD_MSIL2A.xml" in x
                ][0]
                mtd_content = z.read(mtd_name).decode("utf-8")
                # Parse Global_Footprint or EXT_POS_LIST
                fp = re.findall(r"<EXT_POS_LIST[^>]*>(.*?)</EXT_POS_LIST>", mtd_content, re.DOTALL)
                if fp:
                    coords = [float(x) for x in fp[0].split()]
                    lats = coords[0::2]
                    lons = coords[1::2]
                    sentinel2_bounds_cache[tile_name] = [
                        min(lons),
                        min(lats),
                        max(lons),
                        max(lats),
                    ]
        except Exception as e:
            print(f"Error reading S2 bounds for {tile_name}: {e}")
            sentinel2_bounds_cache[tile_name] = [-118.0, 49.0, -114.0, 52.0]
    return sentinel2_bounds_cache.get(tile_name, [-118.0, 49.0, -114.0, 52.0])


def get_sentinel1_bounds(zip_path):
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            kml_name = [x for x in z.namelist() if "map-overlay.kml" in x][0]
            kml_content = z.read(kml_name).decode("utf-8")
            coords_str = re.findall(r"<coordinates>(.*?)</coordinates>", kml_content, re.DOTALL)
            if coords_str:
                pairs = coords_str[0].strip().split()
                lons = [float(p.split(",")[0]) for p in pairs]
                lats = [float(p.split(",")[1]) for p in pairs]
                return [min(lons), min(lats), max(lons), max(lats)]
    except Exception as e:
        print(f"Error reading S1 bounds: {e}")
    return [-118.0, 49.0, -114.0, 52.0]


def get_sentinel3_bounds(zip_path):
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            m_name = [x for x in z.namelist() if "xfdumanifest.xml" in x][0]
            m_content = z.read(m_name).decode("utf-8")
            pos_str = re.findall(r"<gml:posList[^>]*>(.*?)</gml:posList>", m_content, re.DOTALL)
            if pos_str:
                coords = [float(x) for x in pos_str[0].split()]
                lats = coords[0::2]
                lons = coords[1::2]
                return [min(lons), min(lats), max(lons), max(lats)]
    except Exception as e:
        print(f"Error reading S3 bounds: {e}")
    return [-132.0, 49.0, -107.0, 63.0]


def build_catalog():
    print(f"Initializing STAC Catalog at: {output_dir}")
    catalog = Catalog(
        id="bow-valley-selection-raw",
        description="STAC catalog for Bow Valley Selection raw geospatial datasets.",
    )

    # Define collection schemas
    collections_metadata = {
        "dem": {
            "title": "Copernicus DEM GLO-30",
            "description": "Copernicus Digital Elevation Model 30m / 10m reference tiles.",
            "spatial": [-117.0, 50.0, -116.0, 51.0],
            "temporal": [
                datetime.datetime(2011, 2, 6, tzinfo=datetime.timezone.utc),
                datetime.datetime(2014, 8, 27, tzinfo=datetime.timezone.utc),
            ],
        },
        "worldcover": {
            "title": "ESA WorldCover 10m 2021",
            "description": "ESA WorldCover land cover classification product.",
            "spatial": [-117.0, 48.0, -114.0, 51.0],
            "temporal": [
                datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2021, 12, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "era5": {
            "title": "ERA5-Land Meteorological Reanalysis",
            "description": "Daily aggregated ECMWF weather variables.",
            "spatial": [-120.0, 48.0, -114.0, 54.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 3, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "landsat8": {
            "title": "Landsat 8 L1TP Multi-Spectral Imagery",
            "description": "Landsat 8 Top of Atmosphere orthorectified multi-spectral scenes.",
            "spatial": [-118.0, 49.0, -114.0, 52.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "landsat9": {
            "title": "Landsat 9 L1TP Multi-Spectral Imagery",
            "description": "Landsat 9 Top of Atmosphere orthorectified multi-spectral scenes.",
            "spatial": [-118.0, 49.0, -114.0, 52.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "modis": {
            "title": "MODIS Terra Surface Reflectance Daily",
            "description": "MOD09GA daily surface reflectance products on sinusoidal grid.",
            "spatial": [-126.0, 49.0, -108.0, 60.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "sentinel1": {
            "title": "Sentinel-1 SAR IW GRD",
            "description": "Sentinel-1 Ground Range Detected C-band Synthetic Aperture Radar acquisitions.",
            "spatial": [-119.5, 49.8, -115.3, 51.8],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "sentinel2": {
            "title": "Sentinel-2 MultiSpectral Instrument L1C",
            "description": "Sentinel-2 orthorectified multi-temporal top-of-atmosphere reflectance.",
            "spatial": [-118.0, 49.0, -114.0, 52.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "sentinel3": {
            "title": "Sentinel-3 OLCI Level-1 EFR Swath",
            "description": "Sentinel-3 Ocean and Land Colour Instrument top-of-atmosphere full resolution swaths.",
            "spatial": [-132.0, 49.0, -107.0, 63.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
        "viirs": {
            "title": "VIIRS Daily Surface Reflectance",
            "description": "VNP09GA daily surface reflectance products on sinusoidal grid.",
            "spatial": [-126.0, 49.0, -108.0, 60.0],
            "temporal": [
                datetime.datetime(2025, 3, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2025, 5, 31, tzinfo=datetime.timezone.utc),
            ],
        },
    }

    stac_collections = {}
    for col_id, col_meta in collections_metadata.items():
        spatial_extent = SpatialExtent(bboxes=[col_meta["spatial"]])
        temporal_extent = TemporalExtent(intervals=[col_meta["temporal"]])
        extent = Extent(spatial=spatial_extent, temporal=temporal_extent)

        collection = Collection(
            id=col_id, title=col_meta["title"], description=col_meta["description"], extent=extent
        )
        catalog.add_child(collection)
        stac_collections[col_id] = collection

    # Walk all directories and parse items
    for col_id in collections_metadata.keys():
        col_dir = base_dir / col_id
        if not col_dir.exists():
            print(f"Subdirectory {col_dir} does not exist. Skipping.")
            continue

        print(f"Parsing collection: {col_id}...")
        collection = stac_collections[col_id]

        # S1, S2, S3, L8, L9, MODIS, VIIRS, ERA5, DEM, WorldCover
        if col_id == "dem":
            # Walk and find TIFs
            for root, dirs, files in os.walk(col_dir):
                for f in files:
                    if f.endswith("_DEM.tif"):
                        filepath = Path(root) / f
                        rel_path = filepath.relative_to(base_dir)
                        with rasterio.open(filepath) as src:
                            bounds = src.bounds

                        item_id = filepath.stem
                        item = Item(
                            id=item_id,
                            geometry=None,
                            bbox=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                            datetime=datetime.datetime(2014, 8, 27, tzinfo=datetime.timezone.utc),
                            properties={"crs": "EPSG:4326"},
                        )
                        item.add_asset(
                            key="data",
                            asset=Asset(href=str(rel_path), media_type=pystac.MediaType.GEOTIFF),
                        )
                        collection.add_item(item)

        elif col_id == "worldcover":
            for root, dirs, files in os.walk(col_dir):
                for f in files:
                    if f.endswith("_Map.tif"):
                        filepath = Path(root) / f
                        rel_path = filepath.relative_to(base_dir)
                        with rasterio.open(filepath) as src:
                            bounds = src.bounds

                        item_id = filepath.stem
                        item = Item(
                            id=item_id,
                            geometry=None,
                            bbox=[bounds.left, bounds.bottom, bounds.right, bounds.top],
                            datetime=datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
                            properties={"crs": "EPSG:4326"},
                        )
                        item.add_asset(
                            key="data",
                            asset=Asset(href=str(rel_path), media_type=pystac.MediaType.GEOTIFF),
                        )
                        collection.add_item(item)

        elif col_id == "era5":
            for root, dirs, files in os.walk(col_dir):
                for f in files:
                    if f.endswith(".nc"):
                        filepath = Path(root) / f
                        rel_path = filepath.relative_to(base_dir)

                        # Determine date from path or filename
                        date_match = re.search(r"(\d{6})", str(filepath))
                        dt = datetime.datetime(2025, 3, 15, tzinfo=datetime.timezone.utc)
                        if date_match:
                            year_month = date_match.group(1)
                            year = int(year_month[:4])
                            month = int(year_month[4:])
                            dt = datetime.datetime(year, month, 15, tzinfo=datetime.timezone.utc)

                        item_id = filepath.stem
                        item = Item(
                            id=item_id,
                            geometry=None,
                            bbox=[-120.0, 48.0, -114.0, 54.0],
                            datetime=dt,
                            properties={"format": "NetCDF"},
                        )
                        item.add_asset(
                            key="data",
                            asset=Asset(href=str(rel_path), media_type="application/x-netcdf"),
                        )
                        collection.add_item(item)

        elif col_id in ["landsat8", "landsat9"]:
            for f in os.listdir(col_dir):
                if f.endswith(".tar"):
                    filepath = col_dir / f
                    rel_path = filepath.relative_to(base_dir)

                    # Parse path/row and acquisition date from filename
                    # e.g., LC08_L1TP_042024_20250302_...
                    parts = f.split("_")
                    path_row = parts[2]
                    date_str = parts[3]
                    dt = datetime.datetime.strptime(date_str, "%Y%m%d").replace(
                        tzinfo=datetime.timezone.utc
                    )

                    bbox = get_landsat_bounds(filepath, path_row)
                    item_id = filepath.stem
                    item = Item(
                        id=item_id,
                        geometry=None,
                        bbox=bbox,
                        datetime=dt,
                        properties={"path_row": path_row, "sensor": "OLI_TIRS"},
                    )
                    item.add_asset(
                        key="archive",
                        asset=Asset(href=str(rel_path), media_type="application/x-tar"),
                    )
                    collection.add_item(item)

        elif col_id in ["modis", "viirs"]:
            for f in os.listdir(col_dir):
                if f.endswith(".hdf") or f.endswith(".h5"):
                    filepath = col_dir / f
                    rel_path = filepath.relative_to(base_dir)

                    # Parse date from Julian date format (AYYYYDDD)
                    parts = f.split(".")
                    julian_str = parts[1].replace("A", "")
                    dt = datetime.datetime.strptime(julian_str, "%Y%j").replace(
                        tzinfo=datetime.timezone.utc
                    )

                    bbox = get_modis_bounds()
                    item_id = filepath.stem
                    item = Item(
                        id=item_id,
                        geometry=None,
                        bbox=bbox,
                        datetime=dt,
                        properties={"tile_id": "h10v03"},
                    )
                    item.add_asset(
                        key="data", asset=Asset(href=str(rel_path), media_type="application/x-hdf")
                    )
                    collection.add_item(item)

        elif col_id == "sentinel1":
            for f in os.listdir(col_dir):
                if f.endswith(".zip"):
                    filepath = col_dir / f
                    rel_path = filepath.relative_to(base_dir)

                    # Parse datetime from filename (S1C_IW_GRDH_1SDV_20250330T013724...)
                    parts = f.split("_")
                    dt_str = parts[4]
                    dt = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S").replace(
                        tzinfo=datetime.timezone.utc
                    )

                    bbox = get_sentinel1_bounds(filepath)
                    item_id = filepath.stem
                    item = Item(
                        id=item_id,
                        geometry=None,
                        bbox=bbox,
                        datetime=dt,
                        properties={"sensor": "SAR-C", "polarization": "VV/VH"},
                    )
                    item.add_asset(
                        key="archive",
                        asset=Asset(href=str(rel_path), media_type="application/zip"),
                    )
                    collection.add_item(item)

        elif col_id == "sentinel2":
            for f in os.listdir(col_dir):
                if f.endswith(".zip"):
                    filepath = col_dir / f
                    rel_path = filepath.relative_to(base_dir)

                    # Parse tile and datetime
                    parts = f.split("_")
                    dt_str = parts[2]
                    tile_name = parts[5]
                    dt = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S").replace(
                        tzinfo=datetime.timezone.utc
                    )

                    bbox = get_sentinel2_bounds(filepath, tile_name)
                    item_id = filepath.stem
                    item = Item(
                        id=item_id,
                        geometry=None,
                        bbox=bbox,
                        datetime=dt,
                        properties={"tile": tile_name, "sensor": "MSI"},
                    )
                    item.add_asset(
                        key="archive",
                        asset=Asset(href=str(rel_path), media_type="application/zip"),
                    )
                    collection.add_item(item)

        elif col_id == "sentinel3":
            for f in os.listdir(col_dir):
                if f.endswith(".zip"):
                    filepath = col_dir / f
                    rel_path = filepath.relative_to(base_dir)

                    # Parse datetime
                    parts = f.split("_")
                    dt_str = parts[7]
                    dt = datetime.datetime.strptime(dt_str, "%Y%m%dT%H%M%S").replace(
                        tzinfo=datetime.timezone.utc
                    )

                    bbox = get_sentinel3_bounds(filepath)
                    item_id = filepath.stem
                    item = Item(
                        id=item_id,
                        geometry=None,
                        bbox=bbox,
                        datetime=dt,
                        properties={"sensor": "OLCI"},
                    )
                    item.add_asset(
                        key="archive",
                        asset=Asset(href=str(rel_path), media_type="application/zip"),
                    )
                    collection.add_item(item)

    # Save the catalog
    catalog.normalize_hrefs(str(output_dir))
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    print(f"STAC Catalog generated and saved successfully at: {output_dir}")


if __name__ == "__main__":
    build_catalog()
