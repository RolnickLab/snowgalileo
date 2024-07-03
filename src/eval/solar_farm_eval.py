import json
from datetime import date, datetime
from pathlib import Path
from typing import List, NamedTuple, Optional

import geopandas as gpd
import pandas as pd
from shapely import Polygon, box
from tqdm import tqdm

from src.data.config import DATA_FOLDER

FOREST_LOSS_FOLDER = DATA_FOLDER / "forest_loss"
RAW_FOREST_LOSS_DATA = FOREST_LOSS_FOLDER / "raw"
LABELS_GEOJSON_PATH = FOREST_LOSS_FOLDER / "labels.geojson"
POOR_LABELS_GEOJSON_PATH = FOREST_LOSS_FOLDER / "poor_labels.geojson"
LOCAL_FOREST_TIFS_FOLDER = FOREST_LOSS_FOLDER / "tifs"
FOREST_CACHE_FOLDER = FOREST_LOSS_FOLDER / "npys"
GCLOUD_FOREST_TIFS_FOLDER = "forest_loss"
RAW_PROJECTION = "EPSG:3857"
LATLON_PROJECTION = "EPSG:4326"
END_PADDING = 90

LABEL_PATH = "label.json"
METADATA_PATH = "metadata.json"
DATE_STRING_FORMAT = "%Y-%m-%d"


class LabelInfo(NamedTuple):
    group: str
    name: str
    new_label: str
    old_label: str
    bounds: Polygon
    start_date: date
    end_date: date


class LabelInfoList(NamedTuple):
    group: list
    name: list
    new_label: list
    old_label: list
    bounds: list
    start_date: list
    end_date: list

    def update(self, new_info: LabelInfo) -> None:
        self.group.append(new_info.group)
        self.name.append(new_info.name)
        self.new_label.append(new_info.new_label)
        self.old_label.append(new_info.old_label)
        self.bounds.append(new_info.bounds)
        self.start_date.append(new_info.start_date)
        self.end_date.append(new_info.end_date)

    def to_geopandas(self) -> gpd.GeoDataFrame:
        data = gpd.GeoDataFrame(
            {
                "group": self.group,
                "name": self.name,
                "new_label": self.new_label,
                "old_label": self.old_label,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
            geometry=self.bounds,
            crs=RAW_PROJECTION,
        )
        data["start_date"] = pd.to_datetime(data["start_date"])
        data["end_date"] = pd.to_datetime(data["end_date"])
        return data


def folder_to_attributes(labels_path: Path, group: str) -> LabelInfo:
    assert (labels_path / LABEL_PATH).exists(), f"{labels_path} missing {LABEL_PATH}"
    assert (labels_path / METADATA_PATH).exists(), f"{labels_path} missing {METADATA_PATH}"

    with (labels_path / LABEL_PATH).open("r") as l_f:
        label = json.load(l_f)

    with (labels_path / METADATA_PATH).open("r") as m_f:
        metadata = json.load(m_f)
        assert metadata["projection"]["crs"] == RAW_PROJECTION
        start_date = datetime.strptime(
            metadata["time_range"][0].split("T")[0], DATE_STRING_FORMAT
        ).date()
        end_date = datetime.strptime(
            metadata["time_range"][1].split("T")[0], DATE_STRING_FORMAT
        ).date()

    x_res, y_res = metadata["projection"]["x_resolution"], metadata["projection"]["y_resolution"]
    xmin, ymin, xmax, ymax = metadata["bounds"]
    xmin, ymin, xmax, ymax = xmin * x_res, ymin * y_res, xmax * x_res, ymax * y_res
    return LabelInfo(
        # group overrides metadata["group"]
        group,
        metadata["name"],
        label["new_label"],
        label["old_label"],
        box(xmin, ymin, xmax, ymax),
        start_date,
        end_date,
    )


{
    "group": "default",
    "name": "5777408_8118272",
    "projection": {"crs": "EPSG:32621", "x_resolution": 10, "y_resolution": -10},
    "bounds": [60736, -64019, 61223, -63534],
    "time_range": ["2020-09-03T00:00:00+00:00", "2021-01-31T00:00:00+00:00"],
    "options": {},
}


def folders_to_geojson(
    groups: List[str] = LABELLED_GROUPS, check_against: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    output = LabelInfoList([], [], [], [], [], [], [])
    for group in groups:
        all_feature_files = list(RAW_FOREST_LOSS_DATA.glob(f"{group}/feat*"))
        print(f"Processing {len(all_feature_files)} files for group {group}")
        for i in tqdm(all_feature_files):
            info = folder_to_attributes(i, group)
            if info.new_label not in IGNORE_LABELS:
                if check_against is not None:
                    # check that the folder doesn't already exist
                    if info.name not in check_against["name"].unique():
                        output.update(info)
                    else:
                        print(f"{info.name} already in check_against GeoJSON")
                        continue
                else:
                    output.update(info)
    return output.to_geopandas()
