import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from cropharvest.bands import BANDS
from cropharvest.columns import NullableColumns, RequiredColumns
from cropharvest.datasets import CropHarvest, Task
from cropharvest.datasets import CropHarvestLabels as OrgCropHarvestLabels
from cropharvest.utils import NoDataForBoundingBoxError, memoized
from einops import repeat
from torch.utils.data import Dataset as TorchDataset

from ..data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space,
    normalize_space_time,
    normalize_time,
)
from ..utils import DEFAULT_SEED, data_dir
from .eval import EvalTask

logger = logging.getLogger("__main__")


cropharvest_data_dir = data_dir / "cropharvest_data"

CH_BANDS_TO_SPACE_TIME_BANDS = [BANDS.index(s) for s in SPACE_TIME_BANDS]
SPACE_TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_TIME_BANDS) if s in BANDS]

CH_BANDS_TO_SPACE_BANDS = [BANDS.index(s) for s in SPACE_BANDS if s in BANDS]
SPACE_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_BANDS) if s in BANDS]

CH_BANDS_TO_TIME_BANDS = [BANDS.index(s) for s in TIME_BANDS if s in BANDS]
TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(TIME_BANDS) if s in BANDS]


class CropHarvestLabels(OrgCropHarvestLabels):
    def construct_fao_classification_labels(
        self, task: Task, filter_test: bool = True
    ) -> List[Tuple[Tuple[Path, Path], int]]:
        gpdf = self.as_geojson()
        if filter_test:
            gpdf = gpdf[gpdf[RequiredColumns.IS_TEST] == False]  # noqa
        if task.bounding_box is not None:
            gpdf = self.filter_geojson(
                gpdf, task.bounding_box, task.include_externally_contributed_labels
            )

        # This should probably be a required column since it has no
        # None values (and shouldn't have any)
        gpdf = gpdf[~gpdf[NullableColumns.CLASSIFICATION_LABEL].isnull()]

        if len(gpdf) == 0:
            raise NoDataForBoundingBoxError

        ys = gpdf[NullableColumns.CLASSIFICATION_LABEL]
        paths = self._dataframe_to_paths(gpdf)

        return [(path, y) for path, y in zip(paths, ys) if (path[0].exists() and path[1].exists())]


class MultiClassCropHarvest(TorchDataset):
    def __init__(
        self,
        paths_and_y: List[Tuple[Tuple[Path, Path], str]],
        y_string_to_int: Dict[str, int],
    ):
        self.paths_and_y = paths_and_y
        self.y_string_to_int = y_string_to_int

    def __len__(self) -> int:
        return len(self.paths_and_y)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        paths, y = self.paths_and_y[index]
        satellite_data = h5py.File(paths[0], "r")
        lat = satellite_data.attrs["instance_lat"]
        lon = satellite_data.attrs["instance_lon"]
        return (
            satellite_data.get("array")[:],
            np.array([lat, lon]),
            self.y_string_to_int[y],
        )

    def as_array(
        self, flatten_x: bool = False, num_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if num_samples is not None:
            raise NotImplementedError
        indices_to_sample = list(range(len(self)))
        X, latlons, Y = zip(*[self[i] for i in indices_to_sample])
        X_np, latlon_np, y_np = np.stack(X), np.stack(latlons), np.stack(Y)

        if flatten_x:
            X_np = self._flatten_array(X_np)
        return X_np, latlon_np, y_np

    @staticmethod
    def _flatten_array(array: np.ndarray) -> np.ndarray:
        return array.reshape(array.shape[0], -1)


@memoized
def get_eval_datasets():
    return CropHarvest.create_benchmark_datasets(
        root=cropharvest_data_dir, balance_negative_crops=False, normalize=False
    )


def download_cropharvest_data(root_name: str = ""):
    root = Path(root_name) if root_name != "" else cropharvest_data_dir()
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)


class CropHarvestEval(EvalTask):
    regression = False
    multilabel = False
    num_outputs = 1
    start_month = 1
    num_timesteps = None

    country_to_sizes: Dict[str, List] = {
        "Kenya": [20, 32, 64, 96, 128, 160, 192, 224, 256, None],
        "Togo": [20, 50, 126, 254, 382, 508, 636, 764, 892, 1020, 1148, None],
    }

    def __init__(
        self,
        country: str,
        num_timesteps: Optional[int] = None,
        sample_size: Optional[int] = None,
        seed: int = DEFAULT_SEED,
    ):
        download_cropharvest_data()

        evaluation_datasets = get_eval_datasets()
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset = evaluation_datasets[0]
        assert self.dataset.task.normalize is False
        self.num_timesteps = num_timesteps
        self.sample_size = sample_size

        suffix = f"_{sample_size}" if sample_size else ""
        suffix = f"{suffix}_{num_timesteps}" if num_timesteps is not None else suffix

        self.name = f"CropHarvest_{country}{suffix}"
        super().__init__(patch_size=1, seed=seed)

    def truncate_timesteps(self, x):
        if (self.num_timesteps is None) or (x is None):
            return x
        else:
            return x[:, : self.num_timesteps]

    @staticmethod
    def cropharvest_array_to_normalized_presto(array: np.ndarray):
        b, t, _ = array.shape

        s_t_x = np.zeros((b, t, len(SPACE_TIME_BANDS)))
        s_t_x[:, :, SPACE_TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_TIME_BANDS]
        s_t_x = repeat(s_t_x, "b t d -> b h w t d", h=1, w=1)
        s_t_m = np.ones((b, 1, 1, t, len(SPACE_TIME_BANDS)))
        s_t_m[:, :, :, :, SPACE_TIME_BANDS_TO_CH_BANDS] = 0
        s_t_m = s_t_m[:, :, :, :, [g[0] for _, g in SPACE_TIME_BANDS_GROUPS_IDX.items()]]

        s_x = np.zeros((b, t, len(SPACE_BANDS)))
        s_x[:, :, SPACE_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_BANDS]
        s_x = repeat(s_x[:, 0], "b d -> b h w d", h=1, w=1)
        s_m = np.ones((b, 1, 1, len(SPACE_BANDS)))
        s_m[:, :, :, SPACE_BANDS_TO_CH_BANDS] = 0
        s_m = s_m[:, :, :, [g[0] for _, g in SPACE_BAND_GROUPS_IDX.items()]]

        t_x = np.zeros((b, t, len(TIME_BANDS)))
        t_x[:, :, TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_TIME_BANDS]
        t_m = np.ones((b, t, len(TIME_BANDS)))
        t_m[:, :, TIME_BANDS_TO_CH_BANDS] = 0
        t_m = t_m[:, :, [g[0] for _, g in TIME_BAND_GROUPS_IDX.items()]]

        return (
            normalize_space_time(s_t_x),
            normalize_space(s_x),
            normalize_time(t_x),
            s_t_m,
            s_m,
            t_m,
        )
