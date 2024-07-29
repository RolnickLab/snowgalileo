import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import h5py
import numpy as np
import torch
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, default_collate
from torch.utils.data import Dataset as TorchDataset

from ..data.dataset import (
    LOCATION_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space,
    normalize_space_time,
    normalize_static,
    normalize_time,
    to_cartesian,
)
from ..flexipresto import Encoder
from ..masking import MASKING_MODES_COARSE, MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .cropharvest.bands import BANDS
from .cropharvest.columns import NullableColumns, RequiredColumns
from .cropharvest.datasets import CropHarvest, Task, TestInstance
from .cropharvest.datasets import CropHarvestLabels as OrgCropHarvestLabels
from .cropharvest.utils import NoDataForBoundingBoxError, memoized
from .eval import EvalTask, Hyperparams, model_class_name

logger = logging.getLogger("__main__")


cropharvest_data_dir = data_dir / "cropharvest_data"

CH_BANDS_TO_SPACE_TIME_BANDS = [BANDS.index(s) for s in SPACE_TIME_BANDS if s in BANDS]
SPACE_TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_TIME_BANDS) if s in BANDS]

CH_BANDS_TO_SPACE_BANDS = [BANDS.index(s) for s in SPACE_BANDS if s in BANDS]
SPACE_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(SPACE_BANDS) if s in BANDS]

CH_BANDS_TO_TIME_BANDS = [BANDS.index(s) for s in TIME_BANDS if s in BANDS]
TIME_BANDS_TO_CH_BANDS = [idx for idx, s in enumerate(TIME_BANDS) if s in BANDS]

LOCATION_BAND_MAPPING = [idx for idx, x in enumerate(STATIC_BANDS) if x in LOCATION_BANDS]


class CropHarvestLabels(OrgCropHarvestLabels):
    def construct_fao_classification_labels(
        self, task: Task, filter_test: bool = True
    ) -> List[Tuple[Path, int]]:
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

        return [(path, y) for path, y in zip(paths, ys) if path.exists()]


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

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        path, y = self.paths_and_y[index]
        satellite_data = h5py.File(path, "r")
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
    root = Path(root_name) if root_name != "" else cropharvest_data_dir
    if not root.exists():
        root.mkdir()
        CropHarvest(root, download=True)


class CropHarvestEvalBase(EvalTask):
    """
    Data is automatically downloaded by this class
    """

    start_month = 1
    num_timesteps: Optional[int] = None
    multilabel = False
    regression = False

    def __init__(
        self,
        name: str,
        patch_size: int,
        include_latlons: bool = True,
        seed: int = DEFAULT_SEED,
    ):
        self.include_latlons = include_latlons
        self.name = f"{name}{'_latlons' if include_latlons else ''}"
        super().__init__(patch_size, seed)

        output_channels = [0] * len(MASKING_MODES_COARSE)
        for i, val in enumerate(MASKING_MODES_COARSE):
            if val == "static":  # should this be static or space?
                output_channels[i] = 1
        self.condition = {"output_channels": torch.Tensor(output_channels).to(device)}

    @staticmethod
    def truncate_timesteps(x, num_timesteps: Optional[int]):
        if (num_timesteps is None) or (x is None):
            return x
        else:
            return x[:, :num_timesteps]

    def cropharvest_array_to_normalized_presto(
        self,
        array: np.ndarray,
        latlons: np.ndarray,
        start_month: int,
        timesteps: Optional[int] = None,
    ):
        array = self.truncate_timesteps(array, timesteps)
        b, t, _ = array.shape

        s_t_x = np.zeros((b, t, len(SPACE_TIME_BANDS)))
        s_t_x[:, :, SPACE_TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_TIME_BANDS]
        s_t_x = repeat(s_t_x, "b t d -> b h w t d", h=1, w=1)
        s_t_m = np.ones((b, 1, 1, t, len(SPACE_TIME_BANDS)))
        s_t_m[:, :, :, :, SPACE_TIME_BANDS_TO_CH_BANDS] = 0
        s_t_m = s_t_m[:, :, :, :, [g[0] for _, g in SPACE_TIME_BANDS_GROUPS_IDX.items()]]

        sp_x = np.zeros((b, t, len(SPACE_BANDS)))
        sp_x[:, :, SPACE_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_SPACE_BANDS]
        sp_x = repeat(sp_x[:, 0], "b d -> b h w d", h=1, w=1)
        sp_m = np.ones((b, 1, 1, len(SPACE_BANDS)))
        sp_m[:, :, :, SPACE_BANDS_TO_CH_BANDS] = 0
        sp_m = sp_m[:, :, :, [g[0] for _, g in SPACE_BAND_GROUPS_IDX.items()]]

        t_x = np.zeros((b, t, len(TIME_BANDS)))
        t_x[:, :, TIME_BANDS_TO_CH_BANDS] = array[:, :, CH_BANDS_TO_TIME_BANDS]
        t_m = np.ones((b, t, len(TIME_BANDS)))
        t_m[:, :, TIME_BANDS_TO_CH_BANDS] = 0
        t_m = t_m[:, :, [g[0] for _, g in TIME_BAND_GROUPS_IDX.items()]]

        st_x = np.zeros((b, len(STATIC_BANDS)))
        st_m = np.ones((b, len(STATIC_BAND_GROUPS_IDX)))
        if self.include_latlons:
            st_m[:, list(STATIC_BAND_GROUPS_IDX).index("location")] = 0
            st_x[:, LOCATION_BAND_MAPPING] = to_cartesian(latlons[:, 0], latlons[:, 1])

        months = np.fmod(np.arange(start_month - 1, start_month - 1 + t), 12)
        months = repeat(months, "t -> b t", b=b)

        return masked_output_np_to_tensor(
            normalize_space_time(s_t_x),
            normalize_space(sp_x),
            normalize_time(t_x),
            normalize_static(st_x),
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
        )

    @staticmethod
    def collate_fn(batch):
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, label = default_collate(batch)
        return MaskedOutput(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months), label


class BinaryCropHarvestEval(CropHarvestEvalBase):
    num_outputs = 1

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
        include_latlons: bool = True,
        do_condition: bool = False,
    ):
        suffix = f"_{sample_size}" if sample_size else ""
        suffix = f"{suffix}_{num_timesteps}" if num_timesteps is not None else suffix
        super().__init__(
            name=f"CropHarvest_{country}{suffix}",
            include_latlons=include_latlons,
            patch_size=1,
            seed=seed,
        )

        download_cropharvest_data()

        evaluation_datasets = get_eval_datasets()
        evaluation_datasets = [d for d in evaluation_datasets if country in d.id]
        assert len(evaluation_datasets) == 1
        self.dataset: CropHarvest = evaluation_datasets[0]
        assert self.dataset.task.normalize is False
        self.num_timesteps = num_timesteps
        self.sample_size = sample_size
        self.do_condition = do_condition

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_model: BaseEstimator, c_i: Optional[Dict]
    ) -> Dict:
        pretrained_model.eval()
        with tempfile.TemporaryDirectory() as results_dir:
            for test_id, test_instance in self.dataset.test_data(max_size=10000):
                savepath = Path(results_dir) / f"{test_id}.nc"

                latlons = np.stack((test_instance.lats, test_instance.lons), axis=-1)
                masked_output = self.cropharvest_array_to_normalized_presto(
                    cast(np.ndarray, test_instance.x),
                    latlons,
                    self.start_month,
                    self.num_timesteps,
                )
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                    t.to(device) for t in masked_output
                ]
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x=s_t_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_m=s_t_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    patch_size=self.patch_size,
                    c_i=c_i,
                )
                encodings = (
                    pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                    .cpu()
                    .numpy()
                )
                preds = sklearn_model.predict_proba(encodings)[:, 1]
                ds = test_instance.to_xarray(preds)
                ds.to_netcdf(savepath)

            all_nc_files = list(Path(results_dir).glob("*.nc"))
            combined_instance, combined_preds = TestInstance.load_from_nc(all_nc_files)
            combined_results = combined_instance.evaluate_predictions(combined_preds)

        prefix = sklearn_model.__class__.__name__
        return {f"{self.name}: {prefix}_{key}": val for key, val in combined_results.items()}

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        array, latlons, labels = self.dataset.as_array()
        train_dl = DataLoader(
            TensorDataset(
                *self.cropharvest_array_to_normalized_presto(
                    array,
                    latlons,
                    timesteps=self.num_timesteps,
                    start_month=self.start_month,
                ),
                torch.from_numpy(labels),
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, None
        )
        results_dict = {}
        for sklearn_model in trained_sklearn_models:
            results_dict.update(self._evaluate_model(pretrained_model, sklearn_model, None))
        if self.do_condition:
            return results_dict

        conditioned_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, self.condition
        )
        conditioned_results_dict = {}
        for sklearn_model in conditioned_sklearn_models:
            conditioned_results_dict.update(
                self._evaluate_model(pretrained_model, sklearn_model, None)
            )
        conditioned_results_dict = {
            f"{key}_c": value for key, value in conditioned_results_dict.items()
        }

        return {**results_dict, **conditioned_results_dict}


class MultiClassCropHarvestEval(CropHarvestEvalBase):
    num_outputs = 10

    def __init__(
        self,
        test_ratio: float = 0.2,
        n_per_class: Optional[int] = 100,
        seed: int = DEFAULT_SEED,
        include_latlons: bool = True,
        do_condition: bool = False,
    ):
        self.do_condition = do_condition
        name_suffix = f"_{n_per_class}" if n_per_class is not None else ""
        super().__init__(
            name=f"CropHarvest_multiclass_global{name_suffix}_{seed}",
            patch_size=1,
            seed=seed,
            include_latlons=include_latlons,
        )

        download_cropharvest_data()
        task = Task(normalize=False)
        paths_and_y = CropHarvestLabels(cropharvest_data_dir).construct_fao_classification_labels(
            task, filter_test=True
        )

        y = [x[1] for x in paths_and_y]
        unique_ys = np.unique(y)
        y_string_to_int = {val: idx for idx, val in enumerate(np.unique(y))}

        train_paths_and_y, val_paths_and_y = train_test_split(
            paths_and_y, test_size=test_ratio, stratify=y, random_state=42
        )

        if n_per_class is not None:
            indices_to_keep = []
            y_train = np.array([x[1] for x in train_paths_and_y])
            for y_val in unique_ys:
                y_val_indices = np.where(y_train == y_val)[0]
                indices_to_keep.append(y_val_indices[:n_per_class])
            train_paths_and_y = [train_paths_and_y[i] for i in np.concatenate(indices_to_keep)]
            assert len(train_paths_and_y) <= n_per_class * len(unique_ys)

        array, latlons, labels = MultiClassCropHarvest(
            train_paths_and_y, y_string_to_int
        ).as_array()
        self.train_dataset = TensorDataset(
            *self.cropharvest_array_to_normalized_presto(
                array, latlons, self.start_month, timesteps=self.num_timesteps
            ),
            torch.from_numpy(labels),
        )
        self.eval_dataset = MultiClassCropHarvest(val_paths_and_y, y_string_to_int)

    @torch.no_grad()
    def _evaluate_models(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator], c_i
    ) -> Dict:
        dl = DataLoader(
            self.eval_dataset,
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        test_true = []
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }
        for x, latlons, y in dl:
            masked_output = self.cropharvest_array_to_normalized_presto(
                x, latlons, self.start_month, self.num_timesteps
            )
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                s_t_x=s_t_x,
                sp_x=sp_x,
                t_x=t_x,
                st_x=st_x,
                s_t_m=s_t_m,
                sp_m=sp_m,
                t_m=t_m,
                st_m=st_m,
                months=months,
                c_i=c_i,
                patch_size=self.patch_size,
            )
            encodings = (
                pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                .cpu()
                .numpy()
            )
            for model in sklearn_models:
                pred_dict[model_class_name(model)].append(model.predict(encodings))
            test_true.append(y)

        test_true_np = np.concatenate(test_true)
        results_dict = {}

        for model_name_str, pred_list in pred_dict.items():
            test_preds_np = np.concatenate(pred_list, axis=0)
            prefix = f"{model_name_str}"
            results_dict.update(
                {
                    f"{self.name}: {prefix}_num_samples": len(test_true_np),
                    f"{self.name}: {prefix}_f1_score": f1_score(
                        test_true_np, test_preds_np, average="weighted"
                    ),
                }
            )
        return results_dict

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        train_dl = DataLoader(
            self.train_dataset,
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
            collate_fn=self.collate_fn,
        )
        unconditioned_trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, None
        )
        unconditioned_results = self._evaluate_models(
            pretrained_model, unconditioned_trained_sklearn_models, None
        )
        if not self.do_condition:
            return unconditioned_results

        conditioned_trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, self.condition
        )
        conditioned_results = self._evaluate_models(
            pretrained_model, conditioned_trained_sklearn_models, self.condition
        )
        conditioned_results = {f"{key}_c": value for key, value in conditioned_results.items()}
        return {**conditioned_results, **unconditioned_results}
