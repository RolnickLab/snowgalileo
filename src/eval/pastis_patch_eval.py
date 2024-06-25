from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch.multiprocessing
from einops import repeat
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

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
    normalize_space_time,
    to_cartesian,
)
from ..data.earthengine.eo import S1_BANDS, S2_BANDS
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name
from .knn import (
    KNNat5Classifier,
    KNNat5Regressor,
    KNNat20Classifier,
    KNNat20Regressor,
    KNNat100Classifier,
    KNNat100Regressor,
)

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")


class PastisPatchDataset(PyTorchDataset):
    labels_to_int = {
        "Background": 0,
        "Meadow": 1,
        "SoftWinterWheat": 2,
        "Corn": 3,
        "WinterBarley": 4,
        "WinterRapeseed": 5,
        "SpringBarley": 6,
        "Sunflower": 7,
        "Grapevine": 8,
        "Beet": 9,
        "WinterTriticale": 10,
        "WinterDurumWheat": 11,
        "FruitsVegetablesFlowers": 12,
        "Potatoes": 13,
        "LeguminousFodder": 14,
        "Soybeans": 15,
        "Orchard": 16,
        "MixedCereal": 17,
        "Sorghum": 18,
        "VoidLabel": 19,
    }

    input_height_width = 128

    def __init__(
        self,
        folds: List[int],
        data_path: Optional[str] = "pastis/PASTIS-R",
        num_subtiles_per_image: Optional[int] = 4,
        band_mode: str = "combined",
        include_latlons: bool = True,
    ):
        assert band_mode in ["s2", "s1", "combined"]

        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)

        self.data_path = data_path
        self.include_latlons = include_latlons
        self.band_mode = band_mode

        self.metadata = gpd.read_file(data_dir / cast(str, self.data_path) / "metadata.geojson")
        self.metadata.index = self.metadata["ID_PATCH"].astype(int)
        self.metadata.sort_index(inplace=True)

        self.metadata = pd.concat([self.metadata[self.metadata["Fold"] == f] for f in folds])
        self.metadata = self.metadata.to_crs(epsg=4326)

        self.id = self.metadata.index

        # pastis comes in large images, we split them into subtiles
        # must be a square number
        self.num_subtiles_per_image = num_subtiles_per_image
        assert sqrt(cast(float, self.num_subtiles_per_image)).is_integer()

        self.num_timesteps = 12

    def create_pastis_masks(
        self,
        missing_timestep_indeces_s2=None,
        missing_timestep_indeces_s1=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Masks unavailable channels and timesteps.
        """

        # everything is masked by default
        s_t_m = np.ones([len(SPACE_TIME_BANDS_GROUPS_IDX)])

        if self.band_mode in ["combined", "s2"]:
            s_t_channels_s2 = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
            ]
            # unmask available bands
            s_t_m[s_t_channels_s2] = 0

        if self.band_mode in ["combined", "s1"]:
            s_t_channels_s1 = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S1" in key
            ]
            s_t_m[s_t_channels_s1] = 0

        s_t_m = repeat(
            s_t_m,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # mask missing timesteps
        if self.band_mode in ["combined", "s2"]:
            s_t_m[:, :, :, s_t_channels_s2][:, :, missing_timestep_indeces_s2, :] = 1

        if self.band_mode in ["combined", "s1"]:
            s_t_m[:, :, :, s_t_channels_s1][:, :, missing_timestep_indeces_s1, :] = 1

        # no space only / time only channels are available
        sp_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])
        st_m = np.ones([len(STATIC_BAND_GROUPS_IDX)])
        if self.include_latlons:
            location_channels = [
                idx for idx, key in enumerate(STATIC_BAND_GROUPS_IDX) if "location" in key
            ]
            st_m[location_channels] = 0
            assert ((st_m == 0) | (st_m == 1)).all()
        else:
            assert (st_m == 1).all()

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (sp_m == 1).all()
        assert (t_m == 1).all()

        return (s_t_m, sp_m, t_m, st_m)

    def average_over_month(
        self, observations: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the month-wise mean of an input image, pixel- and channel-specific.
        Months without observations are filled with zeros.
        Expected data input shape: T x C x H x W.
        Months are expected to be 0-indexed.
        """
        unique_months = np.unique(months)

        all_months = np.arange(self.num_timesteps)
        missing_timestep_indeces = np.where(~np.isin(all_months, unique_months))[0]

        # stack months and observation indices to group by month
        observation_idx = np.arange(observations.shape[0])
        stacked_months_and_observations_idx = np.column_stack((months, observation_idx))

        # group observations by sorted month https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
        stacked_months_and_observations_idx = stacked_months_and_observations_idx[
            stacked_months_and_observations_idx[:, 0].argsort()
        ]
        observation_idx_per_month = np.split(
            stacked_months_and_observations_idx[:, 1],
            np.unique(stacked_months_and_observations_idx[:, 0], return_index=True)[1][1:],
        )

        averages_months_with_data = np.array(
            [observations[idx].mean(axis=0) for idx in observation_idx_per_month]
        )

        averages_all_months = np.zeros(
            (
                self.num_timesteps,
                observations.shape[1],
                observations.shape[2],
                observations.shape[3],
            )
        )

        # fill up with zeros if there are months without observations
        averages_all_months[unique_months] = averages_months_with_data

        return averages_all_months, missing_timestep_indeces

    def get_eo_array_masks_and_targets(
        self, id: int
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        torch.Tensor,
    ]:
        """
        Loads the image for a given ID, handles missing timesteps and normalizes the data.
        Also provides static and month data, and creates masks for missing data.
        """
        s_t_x = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(SPACE_TIME_BANDS),
            ]
        )

        geometry = self.metadata.geometry[id]
        lon, lat = geometry.centroid.x, geometry.centroid.y

        kept_static_bands = [idx for idx, x in enumerate(STATIC_BANDS) if x in LOCATION_BANDS]

        missing_timestep_indeces_s1 = None
        missing_timestep_indeces_s2 = None

        all_months = np.arange(self.num_timesteps)

        if self.band_mode in ["combined", "s2"]:
            s2 = np.load(
                data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id)
            ).astype(np.float32)
            dates_s2 = self.metadata["dates-S2"][id]

            # the dates are in the format YYYYMMDD
            months_s2 = (
                np.array([int(str(value)[4:6]) for _, value in dates_s2.items()]) - 1
            )  # 0-indexed months
            assert all(0 <= month <= 11 for month in months_s2)

            s2, missing_timestep_indeces_s2 = self.average_over_month(s2, months_s2)

            kept_dynamic_bands_s2 = [
                idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)
            ]
            s_t_x[:, :, :, kept_dynamic_bands_s2] = repeat(s2, "t c h w -> h w t c")

        if self.band_mode in ["combined", "s1"]:
            # s1 ascending
            s1_a = np.load(
                data_dir / cast(str, self.data_path) / "DATA_S1A/S1A_{}.npy".format(id)
            ).astype(np.float32)
            # s1 descending
            s1_d = np.load(
                data_dir / cast(str, self.data_path) / "DATA_S1D/S1D_{}.npy".format(id)
            ).astype(np.float32)
            s1 = np.concatenate([s1_a, s1_d], axis=0)

            dates_s1_a = self.metadata["dates-S1A"][id]
            dates_s1_d = self.metadata["dates-S1D"][id]

            dates_s1 = dict(
                zip(
                    range(len(dates_s1_a) + len(dates_s1_d)),
                    list(dates_s1_a.values()) + list(dates_s1_d.values()),
                )
            )

            months_s1 = (
                np.array([int(str(value)[4:6]) for _, value in dates_s1.items()]) - 1
            )  # 0-indexed months
            assert all(0 <= month <= 11 for month in months_s1)

            s1, missing_timestep_indeces_s1 = self.average_over_month(s1, months_s1)

            # PASTIS s1 includes channels VV, VH, and VV/VH ratio, we only want VV and VH
            s1 = s1[:, :1, :, :]

            kept_dynamic_bands_s1 = [
                idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S1_BANDS)
            ]
            s_t_x[:, :, :, kept_dynamic_bands_s1] = repeat(s1, "t c h w -> h w t c")

        s_t_m, sp_m, t_m, st_m = self.create_pastis_masks(
            missing_timestep_indeces_s2=missing_timestep_indeces_s2,
            missing_timestep_indeces_s1=missing_timestep_indeces_s1,
        )

        # space only / time only bands are not provided by pastis
        sp_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        st_x = np.zeros((len(STATIC_BANDS)))
        st_x[kept_static_bands] = to_cartesian(lat, lon)

        targets = np.load(
            data_dir / cast(str, self.data_path) / "ANNOTATIONS/TARGET_{}.npy".format(id)
        )
        targets = torch.from_numpy(targets[0].astype(int)).long()

        return (
            normalize_space_time(s_t_x),
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            all_months,
            targets,
        )

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        """
        Slices and returns a subtile of the image and the corresponding target.
        """
        img_idx = idx // self.num_subtiles_per_image

        id = self.id[img_idx]

        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            s_t_m,
            sp_m,
            t_m,
            st_m,
            months,
            targets,
        ) = self.get_eo_array_masks_and_targets(id)

        subtiles_per_dim = int(sqrt(cast(float, self.num_subtiles_per_image)))
        h, w = s_t_x.shape[:2]
        assert h == w  # this is the case for PASTIS
        assert h % subtiles_per_dim == 0
        pixels_per_dim = h // subtiles_per_dim
        subtile_idx = idx % self.num_subtiles_per_image

        row_idx = subtile_idx // subtiles_per_dim
        col_idx = subtile_idx % subtiles_per_dim

        return (
            masked_output_np_to_tensor(
                s_t_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                sp_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_x,
                st_x,
                s_t_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                sp_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_m,
                st_m,
                months,
            ),
            targets[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ],
        )

    def __len__(self):
        return self.metadata.shape[0] * self.num_subtiles_per_image


class PastisPatchEval(EvalTask):
    name = "pastis_patch"
    multilabel = False
    regression = False
    spatial_token_prediction = True
    input_height_width = PastisPatchDataset.input_height_width
    num_outputs = len(PastisPatchDataset.labels_to_int) - 1

    all_regression_sklearn_models = [
        "Random Forest",
        "KNNat5 Regressor",
        "KNNat20 Regressor",
        "KNNat100 Regressor",
    ]
    all_classification_sklearn_models = [
        "Random Forest",
        "KNNat5 Classifier",
        "KNNat20 Classifier",
        "KNNat100 Classifier",
    ]

    def __init__(
        self,
        num_subtiles_per_image: int = 4,
        band_mode: str = "combined",
        include_latlons: bool = True,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
        output_mode: str = "norm_counts",
    ):
        assert output_mode in ["mode", "norm_counts"]
        assert band_mode in ["s2", "s1", "combined"]

        self.output_mode = output_mode
        if self.output_mode == "norm_counts":
            self.regression = True
        self.num_subtiles_per_image = num_subtiles_per_image
        self.band_mode = band_mode
        super().__init__(patch_size=patch_size, seed=seed, output_mode=self.output_mode)
        self.input_height_width = self.input_height_width // int(
            sqrt(cast(float, self.num_subtiles_per_image))
        )
        self.include_latlons = include_latlons
        self.name = f"{self.name}_{self.band_mode}_{self.output_mode}{'_latlons' if include_latlons else ''}_{self.input_height_width}"

    @torch.no_grad()
    def train_sklearn_model(
        self,
        train_dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
    ) -> Sequence[BaseEstimator]:
        """
        Patch Pastis specific training of sklearn models.
        This includes spatial token wise predictions and the removal of void labels.
        """

        for model_mode in models:
            # normalized counts are in range [0, 1], so we use regression models
            if self.output_mode == "norm_counts":
                assert model_mode in self.all_regression_sklearn_models
            # mode output mode predicts classes, so we use classification models
            else:
                assert model_mode in self.all_classification_sklearn_models
        pretrained_model.eval()

        encodings_list, targets_list = [], []

        for masked_output, label in tqdm(train_dl, desc="Computing encodings for sklearn"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

            targets = self.group_targets_per_token(label).cpu().numpy()

            void_mask = np.any(targets == 19, axis=1)  # 19 is the void class

            targets_list.append(self.reduce_targets_per_token(targets[~void_mask]))

            with torch.no_grad():
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )
                encodings = (
                    self.group_encodings_per_token(
                        pretrained_model, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
                    )
                    .cpu()
                    .numpy()
                )

                encodings_list.append(encodings[~void_mask])

        targets_np = np.concatenate(targets_list)
        encodings_np = np.concatenate(encodings_list)

        if len(targets_np.shape) == 2 and targets_np.shape[1] == 1:
            # from [[0], [0], [1]] to [0, 0, 1]
            targets_np = targets_np.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Random Forest": self._construct_sklearn_model(
                    RandomForestClassifier(class_weight="balanced", random_state=self.seed)
                ),
                "KNNat5 Classifier": self._construct_sklearn_model(KNNat5Classifier()),
                "KNNat20 Classifier": self._construct_sklearn_model(KNNat20Classifier()),
                "KNNat100 Classifier": self._construct_sklearn_model(KNNat100Classifier()),
            },
            True: {
                "Random Forest": RandomForestRegressor(random_state=self.seed),
                "KNNat5 Regressor": self._construct_sklearn_model(KNNat5Regressor()),
                "KNNat20 Regressor": self._construct_sklearn_model(KNNat20Regressor()),
                "KNNat100 Regressor": self._construct_sklearn_model(KNNat100Regressor()),
            },
        }
        for model in models:
            fit_models.append(
                clone(model_dict[self.regression][model]).fit(encodings_np, targets_np)
            )
        return fit_models

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        if self.output_mode == "mode":
            return {
                f"{self.name}: {model_name}_overall_accuracy": accuracy_score(target, preds),
                f"{self.name}: {model_name}_mean_accuracy": balanced_accuracy_score(target, preds),
            }
        else:
            # regression metrics
            return {
                f"{self.name}_{model_name}_rmse": mean_squared_error(target, preds, squared=False),
                f"{self.name}_{model_name}_r2": r2_score(target, preds),
            }
        return {}

    @torch.no_grad()
    def _evaluate_model(self, pretrained_model, sklearn_models: Sequence[BaseEstimator]) -> Dict:
        test_dl = DataLoader(
            PastisPatchDataset(
                folds=[1],
                num_subtiles_per_image=self.num_subtiles_per_image,
                band_mode=self.band_mode,
                include_latlons=self.include_latlons,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        results_dict = {}
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        encodings_list = []
        targets_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

            targets = self.group_targets_per_token(label).cpu().numpy()
            void_mask = np.any(targets == 19, axis=1)
            targets_list.append(self.reduce_targets_per_token(targets[~void_mask]))

            pretrained_model.eval()
            with torch.no_grad():
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )

                encodings = (
                    self.group_encodings_per_token(
                        pretrained_model, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
                    )
                    .cpu()
                    .numpy()
                )
                encodings_list.append(encodings[~void_mask])

        encodings_np, targets_np = np.concatenate(encodings_list), np.concatenate(targets_list)

        for model in sklearn_models:
            preds = model.predict(encodings_np)
            pred_dict[model_class_name(model)].append(preds)

        for model_name_str, pred_list in pred_dict.items():
            results_dict.update(
                self.compute_metrics(
                    model_name_str,
                    np.concatenate(pred_list),
                    targets_np,
                )
            )
        return results_dict

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if self.output_mode == "norm_counts":
            if model_modes is None:
                model_modes = self.all_regression_sklearn_models
            for model_mode in model_modes:
                assert model_mode in self.all_regression_sklearn_models
        else:
            if model_modes is None:
                model_modes = self.all_classification_sklearn_models
            for model_mode in model_modes:
                assert model_mode in self.all_classification_sklearn_models

        train_dl = DataLoader(
            PastisPatchDataset(
                folds=[2, 3, 4, 5],
                num_subtiles_per_image=self.num_subtiles_per_image,
                band_mode=self.band_mode,
                include_latlons=self.include_latlons,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )

        results_dict = {}

        trained_sklearn_models = self.train_sklearn_model(
            train_dl,
            pretrained_model,
            models=model_modes,
        )
        results_dict.update(self._evaluate_model(pretrained_model, trained_sklearn_models))

        return results_dict
