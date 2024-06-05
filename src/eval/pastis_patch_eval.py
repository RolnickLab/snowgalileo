from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch.multiprocessing
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torchmetrics import JaccardIndex
from tqdm import tqdm

from ..data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space_time,
)
from ..data.earthengine.s2 import S2_BANDS
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name

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
        average_s2_over_month: Optional[bool] = True,
    ):
        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)

        self.data_path = data_path

        self.metadata = gpd.read_file(data_dir / cast(str, self.data_path) / "metadata.geojson")
        self.metadata.index = self.metadata["ID_PATCH"].astype(int)
        self.metadata.sort_index(inplace=True)

        self.metadata = pd.concat([self.metadata[self.metadata["Fold"] == f] for f in folds])

        self.id = self.metadata.index

        # pastis comes in large images, we split them into subtiles
        # must be a square number
        self.num_subtiles_per_image = num_subtiles_per_image
        assert sqrt(cast(float, self.num_subtiles_per_image)).is_integer()

        self.average_s2_over_month = average_s2_over_month

        if average_s2_over_month:
            self.num_timesteps = 12
        else:
            # max number of timesteps in PASTIS
            self.num_timesteps = 61

    def create_pastis_masks(
        self, missing_timestep_indeces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Masks unavailable channels and timesteps.
        """
        s_t_channels = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key]

        # everything is masked by default
        s_t_m = np.ones([len(SPACE_TIME_BANDS_GROUPS_IDX)])
        # unmask available bands
        s_t_m[s_t_channels] = 0
        s_t_m = repeat(
            s_t_m,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # mask missing timesteps
        s_t_m[:, :, missing_timestep_indeces, :] = 1

        # no space only / time only channels are available
        s_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (s_m == 1).all()
        assert (t_m == 1).all()

        return (s_t_m, s_m, t_m)

    def average_over_month(
        self, s2: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the month-wise mean of an input image, pixel- and channel-specific.
        Months without observations are filled with zeros.
        Expected data input shape: T x C x H x W.
        Months are expected to be 0-indexed.
        """
        unique_months = np.unique(months)

        all_months = np.arange(self.num_timesteps)
        missing_timestep_indeces = np.where(~np.isin(all_months, unique_months))[0]

        # stack months and s2 indices to group by month
        s2_idx = np.arange(s2.shape[0])
        stacked_months_and_s2_idx = np.column_stack((months, s2_idx))

        # group observations by sorted month https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
        stacked_months_and_s2_idx = stacked_months_and_s2_idx[
            stacked_months_and_s2_idx[:, 0].argsort()
        ]
        s2_idx_per_month = np.split(
            stacked_months_and_s2_idx[:, 1],
            np.unique(stacked_months_and_s2_idx[:, 0], return_index=True)[1][1:],
        )

        averages_months_with_data = np.array([s2[idx].mean(axis=0) for idx in s2_idx_per_month])

        averages_all_months = np.zeros((self.num_timesteps, s2.shape[1], s2.shape[2], s2.shape[3]))

        # fill up with zeros if there are months without observations
        averages_all_months[unique_months] = averages_months_with_data

        return averages_all_months, all_months, missing_timestep_indeces

    def zero_pad_missing_timesteps(
        self, s2: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pads input image and months with zeros to reach the maximum number of timesteps available in PASTIS.
        """

        s2_all_months = np.zeros((self.num_timesteps, s2.shape[1], s2.shape[2], s2.shape[3]))
        s2_all_months[np.arange(s2.shape[0])] = s2

        all_months = np.zeros(self.num_timesteps)
        all_months[np.arange(months.shape[0])] = months
        missing_timestep_indeces = np.where(~np.isin(all_months, months))[0]

        return s2_all_months, all_months, missing_timestep_indeces

    def get_eo_array_and_masks(
        self, id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the image for a given ID, handles missing timesteps and normalizes the data.
        Also provides static and month data, and creates masks for missing data.
        """
        s2 = np.load(data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id)).astype(
            np.float32
        )

        dates = self.metadata["dates-S2"][id]
        # the dates are in the format YYYYMMDD
        months = (
            np.array([int(str(value)[4:6]) for _, value in dates.items()]) - 1
        )  # 0-indexed months
        assert all(0 <= month <= 11 for month in months)

        if self.average_s2_over_month:
            s2, months, missing_timestep_indeces = self.average_over_month(s2, months)

        # pad missing timesteps, will be masked out later
        else:
            s2, months, missing_timestep_indeces = self.zero_pad_missing_timesteps(s2, months)

        kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)]

        s_t_x = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(SPACE_TIME_BANDS),
            ]
        )
        s_t_x[:, :, :, kept_dynamic_bands] = repeat(s2, "t c h w -> h w t c")

        # space only / time only bands are not provided by pastis
        s_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        s_t_m, s_m, t_m = self.create_pastis_masks(
            missing_timestep_indeces=missing_timestep_indeces
        )

        return normalize_space_time(s_t_x), s_x, t_x, s_t_m, s_m, t_m, months

    def get_target(self, id: int) -> torch.Tensor:
        target = np.load(
            data_dir / cast(str, self.data_path) / "ANNOTATIONS/TARGET_{}.npy".format(id)
        )
        return torch.from_numpy(target[0].astype(int)).long()

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        """
        Slices and returns a subtile of the image and the corresponding target.
        """
        img_idx = idx // self.num_subtiles_per_image

        id = self.id[img_idx]

        s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = self.get_eo_array_and_masks(id)
        target = self.get_target(id)

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
                s_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_x,
                s_t_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                s_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_m,
                months,
            ),
            target[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ],
        )

    def __len__(self):
        return self.metadata.shape[0] * self.num_subtiles_per_image


class PastisPatchEval(EvalTask):
    name = "pastis_patch"
    regression = False
    multilabel = False
    segmentation = True
    num_outputs = len(PastisPatchDataset.labels_to_int)
    input_height_width = PastisPatchDataset.input_height_width

    def __init__(
        self,
        average_months: bool = True,
        num_subtiles_per_image: int = 4,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
    ):
        self.average_months = average_months
        self.num_subtiles_per_image = num_subtiles_per_image
        super().__init__(patch_size, seed)
        self.input_height_width = self.input_height_width // int(
            sqrt(cast(float, self.num_subtiles_per_image))
        )
        self.name = f"{self.name}_{'AVERAGED_MONTHS' if self.average_months else 'ALL_MONTHS'}_hw{self.input_height_width}"

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(self, model, sklearn_models: Optional[Sequence[BaseEstimator]]) -> Dict:
        test_dl = DataLoader(
            PastisPatchDataset(
                folds=[1],
                average_s2_over_month=self.average_months,
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        if sklearn_models is not None:
            results_dict = {}
            pred_dict: Dict[str, BaseEstimator] = {
                model_class_name(model): [] for model in sklearn_models
            }
            labels_list = []

            for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]

                labels_list.append(self.group_and_reduce_targets_per_token(label).cpu().numpy())

                model.eval()
                with torch.no_grad():
                    s_t_x, s_x, t_x, s_t_m, s_m, t_m, _ = model(
                        s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                    )
                    encodings = (
                        self.group_encodings_per_token(model, s_t_x, s_x, t_x, s_t_m, s_m, t_m)
                        .cpu()
                        .numpy()
                    )

                    for model in sklearn_models:
                        preds = model.predict(encodings)
                        pred_dict[model_class_name(model)].append(preds)

            for model_name_str, pred_list in pred_dict.items():
                results_dict.update(
                    self.compute_metrics(
                        model_name_str, np.concatenate(pred_list), np.concatenate(labels_list)
                    )
                )
            return results_dict

        # finetuning
        else:
            mean_IoU = []

            for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]
                label = label.to(device)

                jaccard_mean = JaccardIndex(task="multiclass", num_classes=self.num_outputs).to(
                    device
                )

                model.eval()
                with torch.no_grad():
                    preds = model(
                        s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                    )
                    mean_IoU.append(jaccard_mean(preds, label).item())

            return {f"{self.name}: finetuned_mean_iou": np.mean(mean_IoU)}

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert (
                model_mode in ["finetune"] or model_mode in self.all_classification_sklearn_models
            )

        train_dl = DataLoader(
            PastisPatchDataset(
                folds=[3, 4, 5],
                average_s2_over_month=self.average_months,
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )

        results_dict = {}

        if "finetune" in model_modes:
            val_dl = DataLoader(
                PastisPatchDataset(
                    folds=[2],
                    average_s2_over_month=self.average_months,
                    num_subtiles_per_image=self.num_subtiles_per_image,
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )

            finetuned_model = self.finetune_presto(train_dl, val_dl, pretrained_model)
            results_dict.update(self._evaluate_model(finetuned_model))

        if model_mode in self.all_classification_sklearn_models:
            trained_sklearn_models = self.train_sklearn_model(
                train_dl,
                pretrained_model,
                models=model_modes,
            )
            results_dict.update(self._evaluate_model(pretrained_model, trained_sklearn_models))

        return results_dict
