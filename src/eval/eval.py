import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from einops import rearrange
from scipy.stats import mode
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..flexipresto import Encoder
from ..utils import DEFAULT_SEED, device
from .knn import (
    KNNat5Classifier,
    KNNat5Regressor,
    KNNat20Classifier,
    KNNat20Regressor,
    KNNat100Classifier,
    KNNat100Regressor,
)

logger = logging.getLogger("__main__")


@dataclass
class Hyperparams:
    batch_size: int = 16
    num_workers: int = 0


def model_class_name(model: BaseEstimator) -> str:
    if isinstance(model, MultiOutputClassifier):
        return model.estimator.__class__.__name__
    else:
        return model.__class__.__name__


class EvalTask(ABC):
    name: str = "EvalTask"
    regression: bool = True
    spatial_token_prediction: bool = True
    multilabel: bool = False
    input_height_width: int = 100
    num_outputs: int = 1

    all_regression_sklearn_models = [
        "Regression",
        "Random Forest",
        "KNNat5 Regressor",
        "KNNat20 Regressor",
        "KNNat100 Regressor",
    ]
    all_classification_sklearn_models = [
        "Logistic Regression",
        "Random Forest",
        "KNNat5 Classifier",
        "KNNat20 Classifier",
        "KNNat100 Classifier",
    ]

    def __init__(
        self, patch_size_high_res: int, seed: int = DEFAULT_SEED, output_mode: Optional[str] = None
    ):
        self.output_mode = output_mode
        self.seed = seed
        self.patch_size_high_res = patch_size_high_res
        self.name = f"{self.name}_s{self.seed}_ps{self.patch_size_high_res}"

    def _construct_sklearn_model(self, model) -> BaseEstimator:
        if self.multilabel:
            model = MultiOutputClassifier(model, n_jobs=self.num_outputs)
        if self.output_mode == "norm_counts":
            model = MultiOutputRegressor(model, n_jobs=self.num_outputs)
        return model

    @torch.no_grad()
    def rearrange_targets_into_token_sequence(self, target: torch.Tensor) -> torch.Tensor:
        # group labels per token for segmentation
        return rearrange(
            target,
            "b (h p1) (w p2) -> (b h w p1 p2)",
            p1=self.patch_size_high_res,
            p2=self.patch_size_high_res,
        )
    
    @torch.no_grad()
    def rearrange_token_sequence_into_images(self, sequence: torch.Tensor) -> torch.Tensor:
        # rearrange token sequence into images
        return rearrange(
            sequence,
            "(b h w p1 p2) -> b (h p1) (w p2)",
            p1=self.patch_size_high_res,
            p2=self.patch_size_high_res,
        )

    @torch.no_grad()
    def group_encodings_per_token(
        self,
        model,
        s_t_h_x,
        s_t_m_x,
        s_t_l_x,
        sp_x,
        t_x,
        st_x,
        s_t_h_m,
        s_t_m_m,
        s_t_l_m,
        sp_m,
        t_m,
        st_m,
    ) -> torch.Tensor:
        encodings = rearrange(
            model.apply_mask_and_average_tokens_per_patch(
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ),
            "b n_t n_f -> (b n_t) n_f",
        )
        return encodings
    
    @torch.no_grad()
    def group_encodings_per_token_galileo_baseline(
        self,
        model,
        s_t_x,
        sp_x,
        t_x,
        st_x,
        s_t_m,
        sp_m,
        t_m,
        st_m,
    ) -> torch.Tensor:
        encodings = rearrange(
            model.apply_mask_and_average_tokens_per_patch(
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
            ),
            "b n_t n_f -> (b n_t) n_f",
        )
        return encodings


    def reduce_targets_per_token(self, grouped_label: np.ndarray) -> np.ndarray:
        if self.output_mode == "mode":
            # take the most common label per token
            label = mode(grouped_label, axis=1).mode

        # get normalized counts of each class per token
        else:
            label = np.zeros((grouped_label.shape[0], self.num_outputs))

            for i in range(grouped_label.shape[0]):
                class_counts = np.bincount(grouped_label[i], minlength=self.num_outputs)
                norm_counts = class_counts / np.sum(class_counts)
                label[i] = norm_counts

        return label

    @torch.no_grad()
    def train_sklearn_model(
        self,
        train_dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
        baseline_galileo: bool = False,
    ) -> Sequence[BaseEstimator]:
        """
        Fit sklearn models on the encodings of the pretrained model.
        For spatial token prediction tasks, encodings and targets are grouped token-wise.
        Either the mode class will be computed or the normalized counts of each class per token.
        This is controlled by the output_mode attribute which can be changed in the subclass.
        """

        if baseline_galileo:
            return self.train_sklearn_model_galileo_baseline(train_dl, pretrained_model, models)

        for model_mode in models:
            if self.regression:
                assert model_mode in self.all_regression_sklearn_models
            else:
                assert model_mode in self.all_classification_sklearn_models

        pretrained_model.eval()

        encodings_list, targets_list = [], []

        for masked_output, label, _ in tqdm(train_dl, desc="Computing encodings for sklearn"):
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = [t.to(device) for t in masked_output]

            if self.spatial_token_prediction:
                targets = self.rearrange_targets_into_token_sequence(label).cpu().numpy()

                # targets_list.append(self.reduce_targets_per_token(targets))
                targets_list.append(targets)
            else:
                targets_list.append(label.cpu().numpy())

            with torch.no_grad():
                (
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_h_m,
                    s_t_m_m,
                    s_t_l_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                ) = pretrained_model(
                    s_t_h_x=s_t_h_x,
                    s_t_m_x=s_t_m_x,
                    s_t_l_x=s_t_l_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_h_m=s_t_h_m,
                    s_t_m_m=s_t_m_m,
                    s_t_l_m=s_t_l_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    patch_size_high_res=self.patch_size_high_res,
                    patch_size_med_res=1,
                    patch_size_low_res=1,
                )
                if self.spatial_token_prediction:
                    encodings = (
                        self.group_encodings_per_token(
                            pretrained_model,
                            s_t_h_x,
                            s_t_m_x,
                            s_t_l_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_h_m,
                            s_t_m_m,
                            s_t_l_m,
                            sp_m,
                            t_m,
                            st_m,
                        )
                        .cpu()
                        .numpy()
                    )

                    encodings_list.append(encodings)
                else:
                    encodings_list.append(
                        pretrained_model.average_tokens(
                            s_t_h_x,
                            s_t_m_x,
                            s_t_l_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_h_m,
                            s_t_m_m,
                            s_t_l_m,
                            sp_m,
                            t_m,
                            st_m,
                        )
                        .cpu()
                        .numpy()
                    )

        targets_np = np.concatenate(targets_list)
        encodings_np = np.concatenate(encodings_list)

        if len(targets_np.shape) == 2 and targets_np.shape[1] == 1:
            # from [[0], [0], [1]] to [0, 0, 1]
            targets_np = targets_np.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Logistic Regression": self._construct_sklearn_model(
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=self.seed
                    )
                ),
                "Random Forest": self._construct_sklearn_model(
                    RandomForestClassifier(class_weight="balanced", random_state=self.seed)
                ),
                "KNNat5 Classifier": self._construct_sklearn_model(KNNat5Classifier()),
                "KNNat20 Classifier": self._construct_sklearn_model(KNNat20Classifier()),
                "KNNat100 Classifier": self._construct_sklearn_model(KNNat100Classifier()),
            },
            True: {
                "Regression": LinearRegression(),
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
    
    @torch.no_grad()
    def train_sklearn_model_galileo_baseline(
        self,
        train_dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
    ) -> Sequence[BaseEstimator]:
        """
        Fit sklearn models on the encodings of the pretrained model.
        For spatial token prediction tasks, encodings and targets are grouped token-wise.
        Either the mode class will be computed or the normalized counts of each class per token.
        This is controlled by the output_mode attribute which can be changed in the subclass.
        """

        for model_mode in models:
            if self.regression:
                assert model_mode in self.all_regression_sklearn_models
            else:
                assert model_mode in self.all_classification_sklearn_models

        pretrained_model.eval()

        encodings_list, targets_list = [], []

        for masked_output, label, _ in tqdm(train_dl, desc="Computing encodings for sklearn"):
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
            ) = [t.to(device) for t in masked_output]

            if self.spatial_token_prediction:
                targets = self.rearrange_targets_into_token_sequence(label).cpu().numpy()

                # targets_list.append(self.reduce_targets_per_token(targets))
                targets_list.append(targets)
            else:
                targets_list.append(label.cpu().numpy())

            with torch.no_grad():
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
                ) = pretrained_model(
                    s_t_x=s_t_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_m=s_t_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    patch_size=self.patch_size_high_res,
                )
                if self.spatial_token_prediction:
                    encodings = (
                        self.group_encodings_per_token_galileo_baseline(
                            pretrained_model,
                            s_t_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_m,
                            sp_m,
                            t_m,
                            st_m,
                        )
                        .cpu()
                        .numpy()
                    )

                    encodings_list.append(encodings)
                else:
                    encodings_list.append(
                        pretrained_model.average_tokens(
                            s_t_x,
                            sp_x,
                            t_x,
                            st_x,
                            s_t_m,
                            sp_m,
                            t_m,
                            st_m,
                        )
                        .cpu()
                        .numpy()
                    )

        targets_np = np.concatenate(targets_list)
        encodings_np = np.concatenate(encodings_list)

        if len(targets_np.shape) == 2 and targets_np.shape[1] == 1:
            # from [[0], [0], [1]] to [0, 0, 1]
            targets_np = targets_np.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Logistic Regression": self._construct_sklearn_model(
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=self.seed
                    )
                ),
                "Random Forest": self._construct_sklearn_model(
                    RandomForestClassifier(class_weight="balanced", random_state=self.seed)
                ),
                "KNNat5 Classifier": self._construct_sklearn_model(KNNat5Classifier()),
                "KNNat20 Classifier": self._construct_sklearn_model(KNNat20Classifier()),
                "KNNat100 Classifier": self._construct_sklearn_model(KNNat100Classifier()),
            },
            True: {
                "Regression": LinearRegression(),
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

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        raise NotImplementedError
