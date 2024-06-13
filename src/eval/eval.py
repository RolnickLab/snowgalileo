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
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..flexipresto import Encoder
from ..utils import DEFAULT_SEED, device
from .knn import KNNat5, KNNat20, KNNat100

logger = logging.getLogger("__main__")


@dataclass
class Hyperparams:
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 1
    patience: int = 10
    finetuning_lr: float = 3e-4


def model_class_name(model: BaseEstimator) -> str:
    if isinstance(model, MultiOutputClassifier):
        return model.estimator.__class__.__name__
    else:
        return model.__class__.__name__


class EvalTask(ABC):
    name: str = "EvalTask"
    regression: bool
    segmentation: bool
    multilabel: bool
    input_height_width: int

    all_regression_sklearn_models = ["Regression", "Random Forest"]
    all_classification_sklearn_models = [
        "Logistic Regression",
        "Random Forest",
        "KNNat5",
        "KNNat20",
        "KNNat100",
    ]

    def __init__(self, patch_size: int, seed: int = DEFAULT_SEED, num_outputs: int = 1):
        self.num_outputs = num_outputs
        self.seed = seed
        self.patch_size = patch_size
        self.name = f"{self.name}_s{self.seed}_ps{self.patch_size}_nout{self.num_outputs}"

    @classmethod
    def _construct_sklearn_model(cls, model, num_outputs=1) -> BaseEstimator:
        if cls.multilabel or (cls.segmentation and num_outputs > 1):
            model = MultiOutputClassifier(model, n_jobs=num_outputs)
        return model

    @torch.no_grad()
    def group_targets_per_token(self, target: torch.Tensor) -> torch.Tensor:
        # group labels per token for segmentation
        # grouped_label shape will be (batch_size, n_tokens, t_height * t_width)
        grouped_label = (
            target.reshape(
                target.shape[0],
                target.shape[1] // self.patch_size,
                self.patch_size,
                target.shape[2] // self.patch_size,
                self.patch_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(target.shape[0], -1, self.patch_size * self.patch_size)
        )
        return rearrange(grouped_label, "b n_t hw -> (b n_t) hw")

    @torch.no_grad()
    def group_encodings_per_token(self, model, s_t_x, s_x, t_x, s_t_m, s_m, t_m) -> np.ndarray:
        encodings = rearrange(
            model.apply_mask_and_average_tokens_per_patch(s_t_x, s_x, t_x, s_t_m, s_m, t_m),
            "b n_t n_f -> (b n_t) n_f",
        )
        return encodings

    def reduce_targets_per_token(self, grouped_label: np.ndarray) -> np.ndarray:
        if self.num_outputs == 1:
            # take the most common label per token
            label = mode(grouped_label, axis=1).mode

        # one hot encode the labels
        else:
            label = np.zeros((grouped_label.shape[0], self.num_outputs))

            for i in range(grouped_label.shape[0]):
                classes = np.unique(grouped_label[i])
                label[i][classes] = 1

            assert np.unique(label).shape[0] <= 2
        return label

    @torch.no_grad()
    def train_sklearn_model(
        self,
        train_dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
    ) -> Sequence[BaseEstimator]:
        """
        Fit sklearn models on the encodings of the pretrained model.
        For segmentation tasks, encodings and targets are grouped token-wise.
        Either the mode class will be taken or the classes will be one-hot encoded.
        This is controlled by the num_outputs attribute which can be changed in the subclass.
        """

        for model_mode in models:
            if self.regression:
                assert model_mode in self.all_regression_sklearn_models
            else:
                assert model_mode in self.all_classification_sklearn_models
        pretrained_model.eval()

        encodings_list, targets_list = [], []

        for masked_output, label in tqdm(train_dl, desc="Computing encodings for sklearn"):
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]

            if self.segmentation:
                targets = self.group_targets_per_token(label).cpu().numpy()

                if "pastis_patch" in self.name:
                    void_mask = np.any(targets == 19, axis=1)
                    targets = targets[~void_mask]

                targets_list.append(self.reduce_targets_per_token(targets))
            else:
                targets_list.append(label.cpu().numpy())

            with torch.no_grad():
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, _ = pretrained_model(
                    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                )
                if self.segmentation:
                    encodings = (
                        self.group_encodings_per_token(
                            pretrained_model, s_t_x, s_x, t_x, s_t_m, s_m, t_m
                        )
                        .cpu()
                        .numpy()
                    )

                    if "pastis_patch" in self.name:
                        encodings = encodings[~void_mask]

                    encodings_list.append(encodings)
                else:
                    encodings_list.append(
                        pretrained_model.average_tokens(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
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
                "KNNat5": self._construct_sklearn_model(KNNat5()),
                "KNNat20": self._construct_sklearn_model(KNNat20()),
                "KNNat100": self._construct_sklearn_model(KNNat100()),
            },
            True: {
                "Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=self.seed),
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
