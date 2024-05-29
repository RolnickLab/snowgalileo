import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
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
    batch_size: int = 1000
    num_workers: int = 4


def model_class_name(model: BaseEstimator) -> str:
    if isinstance(model, MultiOutputClassifier):
        return model.estimator.__class__.__name__
    else:
        return model.__class__.__name__


class EvalTask(ABC):
    name: str
    num_outputs: int
    regression: bool
    multilabel: bool

    all_regression_sklearn_models = ["Regression", "Random Forest"]
    all_classification_sklearn_models = [
        "Logistic Regression",
        "Random Forest",
        "KNNat5",
        "KNNat20",
        "KNNat100",
    ]

    def __init__(self, patch_size: int, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.patch_size = patch_size
        self.name = f"{self.name}_s{self.seed}_ps{self.patch_size}"

    @classmethod
    def _construct_sklearn_model(cls, model) -> BaseEstimator:
        if cls.multilabel:
            model = MultiOutputClassifier(model, n_jobs=cls.num_outputs)
        return model

    @torch.no_grad()
    def train_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model: Encoder,
        models: List[str] = ["Random Forest"],
    ) -> Sequence[BaseEstimator]:
        for model_mode in models:
            if self.regression:
                assert model_mode in self.all_regression_sklearn_models
            else:
                assert model_mode in self.all_classification_sklearn_models
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for masked_output, label in tqdm(dl, desc="Computing encodings for sklearn"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]
            target_list.append(label.cpu().numpy())
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
                    pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                    .cpu()
                    .numpy()
                )
                encoding_list.append(encodings)

        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            # from [[0], [0], [1]] to [0, 0, 1]
            targets = targets.ravel()

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
            fit_models.append(clone(model_dict[self.regression][model]).fit(encodings_np, targets))
        return fit_models

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        raise NotImplementedError
