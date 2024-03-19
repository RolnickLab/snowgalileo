import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch.utils.data import DataLoader

from ..utils import DEFAULT_SEED, device
from .knn import KNNat5, KNNat20, KNNat100

logger = logging.getLogger("__main__")


@dataclass
class Hyperparams:
    lr: float = 3e-4
    max_epochs: int = 100
    batch_size: int = 4096
    patience: int = 3
    num_workers: int = 2
    weight_decay: float = 0.05


class EvalTask(ABC):
    name: str
    num_outputs: int
    regression: bool
    multilabel: bool

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.name = f"{self.name}_{self.seed}"

    @classmethod
    def _construct_sklearn_model(cls, model) -> BaseEstimator:
        if cls.multilabel:
            model = MultiOutputClassifier(model, n_jobs=cls.num_outputs)
        return model

    @torch.no_grad()
    def train_sklearn_model(
        self,
        dl: DataLoader,
        pretrained_model,
        models: List[str] = ["Regression", "Random Forest"],
    ) -> Union[Sequence[BaseEstimator], Dict]:
        for model_mode in models:
            if self.regression:
                assert model_mode in ["Regression", "Random Forest"]
            else:
                assert model_mode in [
                    "Regression",
                    "Random Forest",
                    "KNNat5",
                    "KNNat20",
                    "KNNat100",
                ]
        pretrained_model.eval()

        encoding_list, target_list = [], []
        for masked_output, label in dl:
            d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]
            target_list.append(label.cpu().numpy())
            with torch.no_grad():
                encodings = pretrained_model.encoder(d_x, s_x, d_m, s_m, months).cpu().numpy()
                encoding_list.append(encodings)
        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)
        if len(targets.shape) == 2 and targets.shape[1] == 1:
            targets = targets.ravel()

        fit_models = []
        model_dict = {
            False: {
                "Regression": self._construct_sklearn_model(
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

    def evaluate_model_on_task(self, pretrained_model, model_modes: List[str]) -> Dict:
        raise NotImplementedError
