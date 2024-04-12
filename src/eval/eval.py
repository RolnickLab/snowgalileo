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
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..flexipresto import Encoder, FinetuningHead, PrestoFineTuningModel
from ..utils import DEFAULT_SEED, device
from .knn import KNNat5, KNNat20, KNNat100

logger = logging.getLogger("__main__")


@dataclass
class Hyperparams:
    batch_size: int = 8
    num_workers: int = 0
    max_epochs: int = 1
    patience: int = 10
    finetuning_lr: float = 3e-4


def model_class_name(model: BaseEstimator) -> str:
    if isinstance(model, MultiOutputClassifier):
        return model.estimator.__class__.__name__
    else:
        return model.__class__.__name__


class EvalTask(ABC):
    name: str
    num_outputs: int
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

    def __init__(self, patch_size: int, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.patch_size = patch_size
        self.name = f"{self.name}_s{self.seed}_ps{self.patch_size}"

    @classmethod
    def _construct_sklearn_model(cls, model) -> BaseEstimator:
        if cls.multilabel:
            model = MultiOutputClassifier(model, n_jobs=cls.num_outputs)
        return model

    def _construct_finetuning_model(
        self,
        pretrained_model: Encoder,
    ) -> PrestoFineTuningModel:
        head = FinetuningHead(
            regression=self.regression,
            segmentation=self.segmentation,
            input_height_width=self.input_height_width,
            num_outputs=self.num_outputs,
        )
        model = PrestoFineTuningModel(pretrained_model, head)
        model.train()
        return model

    @torch.no_grad()
    def train_sklearn_model(
        self,
        train_dl: DataLoader,
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
        for masked_output, label in tqdm(train_dl, desc="Computing encodings for sklearn"):
            d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]
            target_list.append(label.cpu().numpy())
            with torch.no_grad():
                d_x, s_x, d_m, s_m, _ = pretrained_model(
                    d_x, s_x, d_m, s_m, months, patch_size=self.patch_size
                )
                encodings = pretrained_model.average_tokens(d_x, s_x, d_m, s_m).cpu().numpy()
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

    def finetune_presto(
        self, train_dl: DataLoader, val_dl: DataLoader, pretrained_model: Encoder
    ) -> PrestoFineTuningModel:
        model = self._construct_finetuning_model(pretrained_model)

        optimizer = AdamW(model.parameters(), lr=Hyperparams.finetuning_lr)

        # TODO: change when binary tasks are added
        train_loss_fn = val_loss_fn = nn.CrossEntropyLoss(reduction="mean")

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        for _ in tqdm(range(Hyperparams.max_epochs), desc="Finetuning"):
            model.train()
            epoch_train_loss = 0.0
            num_updates = 0
            for masked_output, label in tqdm(
                train_dl, desc="Training model for batch", leave=False
            ):
                d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]
                y = label.to(device)

                optimizer.zero_grad()
                preds = model(d_x, s_x, d_m, s_m, months, patch_size=self.patch_size)

                loss = train_loss_fn(preds, y.long())
                epoch_train_loss += loss.item()
                num_updates += 1
                loss.backward()
                optimizer.step()

            train_loss.append(epoch_train_loss / num_updates)

            model.eval()
            all_preds, all_y = [], []

            for masked_output, label in tqdm(
                val_dl, desc="Validating model for batch", leave=False
            ):
                d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]
                y = label.to(device)

                with torch.no_grad():
                    preds = model(d_x, s_x, d_m, s_m, months, patch_size=self.patch_size)
                    all_preds.append(preds)
                    all_y.append(y)

            val_loss.append(val_loss_fn(torch.cat(all_preds).cpu(), torch.cat(all_y).cpu().long()))
            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = model.state_dict()
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = model.state_dict()
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= Hyperparams.patience:
                        logger.info("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        return model

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        raise NotImplementedError
