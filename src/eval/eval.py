import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from einops import rearrange
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..flexipresto import Encoder, FinetuningHead, PrestoFineTuningModel
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
    def _construct_sklearn_model(cls, model, num_outputs=1) -> BaseEstimator:
        if cls.multilabel or (cls.segmentation and num_outputs > 1):
            model = MultiOutputClassifier(model, n_jobs=num_outputs)
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
            label = grouped_label.mode(dim=1).values
            print("Label shape after taking the mode: " + str(label.shape))

        # one hot encode the labels
        else:
            label = torch.zeros(grouped_label.shape[0], self.num_outputs)

            for i in range(grouped_label.shape[0]):
                classes = torch.unique(grouped_label)
                label[i][classes] = 1

            assert torch.unique(label).shape[0] == 2
            print("Label shape after one-hot encoding: " + str(label.shape))
        return label

    def remove_void_class(self, targets_np: np.ndarray, encodings_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove tokens labeled with the void class. Code 19 is the void class.
        """
        # incoming shape is (nr_tokens, nr_pixels_per_token)
        mask = torch.any(targets_np == 19, dim=1)
        targets_np = targets_np[~mask]
        encodings_np = targets_np[~mask]

        return targets_np, encodings_np

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
                targets_list.append(self.group_targets_per_token(label).cpu().numpy())
            else:
                targets_list.append(label.cpu().numpy())

            with torch.no_grad():
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, _ = pretrained_model(
                    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                )
                if self.segmentation:
                    encodings_list.append(
                        self.group_encodings_per_token(
                            pretrained_model, s_t_x, s_x, t_x, s_t_m, s_m, t_m
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    encodings_list.append(
                        pretrained_model.average_tokens(s_t_x, s_x, t_x, s_t_m, s_m, t_m)
                        .cpu()
                        .numpy()
                    )

        targets_np = np.concatenate(targets_list)
        encodings_np = np.concatenate(encodings_list)

        print("Targets_np shape before removing void: " + str(targets_np.shape))
        print("Encodings_np shape before removing void: " + str(encodings_np.shape))

        # move to somewhere else ?
        if self.name == "pastis_patch":
            targets_np, encodings_np = self.remove_void_class(targets_np, encodings_np)

        print("Targets_np shape after removing void: " + str(targets_np.shape))
        print("Encodings_np shape after removing void: " + str(encodings_np.shape))

        if self.segmentation:
            targets_np = self.reduce_targets_per_token(targets_np)

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
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]
                y = label.to(device)

                optimizer.zero_grad()
                preds = model(s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size)

                loss = train_loss_fn(preds, y)
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
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]
                y = label.to(device)

                with torch.no_grad():
                    preds = model(
                        s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                    )
                    all_preds.append(preds)
                    all_y.append(y)

            val_loss.append(val_loss_fn(torch.cat(all_preds).cpu(), torch.cat(all_y).cpu()))
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
