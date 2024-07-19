import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import torch.multiprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..flexipresto import Encoder
from ..utils import DEFAULT_SEED, device
from .eval import EvalTask, Hyperparams
from .geobench_dataset import GeobenchBaseDataset

torch.multiprocessing.set_sharing_strategy("file_system")

with (Path(__file__).parents[0] / Path("geobench_configs") / Path("m-sa-crop-type.json")).open(
    "r"
) as f:
    config = json.load(f)


class SACropEval(EvalTask):
    name = "SA-crop-type"
    regression = False
    spatial_token_prediction = True
    multilabel = False
    input_height_width = config["input_height_width"]
    num_outputs = config["num_classes"]

    def __init__(
        self,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
        output_mode: str = "norm_counts",
        num_subtiles_per_image: int = 16,
    ):
        assert output_mode in ["mode", "norm_counts"]

        self.output_mode = output_mode
        if self.output_mode == "norm_counts":
            self.regression = True
        self.num_subtiles_per_image = num_subtiles_per_image
        super().__init__(patch_size=patch_size, seed=seed, output_mode=self.output_mode)
        self.input_height_width = self.input_height_width // int(
            sqrt(cast(float, self.num_subtiles_per_image))
        )
        self.name = f"{self.name}_{self.output_mode}"

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
            GeobenchBaseDataset(
                dataset_config_file="m-sa-crop-type.json",
                split="test",
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        results_dict = {}
        pred_dict: Dict[str, BaseEstimator] = {model: [] for model in sklearn_models}

        encodings_list = []
        targets_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

            targets = self.group_targets_per_token(label).cpu().numpy()
            targets_list.append(self.reduce_targets_per_token(targets))

            pretrained_model.eval()
            with torch.no_grad():
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
                )

                encodings = (
                    self.group_encodings_per_token(
                        pretrained_model, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m
                    )
                    .cpu()
                    .numpy()
                )
                encodings_list.append(encodings)

        encodings_np, targets_np = np.concatenate(encodings_list), np.concatenate(targets_list)

        for model in sklearn_models:
            preds = model.predict(encodings_np)
            pred_dict[model].append(preds)

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
            GeobenchBaseDataset(
                dataset_config_file="m-sa-crop-type.json",
                split="train",
                num_subtiles_per_image=self.num_subtiles_per_image,
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
