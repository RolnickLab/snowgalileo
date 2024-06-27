import json
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

import numpy as np
import torch.multiprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..flexipresto import Encoder
from ..utils import DEFAULT_SEED, device
from .eval import EvalTask, Hyperparams, model_class_name
from .geobench_dataset import GeobenchBaseDataset

torch.multiprocessing.set_sharing_strategy("file_system")

with (Path(__file__).parents[0] / Path("geobench_configs") / Path("m-bigearthnet.json")).open(
    "r"
) as f:
    config = json.load(f)


class BigEarthNetEval(EvalTask):
    name = "bigearthnet"
    regression = False
    spatial_token_prediction = False
    multilabel = False
    input_height_width = config["input_height_width"]
    num_outputs = config["num_classes"]

    def __init__(
        self,
        patch_size: int = 6,
        seed=DEFAULT_SEED,
        num_subtiles_per_image: int = 4,
    ):
        super().__init__(patch_size, seed)
        self.num_subtiles_per_image = num_subtiles_per_image
        self.input_height_width = self.input_height_width // int(
            sqrt(cast(float, self.num_subtiles_per_image))
        )
        # bigearthnet has an unusual input size, so we make sure it's divisible by the patch size
        assert (
            self.input_height_width % patch_size == 0
        ), "Input height/width must be divisible by patch size"

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
            f"{self.name}: {model_name}_average_precision_score": average_precision_score(
                target, preds, average="micro"
            ),
        }

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator]
    ) -> Dict:
        test_dl = DataLoader(
            GeobenchBaseDataset(
                dataset_config_file="m-bigearthnet.json",
                split="test",
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

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
                    pretrained_model.average_tokens(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
                    .cpu()
                    .numpy()
                )

            labels_list.append(label.cpu().numpy())

            for model in sklearn_models:
                preds = model.predict(encodings)
                pred_dict[model_class_name(model)].append(preds)

        target = np.concatenate(labels_list)
        results_dict = {}

        for model_name_str, pred_list in pred_dict.items():
            test_preds_np = np.concatenate(pred_list, axis=0)
            prefix = f"{model_name_str}"
            results_dict.update(self.compute_metrics(prefix, test_preds_np, target))
        return results_dict

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        train_dl = DataLoader(
            GeobenchBaseDataset(
                dataset_config_file="m-bigearthnet.json",
                split="train",
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        return self._evaluate_model(pretrained_model, trained_sklearn_models)
