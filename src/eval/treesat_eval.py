import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    S1_BANDS,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    normalize_dynamic,
)
from ..data.earthengine.s2 import S2_BANDS
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name

treesat_dir = "treesat"
s1_files_dir = "s1/60m"
s2_files_dir = "s2/60m"
labels_file = "TreeSatBA_v9_60m_multi_labels.json"

# https://zenodo.org/record/6780578
# Band order is B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, and B09.
# Spatial resolution is 10 m.
S2_BAND_ORDERING = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]
# Band order is VV, VH, and VV/VH ratio. Spatial resolution is 10 m.
S1_BAND_ORDERING = ["VV", "VH", "VV/VH"]


class TreeSatDataset(Dataset):
    labels_to_int = {
        "Abies": 0,
        "Acer": 1,
        "Alnus": 2,
        "Betula": 3,
        "Cleared": 4,
        "Fagus": 5,
        "Fraxinus": 6,
        "Larix": 7,
        "Picea": 8,
        "Pinus": 9,
        "Populus": 10,
        "Prunus": 11,
        "Pseudotsuga": 12,
        "Quercus": 13,
        "Tilia": 14,
    }

    num_timesteps = 1
    # this is not the true start month!
    # the data is a mosaic of summer months
    start_month = 6
    # TODO: check if we should benchmark the 200m
    # data too
    input_height_width = 6

    def __init__(self, mode: str = "s2", split: str = "train"):
        assert mode in ["s2", "s1", "combined"]
        self.mode = mode
        self.split = split
        self.masks = self.make_masks()

        with (data_dir / treesat_dir / f"{split}_filenames.lst").open("r") as f:
            self.images = [line for line in f]

        with (data_dir / treesat_dir / labels_file).open("r") as f:
            self.labels_dict = json.load(f)

        # band mapping between the presto bands and the treesat bands
        self.kept_treesat_s2_band_idx = [
            i for i, val in enumerate(S2_BAND_ORDERING) if val in S2_BANDS
        ]
        kept_kept_treesat_s2_band_names = [val for val in S2_BAND_ORDERING if val in S2_BANDS]
        self.treesat_to_presto_s2_map = [
            DYNAMIC_BANDS.index(val) for val in kept_kept_treesat_s2_band_names
        ]

        self.kept_treesat_s1_band_idx = [
            i for i, val in enumerate(S1_BAND_ORDERING) if val in S1_BANDS
        ]
        kept_kept_treesat_s1_band_names = [val for val in S1_BAND_ORDERING if val in S1_BANDS]
        self.treesat_to_presto_s1_map = [
            DYNAMIC_BANDS.index(val) for val in kept_kept_treesat_s1_band_names
        ]

    def train_val_split(self, val_ratio: float = 0.1, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.images)
        val_ds = deepcopy(self)
        num_val = int(len(self.images) * val_ratio)
        val_ds.images = self.images[:num_val]
        self.images = self.images[num_val:]
        return self, val_ds

    @staticmethod
    def image_name_to_paths(tif_file: str) -> Tuple[Path, Path]:
        s1_path = data_dir / treesat_dir / s1_files_dir / Path(tif_file).name
        s2_path = data_dir / treesat_dir / s2_files_dir / Path(tif_file).name
        return s1_path, s2_path

    def image_to_dynamic_eo_array(self, tif_file: str):
        s1_image, s2_image = self.image_name_to_paths(tif_file)

        labels_np = np.zeros(len(self.labels_to_int))
        positive_classes = self.labels_dict[tif_file]
        for name, percentage in positive_classes:
            labels_np[self.labels_to_int[name]] = percentage

        d_x = np.zeros([len(DYNAMIC_BANDS), self.input_height_width, self.input_height_width])
        if self.mode in ["s2", "combined"]:
            with cast(xr.DataArray, rioxarray.open_rasterio(s2_image)) as s2:
                d_x[self.treesat_to_presto_s2_map] = s2.values[self.kept_treesat_s2_band_idx]
        if self.mode in ["s1", "combined"]:
            with cast(xr.DataArray, rioxarray.open_rasterio(s1_image)) as s1:
                d_x[self.treesat_to_presto_s1_map] = s1.values[self.kept_treesat_s1_band_idx]

        d_x = repeat(d_x, "c h w -> h w t c", t=self.num_timesteps)

        return normalize_dynamic(d_x), self.min_threshold(labels_np)

    @staticmethod
    def min_threshold(labels: np.ndarray, binarize: bool = True):
        # this is what is also done in
        # https://git.tu-berlin.de/rsim/treesat_benchmark/-/blob/master/TreeSat_Benchmark/trainers/utils.py#L27
        lower_bound = 0.07  # anything below this is ignored
        bounded = np.where(
            labels > lower_bound,
            np.ones_like(lower_bound) if binarize else labels,
            np.zeros_like(lower_bound),
        )
        return bounded

    def make_masks(self):
        if self.mode == "s2":
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key
            ]

        elif self.mode == "s1":
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S1" in key
            ]
        else:
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S" in key
            ]

        # everything is masked by default
        dynamic_mask = np.ones([len(DYNAMIC_BANDS_GROUPS_IDX)])
        # unmask available s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask, "d -> h w t d", h=self.input_height_width, w=self.input_height_width, t=1
        )

        # no static channels are available
        static_mask = np.ones(
            [self.input_height_width, self.input_height_width, len(STATIC_BAND_GROUPS_IDX)]
        )

        assert ((dynamic_mask == 0) | (dynamic_mask == 1)).all()
        assert (static_mask == 1).all()

        return (dynamic_mask, static_mask)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image = self.images[idx]
        d_x, label = self.image_to_dynamic_eo_array(image.strip())

        # static bands are not provided by eurosat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], len(STATIC_BANDS)))

        d_m, s_m = self.masks
        month = np.ones((self.num_timesteps,)) * self.start_month

        label_torch = torch.tensor(label, dtype=torch.long)

        return (masked_output_np_to_tensor(d_x, s_x, d_m, s_m, month), label_torch)

    def __len__(self):
        return len(self.images)


class TreeSatEval(EvalTask):
    name = "treesat"
    regression = False
    multilabel = True
    # different than the paper but this is
    # from all the unique classes in the labels json
    # (above)
    num_outputs = 15

    def __init__(self, mode: str = "s2", patch_size: int = 6, seed: int = DEFAULT_SEED):
        self.mode = mode
        super().__init__(patch_size, seed)
        self.name = f"{self.name}_{self.mode}"

    def compute_metrics(
        self, model_name: str, preds: np.ndarray, target: np.ndarray, threshold: float = 0.5
    ) -> Dict:
        preds_binary = preds > threshold
        return {
            f"{self.name}: {model_name}_num_samples": len(target),
            f"{self.name}: {model_name}_mAP_score_weighted": average_precision_score(
                target, preds, average="weighted"
            ),
            f"{self.name}: {model_name}_mAP_score_micro": average_precision_score(
                target, preds, average="micro"
            ),
            f"{self.name}: {model_name}_f1_score_weighted": f1_score(
                target, preds_binary, average="weighted"
            ),
            f"{self.name}: {model_name}_f1_score_micro": f1_score(
                target, preds_binary, average="micro"
            ),
            f"{self.name}: {model_name}_precision_micro": precision_score(
                target, preds_binary, average="micro"
            ),
            f"{self.name}: {model_name}_precision_weighted": precision_score(
                target, preds_binary, average="weighted"
            ),
            f"{self.name}: {model_name}_recall_micro": recall_score(
                target, preds_binary, average="micro"
            ),
            f"{self.name}: {model_name}_recall_weighted": recall_score(
                target, preds_binary, average="weighted"
            ),
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds_binary),
        }

    @torch.no_grad()
    def _evaluate_model(
        self,
        pretrained_model: Encoder,
        sklearn_models: Sequence[BaseEstimator],
    ) -> Dict:
        pretrained_model.eval()

        test_dl = DataLoader(
            TreeSatDataset(split="test", mode=self.mode),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        labels_list = []
        for masked_output, labels in tqdm(test_dl, desc="Computing test predictions"):
            d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]
            with torch.no_grad():
                d_x, s_x, d_m, s_m, _ = pretrained_model(
                    d_x, s_x, d_m, s_m, months, patch_size=self.patch_size
                )
                encodings = pretrained_model.average_tokens(d_x, s_x, d_m, s_m).cpu().numpy()
            labels_list.append(labels.cpu().numpy())
            for model in sklearn_models:
                preds_list = model.predict_proba(encodings)

                # this is a list of probabilities; we want to take the sum of
                # positive predictions
                preds = np.zeros((preds_list[0].shape[0], self.num_outputs))
                for idx, pred in enumerate(preds_list):
                    if pred.shape[1] == 2:
                        # if not, there are no positive samples
                        preds[:, idx] = pred[:, 1]
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
        dl = DataLoader(
            TreeSatDataset(split="train", mode=self.mode),
            shuffle=False,
            batch_size=Hyperparams.batch_size,
            num_workers=Hyperparams.num_workers,
        )
        trained_sklearn_models = self.train_sklearn_model(
            dl,
            pretrained_model,
            models=model_modes,
        )
        return self._evaluate_model(pretrained_model, trained_sklearn_models)
