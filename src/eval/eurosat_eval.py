import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import rioxarray as xr
import torch.multiprocessing
import xarray
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from ..data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    normalize_dynamic,
)
from ..data.earthengine.s2 import ALL_S2_BANDS, REMOVED_BANDS
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")


class EuroSatDataset(PyTorchDataset):
    """
    EuroSat provides two datasets:
    - 27000 RGB images of 64x64 pixels (3 sen2 bands), 10 land cover classes
    - 27000 MSI images of 64x64 pixels (13 sen2 bands), 10 land cover classes
    """

    labels_to_int = {
        "AnnualCrop": 0,
        "Forest": 1,
        "HerbaceousVegetation": 2,
        "Highway": 3,
        "Industrial": 4,
        "Pasture": 5,
        "PermanentCrop": 6,
        "Residential": 7,
        "River": 8,
        "SeaLake": 9,
    }

    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt",
    }

    input_height_width = 64
    num_timesteps = 1

    def __init__(
        self,
        rgb: bool = True,
        split: str = "train",
        merge_train_val: bool = True,
        tif_files_dir: Optional[str] = "eurosat/EuroSAT_MS",
    ):
        assert split in ["train", "val", "test"]

        self.split = split
        self.rgb = rgb
        self.tif_files_dir = tif_files_dir

        self.images = self.split_images(merge_train_val)[split]
        self.masks = self.create_eurosat_masks()

    def image_name_to_path(self, name: str) -> Path:
        class_name = name.split("_")[0]
        if name.endswith("jpg"):
            name = f"{name.split('.')[0]}.tif"
        return data_dir / cast(str, self.tif_files_dir) / class_name / name

    @staticmethod
    def url_to_list(url: str) -> List[str]:
        data = urllib.request.urlopen(url).read()
        return data.decode("utf-8").split("\n")

    @staticmethod
    def split_images(merge_train_val: bool = True) -> Dict[str, List[str]]:
        # updated to use the splits stored in
        # https://storage.googleapis.com/remote_sensing_representations
        # as per torchgeo
        filename = (
            "eurosat/train_test_split.json"
            if merge_train_val
            else "eurosat/train_val_test_split.json"
        )
        split_path = data_dir / filename
        if split_path.exists():
            with split_path.open("r") as f:
                train_test_split = json.load(f)
        else:
            # this code was only run once (the dictionary is then saved)
            # but is saved here for clarity
            train_images = EuroSatDataset.url_to_list(EuroSatDataset.split_urls["train"])
            test_images = EuroSatDataset.url_to_list(EuroSatDataset.split_urls["test"])
            train_test_split = {"train": train_images, "test": test_images}
            if merge_train_val:
                train_test_split["train"] += EuroSatDataset.url_to_list(
                    EuroSatDataset.split_urls["val"]
                )
            else:
                train_test_split["val"] = EuroSatDataset.url_to_list(
                    EuroSatDataset.split_urls["val"]
                )
            json.dump(train_test_split, split_path.open("w"))
        return train_test_split

    def create_eurosat_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.rgb:
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2_RGB" in key
            ]

        else:
            dynamic_channels = [
                idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key
            ]

        # everything is masked by default
        dynamic_mask = np.ones([len(DYNAMIC_BANDS_GROUPS_IDX)])
        # unmask available s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # no static channels are available
        static_mask = np.ones(
            [self.input_height_width, self.input_height_width, len(STATIC_BAND_GROUPS_IDX)]
        )

        assert ((dynamic_mask == 0) | (dynamic_mask == 1)).all()
        assert (static_mask == 1).all()

        return (dynamic_mask, static_mask)

    def image_to_dynamic_eo_array(self, tif_filename: str) -> Tuple[np.ndarray, np.ndarray]:
        indices_to_remove = []
        for band in REMOVED_BANDS:
            indices_to_remove.append(ALL_S2_BANDS.index(band))
        kept_s2_bands = [i for i in range(len(ALL_S2_BANDS)) if i not in indices_to_remove]
        kept_dynamic_bands = [
            idx
            for idx, x in enumerate(DYNAMIC_BANDS)
            if ((x in ALL_S2_BANDS) and (x not in REMOVED_BANDS))
        ]

        tif_file = self.image_name_to_path(tif_filename)

        with cast(xarray.DataArray, xr.open_rasterio(tif_file)) as image:
            eo_style_array = np.zeros(
                [
                    self.input_height_width,
                    self.input_height_width,
                    self.num_timesteps,
                    len(DYNAMIC_BANDS),
                ]
            )
            image_kept_bands = image.values[kept_s2_bands]
            eo_style_array[:, :, :, kept_dynamic_bands] = repeat(
                image_kept_bands, "c h w -> h w t c", t=self.num_timesteps
            )

        return (
            normalize_dynamic(eo_style_array),
            np.array([self.labels_to_int[tif_file.parents[0].name]]),
        )

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image = self.images[idx]
        d_x, label = self.image_to_dynamic_eo_array(image.strip())

        # static bands are not provided by eurosat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], len(STATIC_BANDS)))

        d_m, s_m = self.masks
        month = np.zeros((self.num_timesteps,))

        label_torch = torch.tensor(label, dtype=torch.long)

        return (masked_output_np_to_tensor(d_x, s_x, d_m, s_m, month), label_torch)

    def __len__(self):
        return len(self.images)


class EuroSatEval(EvalTask):
    name = "eurosat"
    regression = False
    multilabel = False

    def __init__(self, rgb: bool = True, patch_size: int = 8, seed=DEFAULT_SEED):
        self.rgb = rgb
        super().__init__(patch_size, seed)
        self.name = f"{self.name}_{self.rgb}"

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator]
    ) -> Dict:
        test_dl = DataLoader(
            EuroSatDataset(rgb=self.rgb, split="test"),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]

            pretrained_model.eval()

            with torch.no_grad():
                d_x, s_x, d_m, s_m, _ = pretrained_model(
                    d_x, s_x, d_m, s_m, months, patch_size=self.patch_size
                )
                encodings = pretrained_model.average_tokens(d_x, s_x, d_m, s_m).cpu().numpy()

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
            EuroSatDataset(rgb=self.rgb, split="train", merge_train_val=True),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        return self._evaluate_model(pretrained_model, trained_sklearn_models)
