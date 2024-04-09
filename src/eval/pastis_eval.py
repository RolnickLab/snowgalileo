import json
from math import sqrt
from typing import Dict, List, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch.multiprocessing
from einops import repeat
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from ..data.dataset import (
    DYNAMIC_BANDS,
    DYNAMIC_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
)
from ..data.earthengine.s2 import S2_BANDS
from ..flexipresto import Encoder, PrestoFineTuningModel
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")


class PastisDataset(PyTorchDataset):
    labels_to_int = {
        "Background": 0,
        "Meadow": 1,
        "SoftWinterWheat": 2,
        "Corn": 3,
        "WinterBarley": 4,
        "WinterRapeseed": 5,
        "SpringBarley": 6,
        "Sunflower": 7,
        "Grapevine": 8,
        "Beet": 9,
        "WinterTriticale": 10,
        "WinterDurumWheat": 11,
        "FruitsVegetablesFlowers": 12,
        "Potatoes": 13,
        "LeguminousFodder": 14,
        "Soybeans": 15,
        "Orchard": 16,
        "MixedCereal": 17,
        "Sorghum": 18,
        "VoidLabel": 19,
    }

    input_height_width = 128
    # PASTIS comes with a variable number of timesteps, we use the minimum number available in all tiles
    num_timesteps = 38

    def __init__(
        self,
        folds: List[int],
        data_path: Optional[str] = "pastis/PASTIS-R",
        num_subtiles: Optional[int] = 4,
    ):
        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)

        self.data_path = data_path

        self.metadata = gpd.read_file(data_dir / cast(str, self.data_path) / "metadata.geojson")
        self.metadata.index = self.metadata["ID_PATCH"].astype(int)
        self.metadata.sort_index(inplace=True)

        self.metadata = pd.concat([self.metadata[self.metadata["Fold"] == f] for f in folds])
        self.norm = self.get_pastis_norm()

        self.id = self.metadata.index

        self.num_subtiles = num_subtiles

    def get_months_from_metadata(self, id, image_timesteps) -> np.ndarray:
        dates = self.metadata["dates-S2"][id]

        # the dates are in the format YYYYMMDD
        months = [int(str(value)[4:6]) for _, value in dates.items()]

        assert all(1 <= month <= 12 for month in months)

        sampled_timesteps = np.random.default_rng(seed=DEFAULT_SEED).permutation(image_timesteps)[
            : self.num_timesteps
        ]

        return np.array(months)[sampled_timesteps]

    def get_pastis_norm(self):
        with open((data_dir / cast(str, self.data_path) / "NORM_S2_patch.json"), "r") as file:
            normvals = json.loads(file.read())
        means = [normvals["Fold_{}".format(f)]["mean"] for f in self.folds]
        stds = [normvals["Fold_{}".format(f)]["std"] for f in self.folds]

        return np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)

    def create_pastis_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        dynamic_channels = [idx for idx, key in enumerate(DYNAMIC_BANDS_GROUPS_IDX) if "S2" in key]

        # everything is masked by default
        dynamic_mask = np.ones([len(DYNAMIC_BANDS_GROUPS_IDX)])
        # unmask available bands
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

    def get_dynamic_eo_array(self, id) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(
            data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id)
        ).astype(np.float32)
        # data comes in shape T x C x H x W
        image_timesteps = data.shape[0]

        # randomly sample the timesteps if there are more than the minimum
        if image_timesteps > self.num_timesteps:
            sampled_timesteps = np.random.default_rng(seed=DEFAULT_SEED).permutation(
                image_timesteps
            )[: self.num_timesteps]
            data = data[sampled_timesteps, :, :, :]

        # apply normalization
        data = (data - self.norm[0][None, :, None, None]) / self.norm[1][None, :, None, None]

        kept_dynamic_bands = [idx for idx, x in enumerate(DYNAMIC_BANDS) if (x in S2_BANDS)]

        eo_style_array = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(DYNAMIC_BANDS),
            ]
        )
        eo_style_array[:, :, :, kept_dynamic_bands] = repeat(data, "t c h w -> h w t c")

        return eo_style_array, image_timesteps

    def get_target(self, id):
        target = np.load(
            data_dir / cast(str, self.data_path) / "ANNOTATIONS/TARGET_{}.npy".format(id)
        )
        return torch.from_numpy(target[0].astype(int))

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        img_idx = idx // 4

        id = self.id[img_idx]

        d_x, image_timesteps = self.get_dynamic_eo_array(id)
        target = self.get_target(id)

        # static bands are not provided by pastis
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], len(STATIC_BANDS)))

        d_m, s_m = self.create_pastis_masks()
        months = self.get_months_from_metadata(id, image_timesteps)

        subtiles_per_dim = sqrt(self.num_subtiles)
        subtiles_per_dim = int(subtiles_per_dim)
        h, w = d_x.shape[:2]
        assert h == w  # this is the case for PASTIS
        assert h % subtiles_per_dim == 0
        pixels_per_dim = h // subtiles_per_dim
        subtile_idx = idx % subtiles_per_dim

        row_idx = subtile_idx // subtiles_per_dim
        col_idx = subtile_idx % subtiles_per_dim

        return (
            masked_output_np_to_tensor(
                d_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                s_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                d_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                s_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                months,
            ),
            target[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ],
        )

    def __len__(self):
        return self.metadata.shape[0] * self.num_subtiles


class PastisEval(EvalTask):
    name = "pastis"
    regression = False
    multilabel = False
    segmentation = True
    num_outputs = len(PastisDataset.labels_to_int)
    # TODO: change, this is not dynamic!
    input_height_width = PastisDataset.input_height_width // 2

    def __init__(self, patch_size: int = 8, seed=DEFAULT_SEED):
        super().__init__(patch_size, seed)

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_iou": jaccard_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(self, finetuned_model: PrestoFineTuningModel) -> Dict:
        test_dl = DataLoader(
            PastisDataset(folds=[5]),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )
        pred_list = []

        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            d_x, s_x, d_m, s_m, months = [t.to(device) for t in masked_output]

            labels_list.append(label.cpu().numpy())

            finetuned_model.eval()

            with torch.no_grad():
                preds = (
                    finetuned_model(d_x, s_x, d_m, s_m, months, patch_size=self.patch_size)
                    .cpu()
                    .numpy()
                )

            pred_list.append(preds)

        target = np.concatenate(labels_list)
        results_dict = {}

        test_preds_np = np.concatenate(pred_list, axis=0)
        prefix = "finetuned"
        results_dict.update(self.compute_metrics(prefix, test_preds_np, target))

        return results_dict

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = ["finetune"]
        for model_mode in model_modes:
            assert model_mode in ["finetune"]

        train_dl = DataLoader(
            PastisDataset(folds=[1, 2, 3]),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )

        val_dl = DataLoader(
            PastisDataset(folds=[4]),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        finetuned_model = self.finetune_presto(train_dl, val_dl, pretrained_model)
        return self._evaluate_model(finetuned_model)
