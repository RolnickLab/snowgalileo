import logging
from collections import namedtuple
from typing import Tuple, Dict, List
from abc import ABC
import json

import numpy as np
import tqdm
import torch.multiprocessing
from datasets import load_dataset
from einops import repeat
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.presto import Encoder
from src.data.dataset import (
    DYNAMIC_BANDS_GROUPS_IDX,
    NUM_DYNAMIC_BAND_GROUPS,
    NUM_DYNAMIC_BANDS,
    NUM_STATIC_BAND_GROUPS,
)
from prediction_heads.knn import KNNat20

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")
logger = logging.getLogger("__main__")
MaskedOutput = namedtuple(
    "MaskedOutput", ["dynamic_x", "static_x", "dynamic_mask", "static_mask", "months"]
)


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

    def __init__(
        self,
        rgb: bool = True,
        split: str = "train",
        merge_train_val: bool = True,
    ):
        assert split in ["train", "validation", "test"]

        self.split = split
        self.rgb = rgb
        self.input_size = 64

        if self.rgb:
            self.data = load_dataset("blanchon/EuroSAT_RGB", split=self.split)

        # MSI data
        else:
            self.data = load_dataset("blanchon/EuroSAT_MSI", split=self.split)

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
        dynamic_mask = np.ones([NUM_DYNAMIC_BAND_GROUPS])
        # unmask available s2 bands
        dynamic_mask[dynamic_channels] = 0
        dynamic_mask = repeat(
            dynamic_mask, "d -> h w t d", h=self.input_size, w=self.input_size, t=1
        )

        # no static channels are available
        static_mask = np.ones([self.input_size, self.input_size, NUM_STATIC_BAND_GROUPS])

        assert np.unique(dynamic_mask).tolist() == [0, 1]
        assert np.unique(static_mask).tolist() == [1]

        return (dynamic_mask, static_mask)

    def add_missing_channels(self, d_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # s_x is not provided by eurosat
        s_x = np.zeros((d_x.shape[0], d_x.shape[1], 2))

        # 3 presto channels are provided by RGB
        if self.rgb:
            d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 3))
        else:
            d_x_missing = np.zeros((d_x.shape[0], d_x.shape[1], 1, NUM_DYNAMIC_BANDS - 10))

        d_x = np.concatenate((d_x, d_x_missing), axis=-1)

        return (d_x, s_x)

    def image_to_eo_array(self, idx: int) -> Tuple[np.ndarray, int]:
        image = np.array(self.data[idx]["image"])
        label = self.data[idx]["label"]

        # for MSI, remove band 1,9 and 10
        if not self.rgb:
            image = np.delete(image, [0, 9, 10], axis=2)

        return (image, label)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        d_x, label = self.image_to_eo_array(idx)
        d_x = d_x.reshape(d_x.shape[0], d_x.shape[1], 1, d_x.shape[2])

        d_x, s_x = self.add_missing_channels(d_x)

        d_m, s_m = self.create_eurosat_masks()
        month = np.zeros((1,))

        d_x_torch = torch.as_tensor(d_x, dtype=torch.float32)
        s_x_torch = torch.as_tensor(s_x, dtype=torch.float32)
        d_m_torch = torch.as_tensor(d_m, dtype=torch.float32)
        s_m_torch = torch.as_tensor(s_m, dtype=torch.float32)
        month_torch = torch.as_tensor(month, dtype=torch.long)
        label_torch = torch.as_tensor(label, dtype=torch.long)

        return (MaskedOutput(d_x_torch, s_x_torch, d_m_torch, s_m_torch, month_torch), label_torch)

    def __len__(self):
        return len(self.data)


HYPERPARAMS = {
    "batch_size": 64,
    "num_workers": 0,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EuroSatEval(ABC):

    def init(
        self,
        rgb: bool = False
    ):
        self.rgb = rgb
        self.name = f"EuroSat" if not rgb else f"EuroSat_RGB"

    def normalize(self, x: np.ndarray) -> np.ndarray:
        NotImplementedError

    @torch.no_grad()
    def evaluate(
        self,
        finetuned_model,
        pretrained_model,
    ) -> Dict:

        test_ds = EuroSatDataset(split="test")

        test_dl = DataLoader(
            test_ds, 
            batch_size=1, 
            shuffle=False,
            num_workers=HYPERPARAMS["num_workers"],
        )

        labels = []
        pred_list = []

        for b in tqdm(test_dl, desc="Computing test predictions"):
            sample, label = b
            d_x, d_m, s_x, s_m, month = sample
            d_x, d_m, s_x, s_m, month = [t.to(device) for t in (d_x, d_m, s_x, s_m, month)]

            pretrained_model.eval()
            encodings = (
                pretrained_model(
                d_x=d_x,
                s_x=s_x,
                d_m=d_m,
                s_m=s_m,
                month=month)
                .cpu()
                .numpy()
            )

            labels.append(
                label.cpu()
                .numpy()
                .reshape(
                    (
                        encodings.shape[0],
                        1,
                        *label.shape[1:],
                    )
                )[:, 0]
            )
            assert not torch.isnan(encodings).any()

            preds = finetuned_model.predict(encodings)
            pred_list.append(preds)

        target = np.concatenate(labels)
        results_dict = {}
        int_to_labels, _ = zip(*sorted(test_ds.labels_to_int.items(), key=lambda l_i: l_i[1]))
        
        test_preds_np = np.concatenate(pred_list, axis=0)
        prefix = finetuned_model.__class__.__name__
        results_dict.update(
            {
                f"{self.name}: {prefix}_num_samples": len(target),
                f"{self.name}: {prefix}_f1_score": f1_score(
                    target, test_preds_np, average="weighted"
                ),
                f"{self.name}: {prefix}_accuracy_score": accuracy_score(
                    target, test_preds_np
                ),
            }
        )
        class_matrix = confusion_matrix(test_preds_np, target)
        accuracies = class_matrix.diagonal() / class_matrix.sum(axis=1)
        for f1, acc, label in zip(
            f1_score(target, test_preds_np, average=None), accuracies, int_to_labels
        ):
            results_dict[f"{self.name}: {prefix}_f1_score_{label}"] = f1
            results_dict[f"{self.name}: {prefix}_accuracy_score_{label}"] = acc

        return results_dict



    @torch.no_grad()
    def finetune_knn(
        self,
        pretrained_model,
        train_dl: DataLoader,
    ):

        pretrained_model.eval()

        encoding_list, target_list = [], []
        for b in tqdm(train_dl, leave=False, desc="Computing embeddings"):

            sample, label = b
            d_x, d_m, s_x, s_m, month = sample

            d_x, d_m, s_x, s_m, month = [t.to(device) for t in (d_x, d_m, s_x, s_m, month)]

            target_list.append(label.cpu().numpy())
            with torch.no_grad():
                encodings = (
                    pretrained_model(
                        d_x=d_x, 
                        s_x=s_x,
                        d_m=d_m, 
                        s_m=s_m, 
                        month=month
                    )
                    .cpu()
                    .numpy()
                )
                encoding_list.append(encodings)
        encodings_np = np.concatenate(encoding_list)
        targets = np.concatenate(target_list)

        if len(targets.shape) == 2 and targets.shape[1] == 1:
            logger.info("Flattening targets")
            targets = targets.ravel()

        fitted_model = KNNat20().fit(encodings_np, targets)
        logger.info("Fitted model type: %s", type(fitted_model))
        return fitted_model


    def finetune(
        self, 
        pretrained_model,
        prediction_head: str = "knn",
    ):
        # others not implemented yet
        assert prediction_head == "knn"

        results_dict = {}

        train_ds = EuroSatDataset(split="train")
        val_ds = EuroSatDataset(split="validation")

        # TODO: normalization
        train_dl = DataLoader(
            train_ds, 
            batch_size=HYPERPARAMS["batch_size"], 
            shuffle=True,
            num_workers=HYPERPARAMS["num_workers"],
        )

        # TODO: implement train val merging
        val_dl = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False,
            num_workers=HYPERPARAMS["num_workers"],
        )

        finetuned_model = self.finetune_knn(
            train_dl,
            pretrained_model,
        )

        results_dict.update(
            self.evaluate(finetuned_model, pretrained_model)
        )

        return results_dict
    

# random initialized Presto model
model = Encoder(embedding_size=64).to(device)
model.to(device)

eval_task = EuroSatEval(rgb=True)
results = eval_task.finetune(model)

logger.info(json.dumps(results, indent=2))