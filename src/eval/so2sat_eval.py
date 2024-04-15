from typing import Dict, List, Optional, Sequence, Tuple, cast

import h5py
import numpy as np
import torch.multiprocessing
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from ..data.dataset import (
    S1_BANDS,
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space_time,
)
from ..data.earthengine.s2 import S2_BANDS
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name

torch.multiprocessing.set_sharing_strategy("file_system")


class So2SatDataset(PyTorchDataset):
    """
    So2Sat data is provided as .h5 files in the following shapes:
    sen1: [n, 32, 32, 8]
    sen2: [n, 32, 32, 10]
    label: [n, 17] (one-hot encoded labels for 17 LCV classes)
    """

    input_height_width = 32
    num_timesteps = 1

    def __init__(
        self,
        split: str = "training",
        so2sat_dir: str = "so2sat/block/",
    ):
        assert split in ["training", "testing"]

        self.split = split
        self.so2sat_dir = so2sat_dir
        self._len = None
        self.masks = self.create_so2sat_masks()

    def h5_to_eo_array_and_label(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
            assert data["sen1"].shape == (self.__len__(), 32, 32, 8)
            assert data["sen2"].shape == (self.__len__(), 32, 32, 10)
            assert data["label"].shape == (self.__len__(), 17)

            # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
            s1 = np.array(data["sen1"][idx, :, :, 4:6])
            # sen2 bands provided by so2sat correspond to the bands used by presto
            s2 = np.array(data["sen2"][idx, :, :, :10])

            label = np.array(data["label"][idx, :])

        image = np.concatenate([s1, s2], axis=-1)

        return image, label

    def create_so2sat_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s_t_channels = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key]

        # everything is masked by default
        s_t_m = np.ones([len(SPACE_TIME_BANDS_GROUPS_IDX)])
        # unmask available s1 and s2 bands
        s_t_m[s_t_channels] = 0
        s_t_m = repeat(
            s_t_m,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # no static channels are available
        s_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (s_m == 1).all()
        assert (t_m == 1).all()

        return (s_t_m, s_m, t_m)

    def image_to_dynamic_eo_array(self, image: np.ndarray) -> np.ndarray:
        kept_dynamic_bands = [
            idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS or x in S1_BANDS)
        ]

        eo_style_array = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(SPACE_TIME_BANDS),
            ]
        )
        eo_style_array[:, :, :, kept_dynamic_bands] = repeat(
            image, "h w c -> h w t c", t=self.num_timesteps
        )

        return normalize_space_time(eo_style_array)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image, label = self.h5_to_eo_array_and_label(idx)
        s_t_x = self.image_to_dynamic_eo_array(image)

        # space only / time only bands are not provided by so2sat
        s_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        s_t_m, s_m, t_m = self.masks
        month = np.zeros((self.num_timesteps,))

        label_torch = torch.tensor(label, dtype=torch.long)

        return (masked_output_np_to_tensor(s_t_x, s_x, t_x, s_t_m, s_m, t_m, month), label_torch)

    def __len__(self) -> int:
        if self._len is None:
            with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
                self._len = data["sen1"].shape[0]
        return cast(int, self._len)


class So2SatEval(EvalTask):
    name = "so2sat"
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
            So2SatDataset(split="testing"),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]

            pretrained_model.eval()

            with torch.no_grad():
                s_t_x, s_x, t_x, s_t_m, s_m, t_m, _ = pretrained_model(
                    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                )
                encodings = (
                    pretrained_model.average_tokens(s_t_x, s_x, t_x, s_t_m, s_m, t_m).cpu().numpy()
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
            So2SatDataset(split="training"),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        return self._evaluate_model(pretrained_model, trained_sklearn_models)
