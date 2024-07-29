import json
from pathlib import Path
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
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space_time,
)
from ..data.earthengine.s2 import S2_BANDS
from ..flexipresto import Encoder
from ..masking import MASKING_MODES, MASKING_MODES_COARSE, MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name
from .geobench_dataset import GeobenchBaseDataset

torch.multiprocessing.set_sharing_strategy("file_system")

with (Path(__file__).parents[0] / Path("geobench_configs") / Path("m-so2sat.json")).open("r") as f:
    config = json.load(f)


class So2SatTUMDataset(PyTorchDataset):
    """
    So2Sat data is provided as .h5 files in the following shapes:
    sen1: [n, 32, 32, 8]
    sen2: [n, 32, 32, 10]
    label: [n, 17] (one-hot encoded labels for 17 LCV classes)
    """

    input_height_width = config["input_height_width"]
    num_timesteps = config["num_timesteps"]
    num_classes = config["num_classes"]

    def __init__(
        self,
        split: str = "training",
        so2sat_dir: str = "so2sat/block/",
    ):
        assert split in ["training", "testing"]

        self.split = split
        self.so2sat_dir = so2sat_dir
        self._len = None
        self.masks = self.create_so2sat_masks(combined=True)

    def h5_to_eo_array_and_label(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
            assert data["sen1"].shape == (self.__len__(), 32, 32, 8)
            assert data["sen2"].shape == (self.__len__(), 32, 32, 10)
            assert data["label"].shape == (self.__len__(), 17)

            # so2sat provides 8 bands for sen1, we are interested in the filtered vh and vv bands (channel 4 and 5)
            vh = np.expand_dims(np.array(data["sen1"][idx, :, :, 4]), axis=-1)
            vv = np.expand_dims(np.array(data["sen1"][idx, :, :, 5]), axis=-1)
            # sen2 bands provided by so2sat correspond to the bands used by presto
            s2 = np.array(data["sen2"][idx, :, :, :10])

            label = np.array(data["label"][idx, :])

        image = np.concatenate([vv, vh, s2], axis=-1)

        # reverse one-hot encoding, original labels start from 1
        label = np.array(np.argmax(label) + 1)

        return image, label

    def create_so2sat_masks(
        self, combined: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if combined:
            s_t_channels = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key
            ]
        else:
            s_t_channels = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
            ]

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
        sp_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])
        st_m = np.ones([len(STATIC_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (sp_m == 1).all()
        assert (t_m == 1).all()
        assert (st_m == 1).all()

        return (s_t_m, sp_m, t_m, st_m)

    def image_to_space_time_array(self, image: np.ndarray, combined: bool) -> np.ndarray:
        if combined:
            kept_dynamic_bands = [
                idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS or x in S1_BANDS)
            ]
        else:
            kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)]

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
        s_t_x = self.image_to_space_time_array(image, combined=True)

        # space only / time only / static bands are not provided by so2sat
        sp_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))
        st_x = np.zeros((len(STATIC_BANDS)))

        s_t_m, sp_m, t_m, st_m = self.masks
        month = np.zeros((self.num_timesteps,))

        label_torch = torch.tensor(label, dtype=torch.long)

        return (
            masked_output_np_to_tensor(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, month),
            label_torch,
        )

    def __len__(self) -> int:
        if self._len is None:
            with h5py.File(data_dir / self.so2sat_dir / f"{self.split}.h5", "r") as data:
                self._len = data["sen1"].shape[0]
        return cast(int, self._len)


class So2SatEval(EvalTask):
    name = "so2sat"
    regression = False
    spatial_token_prediction = False
    multilabel = False
    input_height_width = config["input_height_width"]
    num_outputs = config["num_classes"]

    def __init__(
        self,
        geobench: bool = True,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
        do_condition: bool = False,
    ):
        self.geobench = geobench
        super().__init__(patch_size, seed)

        self.do_condition = do_condition
        if self.geobench:
            self.name = f"{self.name}_geobench"
        else:
            self.name = f"{self.name}_tum"

        input_channels = [0] * len(MASKING_MODES)
        output_channels = [0] * len(MASKING_MODES)
        for i, val in enumerate(MASKING_MODES):
            if "S2" in val[1]:
                input_channels[i] = 1
            elif val[1] == "DW_static":
                output_channels[i] = 1

        output_channels = [0] * len(MASKING_MODES_COARSE)
        for i, val in enumerate(MASKING_MODES_COARSE):
            if val == "static":
                output_channels[i] = 1
        self.condition = {"output_channels": torch.Tensor(output_channels).to(device)}

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(
        self,
        pretrained_model: Encoder,
        sklearn_models: Sequence[BaseEstimator],
        c_i: Optional[Dict] = None,
    ) -> Dict:
        if self.geobench:
            test_dl = DataLoader(
                GeobenchBaseDataset(dataset_config_file="m-so2sat.json", split="test"),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )
        else:
            test_dl = DataLoader(
                So2SatTUMDataset(split="testing"),
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
                    c_i=c_i,
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

        if self.geobench:
            train_dl = DataLoader(
                GeobenchBaseDataset(dataset_config_file="m-so2sat.json", split="train"),
                batch_size=Hyperparams.batch_size,
                shuffle=True,
                num_workers=Hyperparams.num_workers,
            )
        else:  # TUM version
            train_dl = DataLoader(
                So2SatTUMDataset(split="training"),
                batch_size=Hyperparams.batch_size,
                shuffle=True,
                num_workers=Hyperparams.num_workers,
            )

        unconditioned_trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, None
        )
        unconditioned_results = self._evaluate_model(
            pretrained_model, unconditioned_trained_sklearn_models, None
        )

        if not self.do_condition:
            return unconditioned_results

        conditioned_trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, self.condition
        )
        conditioned_results = self._evaluate_model(
            pretrained_model, conditioned_trained_sklearn_models, self.condition
        )
        conditioned_results = {f"{key}_c": value for key, value in conditioned_results.items()}

        return {**conditioned_results, **unconditioned_results}
