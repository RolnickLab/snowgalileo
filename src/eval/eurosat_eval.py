import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch.multiprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import Normalizer
from src.data.dataset import (
    SPACE_TIME_HIGH_RES_BANDS,
)
from src.eval.eval import EvalTask, Hyperparams, model_class_name
from src.eval.geobench_dataset import GeobenchBaseDataset
from src.flexipresto import Encoder
from src.masking import UNMASKING_CHANNEL_GROUPS
from src.utils import DEFAULT_SEED, device

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")

with (Path(__file__).parents[0] / Path("geobench_configs") / Path("m-eurosat.json")).open(
    "r"
) as f:
    config = json.load(f)


band_info_names_to_band_names = {
    "B2": "02 - Blue",
    "B3": "03 - Green",
    "B4": "04 - Red",
    "B8": "08 - NIR",
    "B11": "11 - SWIR",
    "B12": "12 - SWIR",
}


class EuroSatEval(EvalTask):
    name = "eurosat"
    regression = False
    spatial_token_prediction = False
    multilabel = False
    input_height_width = config["input_height_width"]
    num_outputs = config["num_classes"]

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        rgb: bool = True,
        include_latlons: bool = True,
        patch_size_high_res: int = 8,
        seed=DEFAULT_SEED,
        geobench: bool = False,
    ):
        self.rgb = rgb
        self.geobench = geobench
        self.include_latlons = include_latlons
        if isinstance(normalization, Normalizer):
            self.normalization = "custom"
            self.normalizer = normalization
        else:
            assert normalization in ["std", "scaling"]
            self.normalization = normalization

            if normalization == "scaling":
                self.normalizer = Normalizer(std=False)
            else:
                self.normalizer = self.load_eurosat_normalizer()

        assert not self.geobench or not self.include_latlons, "Geobench does not support latlons"

        super().__init__(patch_size_high_res, seed)
        self.name = f"{self.name}_{'RGB' if self.rgb else 'MS'}{'_latlons' if include_latlons else ''}_{'_geobench' if geobench else ''}"

        output_channels = [0] * len(UNMASKING_CHANNEL_GROUPS)
        for i, val in enumerate(UNMASKING_CHANNEL_GROUPS):
            if val[1] == "DW_static":
                output_channels[i] = 1

        input_channels = [0] * len(UNMASKING_CHANNEL_GROUPS)
        for i, val in enumerate(UNMASKING_CHANNEL_GROUPS):
            if val[1] in ["S2_RGB", "S2_SWIR", "S2_NIR"]:
                input_channels[i] = 1

    @staticmethod
    def load_eurosat_normalizer() -> Normalizer:
        normalizing_dict = {
            "space_time_high_res": {
                "mean": [0] * len(SPACE_TIME_HIGH_RES_BANDS),
                "std": [1] * len(SPACE_TIME_HIGH_RES_BANDS),
            }
        }
        for our_band, c_band in band_info_names_to_band_names.items():
            idx = SPACE_TIME_HIGH_RES_BANDS.index(our_band)
            normalizing_dict["space_time_high_res"]["mean"][idx] = config["band_info"][c_band][
                "mean"
            ]
            normalizing_dict["space_time_high_res"]["std"][idx] = config["band_info"][c_band][
                "std"
            ]
        return Normalizer(std=True, normalizing_dicts=normalizing_dict)

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator], c_i
    ) -> Dict:
        if self.geobench:
            test_dl = DataLoader(
                GeobenchBaseDataset(
                    normalizer=self.normalizer,
                    dataset_config_file="m-eurosat.json",
                    split="valid",
                    rgb=self.rgb,
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )
        else:
            raise NotImplementedError(
                "EuroSat dataset is only implemented for geobench at this point. Please use the geobench version of the dataset."
            )
        pred_dict: Dict[str, BaseEstimator] = {
            model_class_name(model): [] for model in sklearn_models
        }

        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
                months,
            ) = [t.to(device) for t in masked_output]

            pretrained_model.eval()

            with torch.no_grad():
                (
                    s_t_h_x,
                    s_t_m_x,
                    s_t_l_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_h_m,
                    s_t_m_m,
                    s_t_l_m,
                    sp_m,
                    t_m,
                    st_m,
                    _,
                ) = pretrained_model(
                    s_t_h_x=s_t_h_x,
                    s_t_m_x=s_t_m_x,
                    s_t_l_x=s_t_l_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_h_m=s_t_h_m,
                    s_t_m_m=s_t_m_m,
                    s_t_l_m=s_t_l_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    c_i=c_i,
                    patch_size_high_res=self.patch_size_high_res,
                    patch_size_med_res=1,
                    patch_size_low_res=1,
                )
                encodings = (
                    pretrained_model.apply_mask_and_average_tokens(
                        s_t_h_x,
                        s_t_m_x,
                        s_t_l_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_h_m,
                        s_t_m_m,
                        s_t_l_m,
                        sp_m,
                        t_m,
                        st_m,
                    )
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

    def train_and_evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_classification_sklearn_models
        for model_mode in model_modes:
            assert model_mode in self.all_classification_sklearn_models

        if self.geobench:
            train_dl = DataLoader(
                GeobenchBaseDataset(
                    normalizer=self.normalizer,
                    dataset_config_file="m-eurosat.json",
                    split="train",
                    rgb=self.rgb,
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=False,
                num_workers=Hyperparams.num_workers,
            )
        else:
            raise NotImplementedError(
                "EuroSat dataset is only implemented for geobench at this point. Please use the geobench version of the dataset."
            )

        unconditioned_trained_sklearn_models = self.train_sklearn_model(
            train_dl, pretrained_model, model_modes, None
        )
        unconditioned_results = self._evaluate_model(
            pretrained_model, unconditioned_trained_sklearn_models, None
        )

        return unconditioned_results
