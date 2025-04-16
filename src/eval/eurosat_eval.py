import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import rioxarray as xr
import torch.multiprocessing
import xarray
from einops import repeat
from pyproj import Transformer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from tqdm import tqdm

from ..data import Normalizer
from ..data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_HIGH_RES_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
    to_cartesian,
)
from ..data.earthengine.eo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BANDS_GROUPS_IDX,
)
from ..data.earthengine.s2 import ALL_S2_BANDS, REMOVED_BANDS
from ..flexipresto import Encoder
from ..masking import UNMASKING_CHANNEL_GROUPS, MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name
from .geobench_dataset import GeobenchBaseDataset

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
        normalizer: Normalizer,
        rgb: bool = True,
        split: str = "train",
        merge_train_val: bool = True,
        tif_files_dir: Optional[str] = "eurosat/EuroSAT_MS",
        include_latlons: bool = True,
    ):
        assert split in ["train", "val", "test"]

        self.split = split
        self.rgb = rgb
        self.include_latlons = include_latlons
        self.tif_files_dir = tif_files_dir
        self.normalizer = normalizer

        self.images = self.split_images(merge_train_val)[split]
        self.masks = self.create_eurosat_masks()

        # used in the tif_to_array function
        indices_to_remove = []
        for band in REMOVED_BANDS:
            indices_to_remove.append(ALL_S2_BANDS.index(band))
        self.kept_s2_bands = [i for i in range(len(ALL_S2_BANDS)) if i not in indices_to_remove]
        self.kept_dynamic_bands = [
            idx
            for idx, x in enumerate(SPACE_TIME_HIGH_RES_BANDS)
            if ((x in ALL_S2_BANDS) and (x not in REMOVED_BANDS))
        ]
        self.kept_static_bands = [idx for idx, x in enumerate(STATIC_BANDS)]

    def image_name_to_path(self, name: str) -> Path:
        class_name = name.split("_")[0]
        if name.endswith("jpg"):
            name = f"{name.split('.')[0]}.tif"
        return data_dir / cast(str, self.tif_files_dir) / class_name / name

    @staticmethod
    def url_to_list(url: str) -> List[str]:
        data = urllib.request.urlopen(url).read()
        return data.decode("utf-8").split("\n")

    @classmethod
    def split_images(cls, merge_train_val: bool = True) -> Dict[str, List[str]]:
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
            train_images = cls.url_to_list(cls.split_urls["train"])
            test_images = cls.url_to_list(cls.split_urls["test"])
            train_test_split = {"train": train_images, "test": test_images}
            if merge_train_val:
                train_test_split["train"] += cls.url_to_list(cls.split_urls["val"])
            else:
                train_test_split["val"] = cls.url_to_list(cls.split_urls["val"])
            json.dump(train_test_split, split_path.open("w"))
        return train_test_split

    def create_eurosat_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.rgb:
            space_time_high_res_channels = [
                idx
                for idx, key in enumerate(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)
                if "S2_RGB" in key
            ]

        else:
            space_time_high_res_channels = [
                idx for idx, key in enumerate(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX) if "S2" in key
            ]

        # everything is masked by default
        space_time_high_res_mask = np.ones([len(SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX)])
        # unmask available s2 bands
        space_time_high_res_mask[space_time_high_res_channels] = 0
        space_time_high_res_mask = repeat(
            space_time_high_res_mask,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # no space only / time only channels are available
        space_mask = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        time_mask = np.ones([self.num_timesteps, len(TIME_BANDS_GROUPS_IDX)])
        static_mask = np.ones([len(STATIC_BAND_GROUPS_IDX)])
        if self.include_latlons:
            location_channels = [
                idx for idx, key in enumerate(STATIC_BAND_GROUPS_IDX) if "location" in key
            ]
            static_mask[location_channels] = 0
            assert ((static_mask == 0) | (static_mask == 1)).all()
        else:
            assert (static_mask == 1).all()

        assert ((space_time_high_res_mask == 0) | (space_time_high_res_mask == 1)).all()
        assert (space_mask == 1).all()
        assert (time_mask == 1).all()

        return (space_time_high_res_mask, space_mask, time_mask, static_mask)

    def image_to_space_time_array(
        self, tif_filename: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tif_file = self.image_name_to_path(tif_filename)

        with cast(xarray.DataArray, xr.open_rasterio(tif_file)) as image:
            eo_style_array = np.zeros(
                [
                    self.input_height_width,
                    self.input_height_width,
                    self.num_timesteps,
                    len(SPACE_TIME_HIGH_RES_BANDS),
                ]
            )
            image_kept_bands = image.values[self.kept_s2_bands]
            eo_style_array[:, :, :, self.kept_dynamic_bands] = repeat(
                image_kept_bands, "c h w -> h w t c", t=self.num_timesteps
            )
            # from (e.g.) +init=epsg:32630 to epsg:32630
            crs = image.rio.crs.data["init"]
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(np.mean(image.x).item(), np.mean(image.y).item())
            cartesian_array = to_cartesian(lat, lon)

            static_array = np.zeros(
                len(STATIC_BANDS),
            )
            static_array[self.kept_static_bands] = cartesian_array

        return (
            self.normalizer(eo_style_array),
            self.normalizer(static_array),
            np.array([self.labels_to_int[tif_file.parents[0].name]]),
        )

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        image = self.images[idx]
        s_t_x, st_x, label = self.image_to_space_time_array(image.strip())

        # space and time bands are not provided by eurosat
        sp_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        s_t_m, sp_m, t_m, st_m = self.masks
        month = np.zeros((self.num_timesteps,))

        label_torch = torch.tensor(label, dtype=torch.long)

        return (
            masked_output_np_to_tensor(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, month),
            label_torch,
        )

    def __len__(self):
        return len(self.images)


class EuroSatEval(EvalTask):
    name = "eurosat"
    regression = False
    spatial_token_prediction = False
    multilabel = False
    input_height_width = EuroSatDataset.input_height_width
    num_outputs = len(EuroSatDataset.labels_to_int)

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        rgb: bool = True,
        include_latlons: bool = True,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
        geobench: bool = False,
        do_condition: bool = False,
    ):
        self.rgb = rgb
        self.geobench = geobench
        self.include_latlons = include_latlons
        self.do_condition = do_condition
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

        super().__init__(patch_size, seed)
        self.name = f"{self.name}_{'RGB' if self.rgb else 'MS'}{'_latlons' if include_latlons else ''}_{'_geobench' if geobench else ''}"

        output_channels = [0] * len(UNMASKING_CHANNEL_GROUPS)
        for i, val in enumerate(UNMASKING_CHANNEL_GROUPS):
            if val[1] == "DW_static":
                output_channels[i] = 1

        input_channels = [0] * len(UNMASKING_CHANNEL_GROUPS)
        for i, val in enumerate(UNMASKING_CHANNEL_GROUPS):
            if val[1] in ["S2_RGB", "S2_SWIR", "S2_NIR"]:
                input_channels[i] = 1

        self.condition = {
            "hw": 64 // patch_size,
            "patch_size": patch_size,
            "timesteps": 1,  # WE CURRENTLY ARE NOT TRAINING WITH THIS, PROBABLY HURTS PERF
            "input_channels": torch.Tensor(input_channels).to(device),
            "output_channels": torch.Tensor(output_channels).to(device),
            "target_exit_after": 0,
        }

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
            test_dl = DataLoader(
                EuroSatDataset(
                    normalizer=self.normalizer,
                    rgb=self.rgb,
                    include_latlons=self.include_latlons,
                    split="val",
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
                (
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    _,
                ) = pretrained_model(
                    s_t_x=s_t_x,
                    sp_x=sp_x,
                    t_x=t_x,
                    st_x=st_x,
                    s_t_m=s_t_m,
                    sp_m=sp_m,
                    t_m=t_m,
                    st_m=st_m,
                    months=months,
                    c_i=c_i,
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
            train_dl = DataLoader(
                EuroSatDataset(
                    normalizer=self.normalizer,
                    rgb=self.rgb,
                    include_latlons=self.include_latlons,
                    split="train",
                    merge_train_val=True,
                ),
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
