from math import sqrt
from typing import Dict, List, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch.multiprocessing
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
from torchmetrics import JaccardIndex
from tqdm import tqdm

from ..data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space_time,
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

    def __init__(
        self,
        folds: List[int],
        data_path: Optional[str] = "pastis/PASTIS-R",
        num_subtiles_per_image: Optional[int] = 4,
        average_s2_over_month: Optional[bool] = True,
    ):
        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)

        self.data_path = data_path

        self.metadata = gpd.read_file(data_dir / cast(str, self.data_path) / "metadata.geojson")
        self.metadata.index = self.metadata["ID_PATCH"].astype(int)
        self.metadata.sort_index(inplace=True)

        self.metadata = pd.concat([self.metadata[self.metadata["Fold"] == f] for f in folds])

        self.id = self.metadata.index

        # pastis comes in large images, we split them into subtiles
        # must be a square number
        self.num_subtiles_per_image = num_subtiles_per_image
        assert sqrt(cast(float, self.num_subtiles_per_image)).is_integer()

        self.average_s2_over_month = average_s2_over_month

        if average_s2_over_month:
            self.num_timesteps = 12
        else:
            # max number of timesteps in PASTIS
            self.num_timesteps = 61

    def create_pastis_masks(
        self, timesteps_with_data: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Masks unavailable channels and timesteps.
        """
        s_t_channels = [idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key]

        # everything is masked by default
        s_t_m = np.ones([len(SPACE_TIME_BANDS_GROUPS_IDX)])
        # unmask available bands
        s_t_m[s_t_channels] = 0
        s_t_m = repeat(
            s_t_m,
            "d -> h w t d",
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # mask padded timesteps if there are any
        s_t_m[:, :, timesteps_with_data:, :] = 1

        # no space only / time only channels are available
        s_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (s_m == 1).all()
        assert (t_m == 1).all()

        return (s_t_m, s_m, t_m)

    def average_over_month(
        self, s2: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the month-wise mean of an input image, pixel- and channel-specific.
        """

        # data comes in shape T x C x H x W
        s2_idx = np.arange(s2.shape[0])
        stacked_months_and_s2_idx = np.column_stack((months, s2_idx))

        # group observations by sorted month https://stackoverflow.com/questions/38013778/is-there-any-numpy-group-by-function
        stacked_months_and_s2_idx = stacked_months_and_s2_idx[
            stacked_months_and_s2_idx[:, 0].argsort()
        ]
        s2_idx_per_month = np.split(
            stacked_months_and_s2_idx[:, 1],
            np.unique(stacked_months_and_s2_idx[:, 0], return_index=True)[1][1:],
        )

        averages = np.array([s2[idx].mean(axis=0) for idx in s2_idx_per_month])

        # rearrange months to match the order of the averages
        months = np.unique(months).argsort()

        timesteps_with_data = months.shape[0]

        # fill up with zeros if there are months without observations
        averages = np.concatenate(
            [
                averages,
                np.zeros(
                    (
                        self.num_timesteps - timesteps_with_data,
                        s2.shape[1],
                        s2.shape[2],
                        s2.shape[3],
                    )
                ),
            ],
            axis=0,
        )
        months = np.concatenate(
            [months, np.zeros(self.num_timesteps - timesteps_with_data)], axis=0
        )

        return averages, months

    def zero_pad_missing_timesteps(
        self, s2: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Pads input image and months with zeros to reach the maximum number of timesteps available in PASTIS.
        """
        # data comes in shape T x C x H x W
        timesteps_with_data = s2.shape[0]

        s2 = np.concatenate(
            [
                s2,
                np.zeros(
                    (
                        self.num_timesteps - timesteps_with_data,
                        s2.shape[1],
                        s2.shape[2],
                        s2.shape[3],
                    )
                ),
            ],
            axis=0,
        )
        months = np.concatenate(
            [months, np.zeros(self.num_timesteps - timesteps_with_data)], axis=0
        )

        return s2, months, timesteps_with_data

    def get_eo_array_and_masks(
        self, id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the image for a given ID, handles missing timesteps and normalizes the data.
        Also provides static and month data, and creates masks for missing data.
        """
        s2 = np.load(data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id)).astype(
            np.float32
        )

        dates = self.metadata["dates-S2"][id]
        # the dates are in the format YYYYMMDD
        months = (
            np.array([int(str(value)[4:6]) for _, value in dates.items()]) - 1
        )  # 0-indexed months
        assert all(0 <= month <= 11 for month in months)

        if self.average_s2_over_month:
            s2, months = self.average_over_month(s2, months)
            timesteps_with_data = self.num_timesteps

        # pad missing timesteps, will be masked out later
        else:
            s2, months, timesteps_with_data = self.zero_pad_missing_timesteps(s2, months)

        kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)]

        s_t_x = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(SPACE_TIME_BANDS),
            ]
        )
        s_t_x[:, :, :, kept_dynamic_bands] = repeat(s2, "t c h w -> h w t c")

        # space only / time only bands are not provided by pastis
        s_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        s_t_m, s_m, t_m = self.create_pastis_masks(timesteps_with_data=timesteps_with_data)

        return normalize_space_time(s_t_x), s_x, t_x, s_t_m, s_m, t_m, months

    def get_target(self, id: int) -> torch.Tensor:
        target = np.load(
            data_dir / cast(str, self.data_path) / "ANNOTATIONS/TARGET_{}.npy".format(id)
        )
        return torch.from_numpy(target[0].astype(int)).long()

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        """
        Slices and returns a subtile of the image and the corresponding target.
        """
        img_idx = idx // self.num_subtiles_per_image

        id = self.id[img_idx]

        s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = self.get_eo_array_and_masks(id)
        target = self.get_target(id)

        subtiles_per_dim = int(sqrt(cast(float, self.num_subtiles_per_image)))
        h, w = s_t_x.shape[:2]
        assert h == w  # this is the case for PASTIS
        assert h % subtiles_per_dim == 0
        pixels_per_dim = h // subtiles_per_dim
        subtile_idx = idx % self.num_subtiles_per_image

        row_idx = subtile_idx // subtiles_per_dim
        col_idx = subtile_idx % subtiles_per_dim

        return (
            masked_output_np_to_tensor(
                s_t_x[
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
                t_x,
                s_t_m[
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
                t_m,
                months,
            ),
            target[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ],
        )

    def __len__(self):
        return self.metadata.shape[0] * self.num_subtiles_per_image


class PastisEval(EvalTask):
    name = "pastis"
    regression = False
    multilabel = False
    segmentation = True
    num_outputs = len(PastisDataset.labels_to_int)
    input_height_width = PastisDataset.input_height_width

    def __init__(
        self,
        average_months: bool = True,
        num_subtiles_per_image: int = 4,
        patch_size: int = 8,
        seed=DEFAULT_SEED,
    ):
        self.average_months = average_months
        self.num_subtiles_per_image = num_subtiles_per_image
        super().__init__(patch_size, seed)
        self.input_height_width = self.input_height_width // int(
            sqrt(cast(float, self.num_subtiles_per_image))
        )
        self.name = f"{self.name}_{'AVERAGED_MONTHS' if self.average_months else 'ALL_MONTHS'}_hw{self.input_height_width}"

    @torch.no_grad()
    def _evaluate_model(self, finetuned_model: PrestoFineTuningModel) -> Dict:
        test_dl = DataLoader(
            PastisDataset(
                folds=[1],
                average_s2_over_month=self.average_months,
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        mean_IoU = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = [t.to(device) for t in masked_output]
            label = label.to(device)

            jaccard_mean = JaccardIndex(task="multiclass", num_classes=self.num_outputs).to(device)

            finetuned_model.eval()

            with torch.no_grad():
                preds = finetuned_model(
                    s_t_x, s_x, t_x, s_t_m, s_m, t_m, months, patch_size=self.patch_size
                )
                mean_IoU.append(jaccard_mean(preds, label).item())

        return {f"{self.name}: finetuned_mean_iou": np.mean(mean_IoU)}

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = ["finetune"]
        for model_mode in model_modes:
            assert model_mode in ["finetune"]

        train_dl = DataLoader(
            PastisDataset(
                folds=[3, 4, 5],
                average_s2_over_month=self.average_months,
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )

        val_dl = DataLoader(
            PastisDataset(
                folds=[2],
                average_s2_over_month=self.average_months,
                num_subtiles_per_image=self.num_subtiles_per_image,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        finetuned_model = self.finetune_presto(train_dl, val_dl, pretrained_model)
        return self._evaluate_model(finetuned_model)
