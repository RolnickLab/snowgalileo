from typing import Dict, List, Optional, Sequence, Tuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import torch.multiprocessing
from einops import repeat
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as PyTorchDataset
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
from ..flexipresto import Encoder
from ..masking import MaskedOutput
from ..utils import DEFAULT_SEED, data_dir, device, masked_output_np_to_tensor
from .eval import EvalTask, Hyperparams, model_class_name

### SETUP
torch.multiprocessing.set_sharing_strategy("file_system")

Masked_Output_and_Label = Tuple[MaskedOutput, torch.Tensor]


class PastisPixelDataset(PyTorchDataset):
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

    input_height_width = 1

    def __init__(
        self,
        folds: List[int] = [1, 2, 3, 4, 5],
        data_path: Optional[str] = "pastis/PASTIS-R_PixelSet",
        n_pixels_per_parcel: Optional[int] = 32,
        ignore_label: Optional[int] = None,
    ):
        """
        Dataset class to load PASTIS pixel-level data.
        Inspiration: https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader_pixelset.py.

        Args:
            folds: List of numbers specifying which of the 5 official folds to load.
            data_path: Relative path to the data folder starting from the default data path.
            n_pixels_per_parcel: Number of pixels randomly sampled from each parcel.
            ignore_label: If not None, the parcels annotated with this label are removed from the dataset.

        """
        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)
        self.n_pixels_per_parcel = n_pixels_per_parcel

        self.data_path = data_path

        self.meta = pd.read_csv(data_dir / cast(str, self.data_path) / "metadata_parcel.csv")
        self.meta.index = self.meta["ID_PARCEL"].astype(int)
        # multiple parcels form patches. We need the patch metadata to load the correct dates
        self.meta_patch = gpd.read_file(data_dir / cast(str, self.data_path) / "metadata.geojson")
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        if folds is not [1, 2, 3, 4, 5]:
            self.meta = pd.concat([self.meta[self.meta["Fold"] == f] for f in folds])
        if ignore_label is not None:
            self.meta = self.meta[self.meta["Label"] != ignore_label]

        self.meta.sort_index(inplace=True)

        self.id_parcels = self.meta.index
        self.labels = self.meta["Label"].to_dict()
        self.id_patches = self.meta["ID_PATCH"].to_dict()

        self.input_height_width = 1
        self.num_timesteps = 12

        (
            self.s_t_x,
            self.s_x,
            self.t_x,
            self.s_t_m,
            self.s_m,
            self.t_m,
            self.months,
            self.labels,
        ) = self.get_and_cache_data()
        # remove pixels that are masked out entirely to avoid NaNs during prediction
        pixel_mask = (
            np.any(~self.s_t_m.astype(bool), axis=(1, 2, 3, 4))
            | np.any(~self.s_m.astype(bool), axis=(1, 2, 3))
            | np.any(~self.t_m.astype(bool), axis=(1, 2))
        )
        self.s_t_x = self.s_t_x[pixel_mask]
        self.s_x = self.s_x[pixel_mask]
        self.t_x = self.t_x[pixel_mask]
        self.s_t_m = self.s_t_m[pixel_mask]
        self.s_m = self.s_m[pixel_mask]
        self.t_m = self.t_m[pixel_mask]
        self.months = self.months[pixel_mask]
        self.labels = self.labels[pixel_mask]

        self.len = self.s_t_x.shape[0]

    def create_pastis_masks(
        self, missing_timestep_indeces: np.ndarray, pixel_mask: np.ndarray
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
            "d -> n h w t d",
            n=self.n_pixels_per_parcel,
            h=self.input_height_width,
            w=self.input_height_width,
            t=self.num_timesteps,
        )

        # mask missing pixels
        s_t_m[pixel_mask == 1] = 1
        # mask missing timesteps
        s_t_m[:, :, :, missing_timestep_indeces, :] = 1

        # no space only / time only channels are available
        s_m = np.ones(
            [
                s_t_m.shape[0],
                self.input_height_width,
                self.input_height_width,
                len(SPACE_BAND_GROUPS_IDX),
            ]
        )
        t_m = np.ones([s_t_m.shape[0], self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (s_m == 1).all()
        assert (t_m == 1).all()

        return (s_t_m, s_m, t_m)

    @staticmethod
    def repeat_pixel(pixels, n_pixels_per_parcel):
        """
        Repeats a pixel if the parcel has fewer pixels than n_pixel.
        """
        if pixels.shape[-1] < n_pixels_per_parcel:
            if pixels.shape[-1] == 0:
                x = torch.zeros((*pixels.shape[:2], n_pixels_per_parcel))
                pixel_mask = np.array([1 for _ in range(n_pixels_per_parcel)])
                pixel_mask[0] = 0
            else:
                npad = ((0, 0), (0, 0), (0, n_pixels_per_parcel - pixels.shape[-1]))
                x = np.pad(pixels, pad_width=npad, mode="edge")
                pixel_mask = np.array(
                    [0 for _ in range(pixels.shape[-1])]
                    + [1 for _ in range(pixels.shape[-1], n_pixels_per_parcel)]
                )
        else:
            x = pixels
            pixel_mask = np.array([0 for _ in range(n_pixels_per_parcel)])
        return x, pixel_mask

    @staticmethod
    def sample_pixels(pixels, n_pixels_per_parcel):
        """
        Random sampling of pixels within a parcel.
        """
        if pixels.shape[-1] > n_pixels_per_parcel:
            idx = np.random.choice(
                list(range(pixels.shape[-1])), size=n_pixels_per_parcel, replace=False
            )
            x = pixels[:, :, idx]
        else:
            x = pixels
        return x

    def average_over_month(
        self, s2: np.ndarray, months: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the month-wise mean of an input image, pixel- and channel-specific.
        Months without observations are filled with zeros.
        Expected data input shape: T x C x NR_PIXELS.
        Months are expected to be 0-indexed.
        """
        unique_months = np.unique(months)

        all_months = np.arange(self.num_timesteps)
        missing_timestep_indeces = np.where(~np.isin(all_months, unique_months))[0]

        # stack months and s2 indices to group by month
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

        averages_months_with_data = np.array([s2[idx].mean(axis=0) for idx in s2_idx_per_month])

        averages_all_months = np.zeros((self.num_timesteps, s2.shape[1], s2.shape[2]))

        # fill up with zeros if there are months without observations
        averages_all_months[unique_months] = averages_months_with_data

        return (
            averages_all_months,
            repeat(all_months, "m -> n m", n=self.n_pixels_per_parcel),
            missing_timestep_indeces,
        )

    def get_and_cache_data(self):
        """
        Preprocess and cache data pixel-wise.
        """
        # iterate through parcels
        for i in range(self.meta.shape[0]):
            id_parcel = self.id_parcels[i]
            id_patch = self.id_patches[id_parcel]

            # Shape of the data: T x C x NR_PIXELS
            s2 = np.load(
                data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id_parcel)
            ).astype(np.float32)

            # filter pixels with NaNs
            not_nan = ~np.any(np.isnan(s2), (0, 1))
            s2 = s2[:, :, not_nan]

            # Dates are stored patch-wise in format YYYYMMDD
            dates = self.meta_patch["dates-S2"][id_patch]
            months = (
                np.array([int(str(value)[4:6]) for _, value in dates.items()]) - 1
            )  # 0-indexed months
            assert all(0 <= month <= 11 for month in months)

            # bring to consistent number of pixels per parcel
            s2, pixel_mask = self.repeat_pixel(s2, self.n_pixels_per_parcel)
            s2 = self.sample_pixels(s2, self.n_pixels_per_parcel)

            s2, months, missing_timestep_indeces = self.average_over_month(s2, months)
            s2 = repeat(s2, "t c n -> n h w t c", h=1, w=1)

            s_t_x = np.zeros(
                (
                    s2.shape[0],
                    self.input_height_width,
                    self.input_height_width,
                    self.num_timesteps,
                    len(SPACE_TIME_BANDS),
                )
            )

            kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)]

            s_t_x[:, :, :, :, kept_dynamic_bands] = s2
            s_t_x = normalize_space_time(s_t_x)
            # space only / time only bands are not provided by pastis
            s_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], s_t_x.shape[2], len(SPACE_BANDS)))
            t_x = np.zeros((s_t_x.shape[0], s_t_x.shape[3], len(TIME_BANDS)))

            s_t_m, s_m, t_m = self.create_pastis_masks(
                missing_timestep_indeces=missing_timestep_indeces,
                pixel_mask=pixel_mask,
            )

            label = repeat(
                np.array([self.labels[id_parcel] - 1], dtype=int), "l -> n l", n=s_t_x.shape[0]
            )  # 0-indexed

            num_pixels = self.meta.shape[0] * self.n_pixels_per_parcel

            if i == 0:
                s_t_x_cache = np.zeros(
                    (
                        num_pixels,
                        s_t_x.shape[1],
                        s_t_x.shape[2],
                        s_t_x.shape[3],
                        s_t_x.shape[4],
                    )
                )
                s_x_cache = np.zeros((num_pixels, s_x.shape[1], s_x.shape[2], s_x.shape[3]))
                t_x_cache = np.zeros((num_pixels, t_x.shape[1], t_x.shape[2]))
                s_t_m_cache = np.zeros(
                    (
                        num_pixels,
                        s_t_m.shape[1],
                        s_t_m.shape[2],
                        s_t_m.shape[3],
                        s_t_m.shape[4],
                    )
                )
                s_m_cache = np.zeros((num_pixels, s_m.shape[1], s_m.shape[2], s_m.shape[3]))
                t_m_cache = np.zeros((num_pixels, t_m.shape[1], t_m.shape[2]))
                months_cache = np.zeros((num_pixels, months.shape[1]))
                label_cache = np.zeros((num_pixels, 1))

            # fill arrays with parcel-wise data
            s_t_x_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = s_t_x
            s_x_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = s_x
            t_x_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = t_x
            s_t_m_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = s_t_m
            s_m_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = s_m
            t_m_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = t_m
            months_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = months
            label_cache[
                i * self.n_pixels_per_parcel : i * self.n_pixels_per_parcel
                + self.n_pixels_per_parcel
            ] = label

        return (
            s_t_x_cache,
            s_x_cache,
            t_x_cache,
            s_t_m_cache,
            s_m_cache,
            t_m_cache,
            months_cache,
            label_cache,
        )

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        return (
            masked_output_np_to_tensor(
                self.s_t_x[idx],
                self.s_x[idx],
                self.t_x[idx],
                self.s_t_m[idx],
                self.s_m[idx],
                self.t_m[idx],
                self.months[idx],
            ),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )

    def __len__(self):
        return self.len


class PastisPixelEval(EvalTask):
    name = "pastis_pixel"
    regression = False
    multilabel = False
    segmentation = False
    input_height_width = PastisPixelDataset.input_height_width

    def __init__(
        self,
        average_months: bool = True,
        patch_size: int = 1,
        seed=DEFAULT_SEED,
        num_outputs=len(PastisPixelDataset.labels_to_int),
    ):
        self.average_months = average_months
        super().__init__(patch_size, seed, num_outputs)
        self.name = f"{self.name}_{'AVERAGED_MONTHS' if self.average_months else 'ALL_MONTHS'}"

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        return {
            f"{self.name}: {model_name}_accuracy_score": accuracy_score(target, preds),
        }

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator]
    ) -> Dict:
        test_dl = DataLoader(
            PastisPixelDataset(folds=[1]),
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
            PastisPixelDataset(folds=[2, 3, 4, 5]),
            batch_size=Hyperparams.batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )
        trained_sklearn_models = self.train_sklearn_model(train_dl, pretrained_model, model_modes)
        return self._evaluate_model(pretrained_model, trained_sklearn_models)
