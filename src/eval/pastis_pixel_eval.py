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
from torch.nn import functional as F
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
from .eval import EvalTask, Hyperparams

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
            average_s2_over_month: Whether to average the Sentinel-2 data over months.
            n_pixels_per_parcel: Number of pixels randomly sampled from each parcel.
            ignore_label: If not None, the parcels annotated with this label are removed from the dataset.
        
        """
        self.folds = folds
        assert all(fold in [1, 2, 3, 4, 5] for fold in self.folds)
        self.n_pixels_per_parcel = n_pixels_per_parcel

        self.data_path = data_path

        self.meta = pd.read_csv(data_dir / cast(str, self.data_path) / "metadata_parcel.csv")
        self.meta.index = self.meta["ID_PARCEL"].astype(int)
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

        self.data, self.labels = self.get_and_cache_data()
        print(f"Cached data shape: {self.data.shape}")
        self.len = self.data.shape[0]

    def create_pastis_masks(
        self, missing_timestep_indeces: np.ndarray
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

        # mask missing timesteps
        s_t_m[:, :, missing_timestep_indeces, :] = 1

        # no space only / time only channels are available
        s_m = np.ones(
            [self.input_height_width, self.input_height_width, len(SPACE_BAND_GROUPS_IDX)]
        )
        t_m = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])

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
                pixel_mask = np.array([0 for _ in range(n_pixels_per_parcel)])
                pixel_mask[0] = 1
            else:
                x = F.pad(pixels, [0, n_pixels_per_parcel - pixels.shape[-1]], mode="replicate")
                pixel_mask = np.array(
                    [1 for _ in range(pixels.shape[-1])]
                    + [0 for _ in range(pixels.shape[-1], n_pixels_per_parcel)]
                )
        else:
            x = pixels
            pixel_mask = np.array([1 for _ in range(n_pixels_per_parcel)])
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

        return averages_all_months, all_months, missing_timestep_indeces


    def get_eo_array_and_masks(
        self, id_parcel: int, id_patch: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the image for a given ID, handles missing timesteps and normalizes the data.
        Also provides static and month data, and creates masks for missing data.
        """
        
        # Shape of the data: T x C x NR_PIXELS
        s2 = np.load(data_dir / cast(str, self.data_path) / "DATA_S2/S2_{}.npy".format(id_parcel)).astype(
            np.float32
        )

        print(f"Data shape after loading: {s2.shape}")

        s2, mask = self.repeat_pixel(s2, self.n_pixels_per_parcel)
        s2 = self.sample_pixels(s2, self.n_pixels_per_parcel)

        print(f"Data shape after repeat: {s2.shape}")

        dates = self.meta_patch["dates-S2"][id_patch]
        # the dates are in the format YYYYMMDD
        months = (
            np.array([int(str(value)[4:6]) for _, value in dates.items()]) - 1
        )  # 0-indexed months
        assert all(0 <= month <= 11 for month in months)

        s2, months, missing_timestep_indeces = self.average_over_month(s2, months)

        print(f"Data shape after month average: {s2.shape}") 

        kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS)]

        s_t_x = np.zeros(
            [
                self.input_height_width,
                self.input_height_width,
                self.num_timesteps,
                len(SPACE_TIME_BANDS),
            ]
        )
        s_t_x[:, :, kept_dynamic_bands] = repeat(s2, "t c h w -> h w t c")

        # space only / time only bands are not provided by pastis
        s_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))

        s_t_m, s_m, t_m = self.create_pastis_masks(
            missing_timestep_indeces=missing_timestep_indeces
        )

        return normalize_space_time(s_t_x), s_x, t_x, s_t_m, s_m, t_m, months
    
    def get_and_cache_data(self):

        print("Number of parcels: ", self.meta.shape[0])

        for i in range(self.meta.shape[0]):
            id_parcel = self.id_parcels[i]
            id_patch = self.id_patches[id_parcel]
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = self.get_eo_array_and_masks(id_parcel, id_patch)
            label = torch.from_numpy(np.array(self.labels[id_parcel] - 1, dtype=int)) # 0-indexed

            if i == 0:
                s_t_x_cache = np.zeros((self.meta.shape[0], s_t_x.shape[0], s_t_x.shape[1], s_t_x.shape[2], s_t_x.shape[3]))
                s_x_cache = np.zeros((self.meta.shape[0], s_x.shape[0], s_x.shape[1], s_x.shape[2]))
                t_x_cache = np.zeros((self.meta.shape[0], t_x.shape[0], t_x.shape[1]))
                s_t_m_cache = np.zeros((self.meta.shape[0], s_t_m.shape[0], s_t_m.shape[1], s_t_m.shape[2], s_t_m.shape[3]))
                s_m_cache = np.zeros((self.meta.shape[0], s_m.shape[0], s_m.shape[1], s_m.shape[2]))
                t_m_cache = np.zeros((self.meta.shape[0], t_m.shape[0], t_m.shape[1]))
                months_cache = np.zeros((self.meta.shape[0], months.shape[0]))
                label_cache = np.zeros((self.meta.shape[0], 1))

            s_t_x_cache[i] = s_t_x
            s_x_cache[i] = s_x
            t_x_cache[i] = t_x
            s_t_m_cache[i] = s_t_m
            s_m_cache[i] = s_m
            t_m_cache[i] = t_m
            months_cache[i] = months
            label_cache[i] = label
        return np.stack([s_t_x_cache, s_x_cache, t_x_cache, s_t_m_cache, s_m_cache, t_m_cache, months_cache, label_cache], axis=0)


    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:

        id_parcel = self.id_parcels[idx]
        id_patch = self.id_patches[id_parcel]

        if not self.cache or idx not in self.memory.keys():
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = self.get_eo_array_and_masks(id_parcel, id_patch)
            label = torch.from_numpy(np.array(self.labels[id_parcel] - 1, dtype=int)) # 0-indexed
        
            if self.cache:
                self.memory[idx] = (s_t_x, s_x, t_x, s_t_m, s_m, t_m, months)

        else:
            s_t_x, s_x, t_x, s_t_m, s_m, t_m, months = self.memory[idx]
            label = torch.from_numpy(np.array(self.labels[id_parcel] - 1, dtype=int)) # 0-indexed

        return (
            masked_output_np_to_tensor(
                s_t_x,
                s_x,
                t_x,
                s_t_m,
                s_m,
                t_m,
                months,
            ),
            label,
        )

    def __len__(self):
        return self.len
    
ds = PastisPixelDataset()
b, l = ds[0]

"""
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
"""