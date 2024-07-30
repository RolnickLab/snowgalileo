import calendar
import hashlib
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.config import DATA_FOLDER, EXPORTED_HEIGHT_WIDTH_METRES
from src.data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    TIME_BAND_GROUPS_IDX,
    Dataset,
)
from src.data.earthengine.eo import EE_BUCKET_TIFS, EarthEngineExporter, EEBoundingBox
from src.eval.eval import EvalTask, Hyperparams, model_class_name
from src.flexipresto import Encoder, FineTuningModel
from src.utils import DEFAULT_SEED, device, masked_output_np_to_tensor

LFMC_FOLDER = DATA_FOLDER / "lfmc"
LABELS_PATH = LFMC_FOLDER / "LFMC_post_2017.csv"
LOCAL_LFMC_TIFS_FOLDER = LFMC_FOLDER / "tifs"
LFMC_CACHE_FOLDER = LFMC_FOLDER / "h5pys"
GCLOUD_LFMC_TIFS_FOLDER = "lfmc"
RAW_PROJECTION = "EPSG:3857"
LATLON_PROJECTION = "EPSG:4326"
END_PADDING = 30

# the dataset itself has a huge range of labels (up to 599,999 ???)
# but the vast majority of labels (98.78 %) are less than 400. We
# will clip at 300 and normalize
LABEL_MAX = 300

SPLIT_TO_HEX = {
    0: ["0", "1", "2", "3"],
    1: ["4", "5", "6", "7"],
    2: ["8", "9", "a", "b"],
    3: ["c", "d", "e", "f"],
}


def read_labels(data_path: Path = LABELS_PATH) -> pd.DataFrame:
    """
    For remote sensing applications, it is recommended to average the LFMC measurements taken on
    the same date and located within the same pixel of the product employed in the study.
    The choice of which functional type to include in the average can be guided by the land
    cover type of that pixel. For example, in open canopy forests, both trees and
    shrubs (or grass) could be included.
    """
    data = pd.read_csv(data_path)
    grouped = data.groupby(
        [
            "Latitude (WGS84, EPSG:4326)",
            "Longitude (WGS84, EPSG:4326)",
            "Sampling date (YYYYMMDD)",
        ],
        as_index=False,
    ).agg(
        {
            "Site name": "first",
            "Sorting ID": "first",
            "LFMC value (%)": "mean",
            "State/Region": "first",
        }
    )
    return grouped


def check_site_name_in_split_mode(site_name: str, mode: str, split_id: int) -> bool:
    is_val = hashlib.sha256(site_name.encode()).hexdigest()[0] in SPLIT_TO_HEX[split_id]
    if is_val:
        return mode == "val"
    else:
        return mode == "train"


class LFMCExporter(EarthEngineExporter):
    def __init__(
        self,
        check_ee: bool = False,
        check_gcp: bool = False,
        credentials=None,
        mode: str = "batch",
    ) -> None:
        super().__init__(
            dest_bucket=EE_BUCKET_TIFS,
            check_ee=check_ee,
            check_gcp=check_gcp,
            credentials=credentials,
            mode=mode,
            local_tifs_folder=LOCAL_LFMC_TIFS_FOLDER,
            gcloud_tifs_folder=GCLOUD_LFMC_TIFS_FOLDER,
        )

    @staticmethod
    def pad_dates(end_date: date, end_padding: int) -> Tuple[date, date]:
        new_end_date = end_date + timedelta(days=end_padding)
        last_day_of_month = calendar.monthrange(new_end_date.year, new_end_date.month)[1]
        new_end_date = date(new_end_date.year, new_end_date.month, last_day_of_month)
        start_date = new_end_date - timedelta(days=365)
        start_date = date(start_date.year, start_date.month, 1)
        return start_date, new_end_date

    def export_lfmc_data(
        self,
        labels_path: Path = LABELS_PATH,
        end_padding: int = END_PADDING,
        num_exports_to_start: int = 3000,
        state_region: Optional[str] = "Idaho",
        surrounding_metres: int = EXPORTED_HEIGHT_WIDTH_METRES // 2,
    ) -> None:
        """
        Export boxes with length and width EXPORTED_HEIGHT_WIDTH_METRES
        for the points in latlons (where latlons is a dataframe with
        the columns "lat" and "lon")
        """

        data = read_labels(labels_path)
        if state_region is not None:
            data = data[(data["State/Region"] == state_region)]
        data["sampling_date"] = pd.to_datetime(data["Sampling date (YYYYMMDD)"]).dt.date

        exports_started = 0
        print(f"Exporting {len(data)} latlons: ")

        for _, row in tqdm(data.iterrows(), desc="Exporting", total=len(data)):
            lat = row["Latitude (WGS84, EPSG:4326)"]
            lon = row["Longitude (WGS84, EPSG:4326)"]
            ee_bbox = EEBoundingBox.from_centre(
                mid_lat=lat, mid_lon=lon, surrounding_metres=surrounding_metres
            )
            start_date, end_date = self.pad_dates(row["sampling_date"], end_padding=end_padding)

            export_started = self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=row["Sorting ID"],
                start_date=start_date,
                end_date=end_date,
            )
            if export_started:
                exports_started += 1
                if num_exports_to_start is not None and exports_started >= num_exports_to_start:
                    print(f"Started {exports_started} exports. Ending export")
                    return None
        if self.mode == "url":
            print("Export finished. Syncing to google cloud")
            self.sync_local_and_gcloud()
            print("Finished sync")


class LFMCDataset(Dataset):
    def __init__(
        self,
        data_folder: Path = LOCAL_LFMC_TIFS_FOLDER,
        labels_path: Path = LABELS_PATH,
        download: bool = False,
        output_hw: int = 32,
        num_timesteps: int = 12,
        h5py_folder: Path = LFMC_CACHE_FOLDER,
        mode: str = "train",
        split_id: int = 0,
        space_time_bands: Optional[List[str]] = None,
        space_bands: Optional[List[str]] = None,
        time_bands: Optional[List[str]] = None,
        static_bands: Optional[List[str]] = None,
        return_instance_weights: bool = False,
        end_padding: int = END_PADDING,
    ):
        # set npys_only to False since we use the tifs list quite a bit.
        # could be easily updated
        super().__init__(data_folder, download, h5py_folder, False)

        self.output_hw = output_hw
        self.num_timesteps = num_timesteps
        self.split_id = split_id
        self.mode = mode

        data = data = read_labels(labels_path)
        data["sampling_date"] = pd.to_datetime(data["Sampling date (YYYYMMDD)"]).dt.date

        self.tifs = []
        self.filepath_to_start_month = {}
        self.filepath_to_label = {}
        self.filepath_to_id = {}

        for _, row in tqdm(data.iterrows()):
            filepath = data_folder / f"{row['Sorting ID']}.tif"
            if check_site_name_in_split_mode(str(row["Site name"]), mode, split_id):
                if filepath.exists():
                    self.tifs.append(filepath)
                    start_date, _ = LFMCExporter.pad_dates(
                        row["sampling_date"], end_padding=end_padding
                    )
                    self.filepath_to_start_month[filepath.stem] = start_date.month
                    self.filepath_to_label[filepath.stem] = row["LFMC value (%)"]
                    self.filepath_to_id[filepath.stem] = row["Sorting ID"]

        if space_time_bands is None:
            space_time_bands = list(SPACE_TIME_BANDS_GROUPS_IDX.keys())
        if space_bands is None:
            space_bands = list(SPACE_BAND_GROUPS_IDX.keys())
        if time_bands is None:
            time_bands = list(TIME_BAND_GROUPS_IDX.keys())
        if static_bands is None:
            static_bands = list(STATIC_BAND_GROUPS_IDX.keys())
        self.masks = self.make_masks(space_time_bands, space_bands, time_bands, static_bands)
        self.return_instance_weights = return_instance_weights

    def month_array_from_file(self, tif_path: Path, num_timesteps: int) -> np.ndarray:
        """
        Given a filepath and num_timesteps, extract start_month and return an array of
        months where months[idx] is the month for list(range(num_timesteps))[i]
        """
        start_month = self.filepath_to_start_month[tif_path.stem]
        # >>> np.fmod(np.array([9., 10, 11, 12, 13, 14]), 12)
        # array([ 9., 10., 11.,  0.,  1.,  2.])
        # - 1 because we want to index from 0
        return np.fmod(np.arange(start_month - 1, start_month - 1 + num_timesteps), 12)

    @staticmethod
    def subset_image(
        space_time_x: np.ndarray,
        space_x: np.ndarray,
        time_x: np.ndarray,
        static_x: np.ndarray,
        months: np.ndarray,
        size: int,
        num_timesteps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Differs from the parent function because subsetting is always centered
        and time cuts always remove from the end
        """
        assert (space_time_x.shape[0] == space_x.shape[0]) & (
            space_time_x.shape[1] == space_x.shape[1]
        )
        assert space_time_x.shape[2] == time_x.shape[0]
        assert space_time_x.shape[0] >= size
        assert space_time_x.shape[1] >= size
        assert space_time_x.shape[2] >= num_timesteps
        start_h = int((space_time_x.shape[0] - size) / 2)
        start_w = int((space_time_x.shape[1] - size) / 2)
        assert (start_h >= 0) & (start_w >= 0)

        total_timesteps = space_time_x.shape[2]
        assert num_timesteps <= total_timesteps
        timesteps_to_sample = range(total_timesteps - num_timesteps, total_timesteps)

        return (
            space_time_x[start_h : start_h + size, start_w : start_w + size, timesteps_to_sample],
            space_x[start_h : start_h + size, start_w : start_w + size],
            time_x[timesteps_to_sample],
            static_x,
            months[timesteps_to_sample],
        )

    def make_masks(
        self,
        space_time_bands: List[str],
        space_bands: List[str],
        time_bands: List[str],
        static_bands: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        space_time_mask = np.ones(
            [self.output_hw, self.output_hw, self.num_timesteps, len(SPACE_TIME_BANDS_GROUPS_IDX)]
        )
        space_time_include = [
            idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if key in space_time_bands
        ]
        space_time_mask[:, :, :, space_time_include] = 0

        space_mask = np.ones([self.output_hw, self.output_hw, len(SPACE_BAND_GROUPS_IDX)])
        space_include = [
            idx for idx, key in enumerate(SPACE_BAND_GROUPS_IDX) if key in space_bands
        ]
        space_mask[:, :, space_include] = 0

        time_mask = np.ones([self.num_timesteps, len(TIME_BAND_GROUPS_IDX)])
        time_include = [idx for idx, key in enumerate(TIME_BAND_GROUPS_IDX) if key in time_bands]
        time_mask[:, time_include] = 0

        static_mask = np.ones([len(STATIC_BAND_GROUPS_IDX)])
        static_include = [
            idx for idx, key in enumerate(STATIC_BAND_GROUPS_IDX) if key in static_bands
        ]
        static_mask[static_include] = 0

        return space_time_mask, space_mask, time_mask, static_mask

    def __getitem__(self, idx):
        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            months,
        ) = self.load_tif(idx)
        (
            s_t_x,
            sp_x,
            t_x,
            st_x,
            months,
        ) = self.subset_image(s_t_x, sp_x, t_x, st_x, months, self.output_hw, self.num_timesteps)

        # get the label
        tif_path = self.tifs[idx]
        label = min(self.filepath_to_label[tif_path.stem], LABEL_MAX) / LABEL_MAX
        masks = self.masks
        return masked_output_np_to_tensor(s_t_x, sp_x, t_x, st_x, *masks, months), label

    def split(self):
        next_split = (self.split_id + 1) % len(SPLIT_TO_HEX)
        train_tifs, val_tifs = [], []

        for idx, tif in enumerate(self.tifs):
            sorting_id = self.filepath_to_id[self.tifs[idx].stem]
            if check_site_name_in_split_mode(str(sorting_id), "val", next_split):
                val_tifs.append(tif)
            else:
                train_tifs.append(tif)
        val_ds = deepcopy(self)
        self.tifs = train_tifs
        val_ds.tifs = val_tifs
        val_ds.split_id = next_split
        val_ds.mode = "val"
        return self, val_ds


class LFMCTask(EvalTask):
    name = "lfmc"
    num_classes = 1
    regression = True
    multilabel = False

    def __init__(
        self,
        patch_size: int = 8,
        seed: int = DEFAULT_SEED,
        output_hw: int = 32,
        num_timesteps: int = 12,
        split_id: int = 0,
        return_result_arrays: bool = False,
        space_time_bands: Optional[List[str]] = None,
        space_bands: Optional[List[str]] = None,
        time_bands: Optional[List[str]] = None,
        static_bands: Optional[List[str]] = None,
    ):
        super().__init__(patch_size, seed)
        self.output_hw = output_hw
        self.num_timesteps = num_timesteps
        self.split_id = split_id
        self.return_result_arrays = return_result_arrays
        self.name = f"{self.name}_hw{output_hw}_timesteps{num_timesteps}"
        if not return_result_arrays:
            self.name = f"{self.name}_split{split_id}"

        self.space_time_bands = space_time_bands
        self.space_bands = space_bands
        self.time_bands = time_bands
        self.static_bands = static_bands
        self.labels = LABELS_PATH

    @classmethod
    def _construct_finetuning_model(cls, model: Encoder) -> FineTuningModel:
        head = nn.Linear(model.embedding_size, cls.num_classes)
        finetuning_model = FineTuningModel(model, head).to(device)
        finetuning_model.train()
        return finetuning_model

    def compute_metrics(self, model_name: str, preds: np.ndarray, target: np.ndarray) -> Dict:
        results = {
            f"{self.name}: {model_name}_mse": mean_squared_error(
                target * LABEL_MAX, preds * LABEL_MAX
            )
        }
        if self.return_result_arrays:
            results[f"{self.name}: {model_name}"] = {
                "targets": target * LABEL_MAX,
                "preds": preds * LABEL_MAX,
            }
        return results

    @torch.no_grad()
    def _evaluate_model(
        self, pretrained_model: Encoder, sklearn_models: Sequence[BaseEstimator]
    ) -> Dict:
        test_dl = DataLoader(
            LFMCDataset(
                LOCAL_LFMC_TIFS_FOLDER,
                mode="val",
                split_id=self.split_id,
                output_hw=self.output_hw,
                num_timesteps=self.num_timesteps,
                space_time_bands=self.space_time_bands,
                space_bands=self.space_bands,
                time_bands=self.time_bands,
                static_bands=self.static_bands,
                labels_path=self.labels,
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
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, _ = pretrained_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
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

    def finetune_model(self, pretrained_model: Encoder) -> FineTuningModel:
        max_epochs: int = 100
        weight_decay: float = 0.05
        lr: float = 3e-4
        patience: int = 5
        batch_size: int = 64
        loss_fn = nn.MSELoss()

        train_ds, val_ds = LFMCDataset(
            LOCAL_LFMC_TIFS_FOLDER,
            mode="train",
            split_id=self.split_id,
            output_hw=self.output_hw,
            num_timesteps=self.num_timesteps,
            space_time_bands=self.space_time_bands,
            space_bands=self.space_bands,
            time_bands=self.time_bands,
            static_bands=self.static_bands,
            labels_path=self.labels,
        ).split()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Hyperparams.num_workers,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        model = self._construct_finetuning_model(pretrained_model)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loss = []
        val_loss = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        for _ in (pbar := tqdm(range(max_epochs), desc="Finetuning")):
            model.train()
            epoch_train_loss = 0.0
            for masked_output, label in tqdm(train_dl, desc="Training", leave=False):
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                    t.to(device) for t in masked_output
                ]
                optimizer.zero_grad()
                preds = model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )
                loss = loss_fn(preds, label.float().to(device))
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_train_loss / len(train_dl))

            model.eval()
            all_preds, all_labels = [], []
            for masked_output, label in val_dl:
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                    t.to(device) for t in masked_output
                ]
                with torch.no_grad():
                    preds = model(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months,
                        patch_size=self.patch_size,
                    )
                    all_preds.append(preds)
                    all_labels.append(label)

            val_loss.append(
                torch.mean(loss_fn(torch.cat(all_preds), torch.cat(all_labels).float().to(device)))
            )
            pbar.set_description(f"Train metric: {train_loss[-1]}, Val metric: {val_loss[-1]}")
            if best_loss is None:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
            else:
                if val_loss[-1] < best_loss:
                    best_loss = val_loss[-1]
                    best_model_dict = deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        print("Early stopping!")
                        break
        assert best_model_dict is not None
        model.load_state_dict(best_model_dict)

        model.eval()
        return model

    @torch.no_grad()
    def _evaluate_finetuned_model(self, finetuned_model: FineTuningModel):
        test_dl = DataLoader(
            LFMCDataset(
                LOCAL_LFMC_TIFS_FOLDER,
                mode="val",
                split_id=self.split_id,
                output_hw=self.output_hw,
                num_timesteps=self.num_timesteps,
                space_time_bands=self.space_time_bands,
                space_bands=self.space_bands,
                time_bands=self.time_bands,
                static_bands=self.static_bands,
                labels_path=self.labels,
            ),
            batch_size=Hyperparams.batch_size,
            shuffle=False,
            num_workers=Hyperparams.num_workers,
        )

        preds_list = []
        labels_list = []

        for masked_output, label in tqdm(test_dl, desc="Computing test predictions"):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [
                t.to(device) for t in masked_output
            ]

            finetuned_model.eval()

            with torch.no_grad():
                predictions = finetuned_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )

            labels_list.append(label.cpu().numpy())
            preds_list.append(predictions.cpu().numpy())

        target = np.concatenate(labels_list)
        test_preds_np = np.concatenate(preds_list, axis=0)
        return self.compute_metrics("finetuning", test_preds_np, target)

    def evaluate_model_on_task(
        self, pretrained_model: Encoder, model_modes: Optional[List[str]] = None
    ) -> Dict:
        if model_modes is None:
            model_modes = self.all_regression_sklearn_models + ["finetuning"]
        for model_mode in model_modes:
            assert model_mode in (self.all_regression_sklearn_models + ["finetuning"])

        results = {}
        sklearn_model_modes = [x for x in model_modes if x != "finetuning"]
        if len(sklearn_model_modes) > 0:
            train_dl = DataLoader(
                LFMCDataset(
                    LOCAL_LFMC_TIFS_FOLDER,
                    mode="train",
                    split_id=self.split_id,
                    output_hw=self.output_hw,
                    num_timesteps=self.num_timesteps,
                    space_time_bands=self.space_time_bands,
                    space_bands=self.space_bands,
                    time_bands=self.time_bands,
                    static_bands=self.static_bands,
                    labels_path=self.labels,
                ),
                batch_size=Hyperparams.batch_size,
                shuffle=True,
                num_workers=Hyperparams.num_workers,
            )
            trained_sklearn_models = self.train_sklearn_model(
                train_dl, pretrained_model, sklearn_model_modes
            )
            results.update(self._evaluate_model(pretrained_model, trained_sklearn_models))
        if "finetuning" in model_modes:
            finetuned_model = self.finetune_model(pretrained_model)
            results.update(self._evaluate_finetuned_model(finetuned_model))

        return results

    @classmethod
    def evaluate_model_on_task_for_all_splits(
        cls,
        pretrained_model: Encoder,
        model_modes: Optional[List[str]] = None,
        patch_size: int = 8,
        seed: int = DEFAULT_SEED,
        output_hw: int = 32,
        num_timesteps: int = 12,
        split_id: int = 0,
        space_time_bands: Optional[List[str]] = None,
        space_bands: Optional[List[str]] = None,
        time_bands: Optional[List[str]] = None,
        static_bands: Optional[List[str]] = None,
    ):
        aggregate_results: Dict[str, Dict] = {}
        for split_id in SPLIT_TO_HEX.keys():
            task = cls(
                patch_size=patch_size,
                seed=seed,
                output_hw=output_hw,
                num_timesteps=num_timesteps,
                space_time_bands=space_time_bands,
                space_bands=space_bands,
                time_bands=time_bands,
                static_bands=static_bands,
                split_id=split_id,
                return_result_arrays=True,
            )

            results = task.evaluate_model_on_task(pretrained_model, model_modes)
            for r_key, r_val in results.items():
                if isinstance(r_val, dict):
                    if r_key in aggregate_results:
                        aggregate_results[r_key]["preds"].append(r_val["preds"])
                        aggregate_results[r_key]["targets"].append(r_val["targets"])
                    else:
                        aggregate_results[r_key] = {
                            "preds": [r_val["preds"]],
                            "targets": [r_val["targets"]],
                        }

        return {
            f"{key}_mse": mean_squared_error(
                np.concatenate(val["targets"], axis=-1),
                np.concatenate(val["preds"], axis=-1),
            )
            for key, val in aggregate_results.items()
        }
