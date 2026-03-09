import random
from typing import Dict, Union

import psutil
from torch.utils.data import Subset

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Normalizer
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import config_dir, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class DatasetSizeAblationsEval(LandsatEval):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        exclude_prediction_era5: bool = False,
        h5pys_only: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = {},
        job_id="",
    ):
        super().__init__(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            h5pys_only=h5pys_only,
            num_finetune_epochs=num_finetune_epochs,
            decoder_mode=decoder_mode,
            eval_config=eval_config,
            job_id=job_id,
        )

    def _get_dataset(
        self,
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        exclude_prediction_sensors: bool,
        exclude_prediction_era5: bool,
        split: str,
        augmentation,
        h5pys_only: bool = False,
        data_config: Dict = {},
        normalization: Union[str, Normalizer] = "std",
    ):
        dataset = LandsatEvalDataset(
            augmentation=augmentation,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            split=split,
            h5pys_only=h5pys_only,
            data_config=data_config,
        )

        if normalization == "std":
            normalizing_dict = dataset.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        else:
            normalizer = Normalizer(std=False)
        dataset.normalizer = normalizer

        if data_config["dataset_subset_size"] > 0 and split == "train":
            indices = random.sample(range(len(dataset)), data_config["dataset_subset_size"])
            subset_dataset = Subset(dataset, indices)

        return subset_dataset
