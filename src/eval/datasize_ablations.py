from typing import Dict

import psutil
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

from src.config import DEFAULT_SEED
from src.eval.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import seed_everything

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
        h5pys_only: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = {},
    ):
        super().__init__(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            h5pys_only=h5pys_only,
            num_finetune_epochs=num_finetune_epochs,
            decoder_mode=decoder_mode,
            eval_config=eval_config,
        )

    def _get_dataset(
        self,
        augmentation,
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        split: str,
        h5pys_only: bool = False,
        data_config: Dict = {},
    ) -> LandsatEvalDataset:
        dataset = LandsatEvalDataset(
            augmentation=augmentation,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            split=split,
            h5pys_only=h5pys_only,
            data_config=data_config,
            eval_config=self.eval_config,
        )

        if data_config["dataset_subset_size"] > 0:
            subset_size = data_config["dataset_subset_size"]

            labels = [dataset[i][1] for i in range(len(dataset))]

            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=subset_size,
                random_state=42,
            )
            indices, _ = next(splitter.split(range(len(dataset)), labels))

            dataset = Subset(dataset, indices)

        return dataset
