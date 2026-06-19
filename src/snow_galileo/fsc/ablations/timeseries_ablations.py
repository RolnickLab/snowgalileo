from typing import Dict, Union

import psutil

from snow_galileo.config import DEFAULT_SEED
from snow_galileo.data.config import NORMALIZATION_DICT_FILENAME
from snow_galileo.data.dataset import Normalizer
from snow_galileo.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from snow_galileo.utils import config_dir, masked_output_np_to_tensor, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class TimeseriesAblationsMetaDataset(LandsatEvalDataset):
    def __init__(
        self,
        augmentation,
        data_config={},
        split="train",
        h5pys_only=False,
        eval_config=None,
        exclude_prediction_date=False,
        exclude_prediction_high_res=False,
        exclude_prediction_sensors=False,
        exclude_prediction_era5=False,
    ):
        super().__init__(
            data_config=data_config,
            split=split,
            h5pys_only=h5pys_only,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            augmentation=augmentation,
        )
        self.eval_config = eval_config
        assert self.eval_config is not None, (
            "eval_config must be provided for timeseries ablations"
        )
        assert "timeseries_ablations" in self.eval_config, "timeseries_ablations config missing"

    def __getitem__(self, idx):
        (
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
            ),
            label,
            filename,
        ) = super().__getitem__(idx)

        if self.eval_config["timeseries_ablations"]:
            s_t_h_m[:, :, :-1, :] = 1
            s_t_l_m[:, :, :-1, :] = 1
            s_t_m_m[:, :, :-1, :] = 1
            t_m[:-1, :] = 1

        return (
            masked_output_np_to_tensor(
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
            ),
            label,
            filename,
        )


class TimeseriesAblationsEval(LandsatEval):
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
        ds = TimeseriesAblationsMetaDataset(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            split=split,
            h5pys_only=h5pys_only,
            augmentation=augmentation,
            data_config=data_config,
            eval_config=self.eval_config,
        )

        if normalization == "std":
            normalizing_dict = ds.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        else:
            normalizer = Normalizer(std=False)
        ds.normalizer = normalizer

        return ds
