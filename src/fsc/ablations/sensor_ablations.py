from typing import Dict

import psutil

from src.config import DEFAULT_SEED
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import masked_output_np_to_tensor, seed_everything
from src.data.dataset import Normalizer
from src.utils import config_dir
from src.data.config import NORMALIZATION_DICT_FILENAME
from typing import Union

seed_everything(DEFAULT_SEED)
process = psutil.Process()


class SensorAblationsMetaDataset(LandsatEvalDataset):
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
        assert self.eval_config is not None, "eval_config must be provided for sensor ablations"
        assert "sensor_ablations" in self.eval_config, "sensor_ablations config missing"

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

        # TODO: make dynamic
        # ablate Sentinel-1
        if self.eval_config["sensor_ablations"]["ablate_high_res_sar"]:
            s_t_h_m[:, :, :, 0] = 1
        # ablate Sentinel-2, Landsat data
        if self.eval_config["sensor_ablations"]["ablate_high_res_optical"]:
            s_t_h_m[:, :, :, 1:] = 1
        # ablate Sentinel-3 data
        if self.eval_config["sensor_ablations"]["ablate_med_res_sensor"]:
            s_t_m_m[:, :, :, :] = 1
        # ablate MODIS, VIIRS data
        if self.eval_config["sensor_ablations"]["ablate_low_res_sensor"]:
            s_t_l_m[:, :, :, :-2] = 1
            t_m[:, :3] = 1
        # ablate indeces
        if self.eval_config["sensor_ablations"]["ablate_indeces"]:
            s_t_l_m[:, :, :, -2:] = 1
        # ablate ERA5 data
        if self.eval_config["sensor_ablations"]["ablate_era5"]:
            t_m[:, 3:] = 1
        # ablate topography
        if self.eval_config["sensor_ablations"]["ablate_topography"]:
            sp_m[:, :, 0] = 1
        # ablate landcover
        if self.eval_config["sensor_ablations"]["ablate_landcover"]:
            sp_m[:, :, 1] = 1

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


class SensorAblationsEval(LandsatEval):
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
        job_id = ""
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
        )

    def _get_dataset(
        self,
        augmentation,
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        exclude_prediction_sensors: bool,
        exclude_prediction_era5: bool,
        split: str,
        h5pys_only: bool = False,
        data_config: Dict = {},
        normalization: Union[str, Normalizer] = "std",
    ) -> SensorAblationsMetaDataset:
        ds = SensorAblationsMetaDataset(
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
