from typing import Dict

import psutil

from src.config import DEFAULT_SEED
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import masked_output_np_to_tensor, seed_everything
from src.data.dataset import Normalizer
from src.utils import config_dir
from src.data.config import (
    NORMALIZATION_DICT_FILENAME,
    DATASET_OUTPUT_HW_HIGH_RES,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_LOW_RES_PIXELS_PER_DIM,
)
from typing import Union
import numpy as np


seed_everything(DEFAULT_SEED)
process = psutil.Process()


class PixelAblationsMetaDataset(LandsatEvalDataset):
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
        assert self.eval_config is not None, "eval_config must be provided for pixel ablations"
        assert "pixel_ablations" in self.eval_config, "pixel_ablations config missing"

        self.cum_pixels = [0]
        for _ in range(super().__len__()):
            self.cum_pixels.append(
                self.cum_pixels[-1] + DATASET_OUTPUT_HW_HIGH_RES * DATASET_OUTPUT_HW_HIGH_RES
            )

    def __len__(self):
        return self.cum_pixels[-1]

    def __getitem__(self, idx):
        # find which image the pixel belongs to
        img_idx = np.searchsorted(self.cum_pixels, idx, side="right") - 1

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
        ) = super().__getitem__(img_idx)

        HR_H, HR_W = DATASET_OUTPUT_HW_HIGH_RES, DATASET_OUTPUT_HW_HIGH_RES
        MR_H, MR_W = NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM
        LR_H, LR_W = NUM_LOW_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM
        label_H, label_W = 10, 10

        pixel_idx = idx - self.cum_pixels[img_idx]
        hr_row = pixel_idx // HR_W
        hr_col = pixel_idx % HR_W

        mr_row = hr_row * MR_H // HR_H
        mr_col = hr_col * MR_W // HR_W

        lr_row = hr_row * LR_H // HR_H
        lr_col = hr_col * LR_W // HR_W

        label_row = hr_row * label_H // HR_H
        label_col = hr_col * label_W // HR_W

        # pixel ablation: mask everything except from the selected positions
        # time-only and static data, and month stay the same
        if self.eval_config["pixel_ablations"]:
            default_s_t_h_m = s_t_h_m.clone()
            default_s_t_m_m = s_t_m_m.clone()
            default_s_t_l_m = s_t_l_m.clone()
            default_sp_m = sp_m.clone()

            s_t_h_m.fill_(1)
            s_t_m_m.fill_(1)
            s_t_l_m.fill_(1)
            sp_m.fill_(1)

            # restore the selected cells
            s_t_h_m[hr_row, hr_col, :, :] = default_s_t_h_m[hr_row, hr_col, :, :]
            s_t_m_m[mr_row, mr_col, :, :] = default_s_t_m_m[mr_row, mr_col, :, :]
            s_t_l_m[lr_row, lr_col, :, :] = default_s_t_l_m[lr_row, lr_col, :, :]
            sp_m[hr_row, hr_col, :] = default_sp_m[hr_row, hr_col, :]

            # becomes shape (1, 1)
            label = label[label_row, label_col]

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


class PixelAblationsEval(LandsatEval):
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
    ) -> PixelAblationsMetaDataset:
        ds = PixelAblationsMetaDataset(
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
