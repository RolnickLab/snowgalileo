import argparse
from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from torch.utils.data import DataLoader

from src.data.config import (
    DATASET_OUTPUT_HW_HIGH_RES,
    DATASET_OUTPUT_HW_LOW_RES,
    DATASET_OUTPUT_HW_MED_RES,
    NORMALIZATION_DICT_FILENAME,
    NUM_TIMESTEPS,
)
from src.fsc.landsat_eval import LandsatEvalDataset as BaseDataset
from src.fsc.downstream_augmentation import DownstreamAugmentation
from src.data.dataset import Normalizer
from src.utils import config_dir


def plot_distribution(data, channel_idx, channel_name, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data.numpy().flatten(), bins=100, kde=True)
    plt.title(f"Distribution of {channel_name} (Channel {channel_idx})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(Path("assets_eval") / filename)
    plt.close()


class PlottingDataset(BaseDataset):
    def __init__(
        self,
        split: str = "train",
        h5pys_only: bool = False,
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        normalizer: Optional[Normalizer] = None,
        augmentation=DownstreamAugmentation(False),
        data_config: Dict = {},
    ):
        super().__init__(
            split,
            h5pys_only,
            exclude_prediction_date,
            exclude_prediction_high_res,
            exclude_prediction_sensors,
            normalizer,
            augmentation,
            data_config
        )

argparser = argparse.ArgumentParser()
argparser.add_argument("--eval_config", type=str, default="fsc_train_balanced_tiny.json")
argparser.add_argument("--normalize", action="store_true", help="Whether to normalize the data")

args = argparser.parse_args().__dict__

if __name__ == "__main__":

    with (Path("configs/finetune") / Path(args["eval_config"])).open("r") as f:
        eval_config = json.load(f)

    dataset = PlottingDataset(
        data_config = eval_config["data"]
    )

    if args["normalize"]:
        normalizing_dict = dataset.load_normalization_values(
            path=config_dir / NORMALIZATION_DICT_FILENAME
        )
        print(normalizing_dict, flush=True)
        normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        dataset.normalizer = normalizer

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)

    for i, (masked_output, labels, _) in enumerate(dataloader):
        if i == 2:
            break
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
        ) = masked_output

        valid_data_mask_s_t_h = torch.logical_not(s_t_h_m)
        valid_data_mask_s_t_m = torch.logical_not(s_t_m_m)
        valid_data_mask_s_t_l = torch.logical_not(s_t_l_m)
        valid_data_mask_sp = torch.logical_not(sp_m)
        valid_data_mask_t = torch.logical_not(t_m)
        valid_data_mask_st = torch.logical_not(st_m)

        s_t_h_x_c0_valid = s_t_h_x[..., 0][valid_data_mask_s_t_h[..., 0].bool()]
        s_t_h_x_c1_valid = s_t_h_x[..., 1][valid_data_mask_s_t_h[..., 0].bool()]
        s_t_h_x_c2_valid = s_t_h_x[..., 2][valid_data_mask_s_t_h[..., 0].bool()]
        s_t_h_x_c3_valid = s_t_h_x[..., 3][valid_data_mask_s_t_h[..., 1].bool()]
        s_t_h_x_c4_valid = s_t_h_x[..., 4][valid_data_mask_s_t_h[..., 1].bool()]
        s_t_h_x_c5_valid = s_t_h_x[..., 5][valid_data_mask_s_t_h[..., 1].bool()]
        s_t_h_x_c6_valid = s_t_h_x[..., 6][valid_data_mask_s_t_h[..., 2].bool()]
        s_t_h_x_c7_valid = s_t_h_x[..., 7][valid_data_mask_s_t_h[..., 3].bool()]
        s_t_h_x_c8_valid = s_t_h_x[..., 8][valid_data_mask_s_t_h[..., 3].bool()]
        s_t_h_x_c9_valid = s_t_h_x[..., 9][valid_data_mask_s_t_h[..., 4].bool()]
        s_t_h_x_c10_valid = s_t_h_x[..., 10][valid_data_mask_s_t_h[..., 4].bool()]
        s_t_h_x_c11_valid = s_t_h_x[..., 11][valid_data_mask_s_t_h[..., 4].bool()]
        s_t_h_x_c12_valid = s_t_h_x[..., 12][valid_data_mask_s_t_h[..., 5].bool()]
        s_t_h_x_c13_valid = s_t_h_x[..., 13][valid_data_mask_s_t_h[..., 6].bool()]
        s_t_h_x_c14_valid = s_t_h_x[..., 14][valid_data_mask_s_t_h[..., 6].bool()]

        s_t_m_x_c0_valid = s_t_m_x[..., 0][valid_data_mask_s_t_m[..., 0].bool()]
        s_t_m_x_c1_valid = s_t_m_x[..., 1][valid_data_mask_s_t_m[..., 0].bool()]

        s_t_l_x_c0_valid = s_t_l_x[..., 0][valid_data_mask_s_t_l[..., 0].bool()]
        s_t_l_x_c1_valid = s_t_l_x[..., 1][valid_data_mask_s_t_l[..., 0].bool()]
        s_t_l_x_c2_valid = s_t_l_x[..., 2][valid_data_mask_s_t_l[..., 0].bool()]
        s_t_l_x_c3_valid = s_t_l_x[..., 3][valid_data_mask_s_t_l[..., 1].bool()]
        s_t_l_x_c4_valid = s_t_l_x[..., 4][valid_data_mask_s_t_l[..., 2].bool()]
        s_t_l_x_c5_valid = s_t_l_x[..., 5][valid_data_mask_s_t_l[..., 2].bool()]
        s_t_l_x_c6_valid = s_t_l_x[..., 6][valid_data_mask_s_t_l[..., 2].bool()]
        s_t_l_x_c7_valid = s_t_l_x[..., 7][valid_data_mask_s_t_l[..., 3].bool()]
        s_t_l_x_c8_valid = s_t_l_x[..., 8][valid_data_mask_s_t_l[..., 4].bool()]
        s_t_l_x_c9_valid = s_t_l_x[..., 9][valid_data_mask_s_t_l[..., 5].bool()]
        s_t_l_x_c10_valid = s_t_l_x[..., 10][valid_data_mask_s_t_l[..., 6].bool()]

        sp_x_c0_valid = sp_x[..., 0][valid_data_mask_sp[..., 0].bool()]
        sp_x_c1_valid = sp_x[..., 1][valid_data_mask_sp[..., 0].bool()]
        sp_x_c2_valid = sp_x[..., 2][valid_data_mask_sp[..., 0].bool()]
        sp_x_c3_valid = sp_x[..., 3][valid_data_mask_sp[..., 1].bool()]
        sp_x_c4_valid = sp_x[..., 4][valid_data_mask_sp[..., 1].bool()]
        sp_x_c5_valid = sp_x[..., 5][valid_data_mask_sp[..., 1].bool()]
        sp_x_c6_valid = sp_x[..., 6][valid_data_mask_sp[..., 1].bool()]
        sp_x_c7_valid = sp_x[..., 7][valid_data_mask_sp[..., 1].bool()]
        sp_x_c8_valid = sp_x[..., 8][valid_data_mask_sp[..., 1].bool()]
        sp_x_c9_valid = sp_x[..., 9][valid_data_mask_sp[..., 1].bool()]
        sp_x_c10_valid = sp_x[..., 10][valid_data_mask_sp[..., 1].bool()]
        sp_x_c11_valid = sp_x[..., 11][valid_data_mask_sp[..., 1].bool()]
        sp_x_c12_valid = sp_x[..., 12][valid_data_mask_sp[..., 1].bool()]
        sp_x_c13_valid = sp_x[..., 13][valid_data_mask_sp[..., 1].bool()]

        t_x_c0_valid = t_x[..., 0][valid_data_mask_t[..., 0].bool()]
        t_x_c1_valid = t_x[..., 1][valid_data_mask_t[..., 0].bool()]
        t_x_c2_valid = t_x[..., 2][valid_data_mask_t[..., 2].bool()]
        t_x_c3_valid = t_x[..., 3][valid_data_mask_t[..., 3].bool()]
        t_x_c4_valid = t_x[..., 4][valid_data_mask_t[..., 4].bool()]
        t_x_c5_valid = t_x[..., 5][valid_data_mask_t[..., 4].bool()]
        t_x_c6_valid = t_x[..., 6][valid_data_mask_t[..., 4].bool()]
        t_x_c7_valid = t_x[..., 7][valid_data_mask_t[..., 4].bool()]
        t_x_c8_valid = t_x[..., 8][valid_data_mask_t[..., 4].bool()]

        st_x_c0_valid = st_x[..., 0][valid_data_mask_st[..., 0].bool()]
        st_x_c1_valid = st_x[..., 1][valid_data_mask_st[..., 0].bool()]
        st_x_c2_valid = st_x[..., 2][valid_data_mask_st[..., 0].bool()]

        # plot per channel distribution and save to file
        channel_names = {
            "s_t_h_x": [
                "S1 VV",
                "S1 VH",
                "S1 angle",
                "S2 B2",
                "S2 B3",
                "S2 B4",
                "S2 B8",
                "S2 B11",
                "S2 B12",
                "Landsat B2",
                "Landsat B3",
                "Landsat B4",
                "Landsat B5",
                "Landsat B6",
                "Landsat B7",
            ],
            "s_t_m_x": ["S3 Band 1", "S3 Band 2"],
            "s_t_l_x": [
                "MODIS Band 1",
                "MODIS Band 2",
                "MODIS Band 3",
                "MODIS Band 4",
                "MODIS Band 5",
                "MODIS Band 6",
                "MODIS Band 7",
                "VIIRS Band 1",
                "VIIRS Band 2",
                "NDSI",
                "NDVI",
            ],
        }
        for idx, (data, channel_name) in enumerate(
            zip(
                [
                    s_t_h_x_c0_valid,
                    s_t_h_x_c1_valid,
                    s_t_h_x_c2_valid,
                    s_t_h_x_c3_valid,
                    s_t_h_x_c4_valid,
                    s_t_h_x_c5_valid,
                    s_t_h_x_c6_valid,
                    s_t_h_x_c7_valid,
                    s_t_h_x_c8_valid,
                    s_t_h_x_c9_valid,
                    s_t_h_x_c10_valid,
                    s_t_h_x_c11_valid,
                    s_t_h_x_c12_valid,
                    s_t_h_x_c13_valid,
                    s_t_h_x_c14_valid,
                    s_t_m_x_c0_valid,
                    s_t_m_x_c1_valid,
                    s_t_l_x_c0_valid,
                    s_t_l_x_c1_valid,
                    s_t_l_x_c2_valid,
                    s_t_l_x_c3_valid,
                    s_t_l_x_c4_valid,
                    s_t_l_x_c5_valid,
                    s_t_l_x_c6_valid,
                    s_t_l_x_c7_valid,
                    s_t_l_x_c8_valid,
                    s_t_l_x_c9_valid,
                    s_t_l_x_c10_valid,
                ],
                channel_names["s_t_h_x"] + channel_names["s_t_m_x"] + channel_names["s_t_l_x"],
            )
        ):
            plot_distribution(
                data, idx, channel_name, f"{channel_name.replace(' ', '_')}_distribution.png"
            )

        # print the number of values that are below -1 or above 1 for NDSI and NDVI
        ndsi_out_of_bounds = torch.sum((s_t_l_x_c9_valid < -1) | (s_t_l_x_c9_valid > 1)).item()
        ndvi_out_of_bounds = torch.sum((s_t_l_x_c10_valid < -1) | (s_t_l_x_c10_valid > 1)).item()
        print(f"NDSI out of bounds count: {ndsi_out_of_bounds}")
        print(f"NDVI out of bounds count: {ndvi_out_of_bounds}")
        print(f"Total possible NDSI / NDVI values in this batch: {s_t_l_x_c9_valid.numel()}")

        for idx, (data, channel_name) in enumerate(
            zip(
                [
                    sp_x_c0_valid,
                    sp_x_c1_valid,
                    sp_x_c2_valid,
                    sp_x_c3_valid,
                    sp_x_c4_valid,
                    sp_x_c5_valid,
                    sp_x_c6_valid,
                    sp_x_c7_valid,
                    sp_x_c8_valid,
                    sp_x_c9_valid,
                    sp_x_c10_valid,
                    sp_x_c11_valid,
                    sp_x_c12_valid,
                    sp_x_c13_valid,
                ],
                [
                    "elevation",
                    "slope",
                    "aspect",
                    "WC Var 1",
                    "WC Var 2",
                    "WC Var 3",
                    "WC Var 4",
                    "WC Var 5",
                    "WC Var 6",
                    "WC Var 7",
                    "WC Var 8",
                    "WC Var 9",
                    "WC Var 10",
                    "WC Var 11",
                ],
            )
        ):
            plot_distribution(
                data, idx, channel_name, f"{channel_name.replace(' ', '_')}_distribution.png"
            )

        for idx, (data, channel_name) in enumerate(
            zip(
                [
                    t_x_c0_valid,
                    t_x_c1_valid,
                    t_x_c2_valid,
                    t_x_c3_valid,
                    t_x_c4_valid,
                    t_x_c5_valid,
                    t_x_c6_valid,
                    t_x_c7_valid,
                    t_x_c8_valid,
                ],
                [
                    "VIIRS Band 3",
                    "VIIRS Band 4",
                    "VIIRS Band 5",
                    "VIIRS Band 6",
                    "ERA5 Var 1",
                    "ERA5 Var 2",
                    "ERA5 Var 3",
                    "ERA5 Var 4",
                    "ERA5 Var 5",
                ],
            )
        ):
            plot_distribution(
                data, idx, channel_name, f"{channel_name.replace(' ', '_')}_distribution.png"
            )

        for idx, (data, channel_name) in enumerate(
            zip([st_x_c0_valid, st_x_c1_valid, st_x_c2_valid], ["x", "y", "z"])
        ):
            plot_distribution(
                data, idx, channel_name, f"{channel_name.replace(' ', '_')}_distribution.png"
            )
