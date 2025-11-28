import argparse
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset as BaseDataset
from src.data.dataset import Normalizer
from src.utils import config_dir
from typing import Optional
from src.data.config import (
    DATASET_OUTPUT_HW_HIGH_RES,
    DATASET_OUTPUT_HW_MED_RES,
    DATASET_OUTPUT_HW_LOW_RES,
    NUM_TIMESTEPS,
)

def plot_distribution(data, channel_idx, channel_name, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data.numpy().flatten(), bins=100, kde=True)
    plt.title(f"Distribution of {channel_name} (Channel {channel_idx})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(Path("assets") / filename)
    plt.close()


class PlottingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: Path,
        download: bool = True,
        h5py_folder: Optional[Path] = None,
        h5pys_only: bool = False,
        output_hw_high_res: int = DATASET_OUTPUT_HW_HIGH_RES,
        output_hw_med_res: int = DATASET_OUTPUT_HW_MED_RES,
        output_hw_low_res: int = DATASET_OUTPUT_HW_LOW_RES,
        output_timesteps: int = NUM_TIMESTEPS,
        normalizer: Optional[Normalizer] = None,
    ):
        super().__init__(
            data_folder,
            download,
            h5py_folder,
            h5pys_only,
            output_hw_high_res,
            output_hw_med_res,
            output_hw_low_res,
            output_timesteps,
            normalizer,
        )

    def __getitem__(self, idx):
        if self.h5pys_only:
            h5pys = self.read_and_slice_h5py_file(self.h5pys[idx])
            if self.normalizer is None:
                return h5pys, self.input_tifs[idx].name
            return h5pys.normalize(self.normalizer), self.input_tifs[idx].name
        else:
            h5py = self.load_tif(idx)
            if self.normalizer is None:
                return h5py, self.input_tifs[idx].name
            return h5py.normalize(self.normalizer), self.input_tifs[idx].name


argparser = argparse.ArgumentParser()
argparser.add_argument("--h5py_folder", type=str, default="data/h5pys_pretrain_new_new")
argparser.add_argument("--tif_folder", type=str, default="data/tifs_all_bands")
argparser.add_argument("--normalize", action="store_true", help="Whether to normalize the data")

args = argparser.parse_args().__dict__

if __name__ == "__main__":
    dataset = PlottingDataset(
        data_folder=Path(args["tif_folder"]),
        download=False,
        h5py_folder=Path(args["h5py_folder"]),
        h5pys_only=False,
    )

    if args["normalize"]:
        normalizing_dict = dataset.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
        print(normalizing_dict, flush=True)
        normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        dataset.normalizer = normalizer

    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        if i == 2:
            break
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            months,
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ), name = batch

        s_t_h_x_c0_valid = s_t_h_x[..., 0][valid_data_mask_s_t_h[..., 0].bool()]
        s_t_h_x_c1_valid = s_t_h_x[..., 1][valid_data_mask_s_t_h[..., 1].bool()]
        s_t_h_x_c2_valid = s_t_h_x[..., 2][valid_data_mask_s_t_h[..., 2].bool()]
        s_t_h_x_c3_valid = s_t_h_x[..., 3][valid_data_mask_s_t_h[..., 3].bool()]
        s_t_h_x_c4_valid = s_t_h_x[..., 4][valid_data_mask_s_t_h[..., 4].bool()]
        s_t_h_x_c5_valid = s_t_h_x[..., 5][valid_data_mask_s_t_h[..., 5].bool()]
        s_t_h_x_c6_valid = s_t_h_x[..., 6][valid_data_mask_s_t_h[..., 6].bool()]
        s_t_h_x_c7_valid = s_t_h_x[..., 7][valid_data_mask_s_t_h[..., 7].bool()]
        s_t_h_x_c8_valid = s_t_h_x[..., 8][valid_data_mask_s_t_h[..., 8].bool()]
        s_t_h_x_c9_valid = s_t_h_x[..., 9][valid_data_mask_s_t_h[..., 9].bool()]
        s_t_h_x_c10_valid = s_t_h_x[..., 10][valid_data_mask_s_t_h[..., 10].bool()]
        s_t_h_x_c11_valid = s_t_h_x[..., 11][valid_data_mask_s_t_h[..., 11].bool()]
        s_t_h_x_c12_valid = s_t_h_x[..., 12][valid_data_mask_s_t_h[..., 12].bool()]
        s_t_h_x_c13_valid = s_t_h_x[..., 13][valid_data_mask_s_t_h[..., 13].bool()]
        s_t_h_x_c14_valid = s_t_h_x[..., 14][valid_data_mask_s_t_h[..., 14].bool()]

        s_t_m_x_c0_valid = s_t_m_x[..., 0][valid_data_mask_s_t_m[..., 0].bool()]
        s_t_m_x_c1_valid = s_t_m_x[..., 1][valid_data_mask_s_t_m[..., 1].bool()]

        s_t_l_x_c0_valid = s_t_l_x[..., 0][valid_data_mask_s_t_l[..., 0].bool()]
        s_t_l_x_c1_valid = s_t_l_x[..., 1][valid_data_mask_s_t_l[..., 1].bool()]
        s_t_l_x_c2_valid = s_t_l_x[..., 2][valid_data_mask_s_t_l[..., 2].bool()]
        s_t_l_x_c3_valid = s_t_l_x[..., 3][valid_data_mask_s_t_l[..., 3].bool()]
        s_t_l_x_c4_valid = s_t_l_x[..., 4][valid_data_mask_s_t_l[..., 4].bool()]
        s_t_l_x_c5_valid = s_t_l_x[..., 5][valid_data_mask_s_t_l[..., 5].bool()]
        s_t_l_x_c6_valid = s_t_l_x[..., 6][valid_data_mask_s_t_l[..., 6].bool()]
        s_t_l_x_c7_valid = s_t_l_x[..., 7][valid_data_mask_s_t_l[..., 7].bool()]
        s_t_l_x_c8_valid = s_t_l_x[..., 8][valid_data_mask_s_t_l[..., 8].bool()]
        s_t_l_x_c9_valid = s_t_l_x[..., 9][valid_data_mask_s_t_l[..., 9].bool()]
        s_t_l_x_c10_valid = s_t_l_x[..., 10][valid_data_mask_s_t_l[..., 10].bool()]

        sp_x_c0_valid = sp_x[..., 0][valid_data_mask_sp[..., 0].bool()]
        sp_x_c1_valid = sp_x[..., 1][valid_data_mask_sp[..., 1].bool()]
        sp_x_c2_valid = sp_x[..., 2][valid_data_mask_sp[..., 2].bool()]
        sp_x_c3_valid = sp_x[..., 3][valid_data_mask_sp[..., 3].bool()]
        sp_x_c4_valid = sp_x[..., 4][valid_data_mask_sp[..., 4].bool()]
        sp_x_c5_valid = sp_x[..., 5][valid_data_mask_sp[..., 5].bool()]
        sp_x_c6_valid = sp_x[..., 6][valid_data_mask_sp[..., 6].bool()]
        sp_x_c7_valid = sp_x[..., 7][valid_data_mask_sp[..., 7].bool()]
        sp_x_c8_valid = sp_x[..., 8][valid_data_mask_sp[..., 8].bool()]
        sp_x_c9_valid = sp_x[..., 9][valid_data_mask_sp[..., 9].bool()]
        sp_x_c10_valid = sp_x[..., 10][valid_data_mask_sp[..., 10].bool()]
        sp_x_c11_valid = sp_x[..., 11][valid_data_mask_sp[..., 11].bool()]
        sp_x_c12_valid = sp_x[..., 12][valid_data_mask_sp[..., 12].bool()]
        sp_x_c13_valid = sp_x[..., 13][valid_data_mask_sp[..., 13].bool()]

        t_x_c0_valid = t_x[..., 0][valid_data_mask_t[..., 0].bool()]
        t_x_c1_valid = t_x[..., 1][valid_data_mask_t[..., 1].bool()]
        t_x_c2_valid = t_x[..., 2][valid_data_mask_t[..., 2].bool()]
        t_x_c3_valid = t_x[..., 3][valid_data_mask_t[..., 3].bool()]
        t_x_c4_valid = t_x[..., 4][valid_data_mask_t[..., 4].bool()]
        t_x_c5_valid = t_x[..., 5][valid_data_mask_t[..., 5].bool()]
        t_x_c6_valid = t_x[..., 6][valid_data_mask_t[..., 6].bool()]
        t_x_c7_valid = t_x[..., 7][valid_data_mask_t[..., 7].bool()]
        t_x_c8_valid = t_x[..., 8][valid_data_mask_t[..., 8].bool()]

        st_x_c0_valid = st_x[..., 0][valid_data_mask_st[..., 0].bool()]
        st_x_c1_valid = st_x[..., 1][valid_data_mask_st[..., 1].bool()]
        st_x_c2_valid = st_x[..., 2][valid_data_mask_st[..., 2].bool()]

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

        # for debugging purposes, if S1 VH contains zero values, print how many
        s1_vh_zero_count = torch.sum(s_t_h_x_c1_valid == 0).item()
        print(f"S1 VH zero count: {s1_vh_zero_count} in tif {name} in batch {i}")

        # for debugging purposes, if S1 VH contains zero values, print how many
        s2_b11_zero_count = torch.sum(s_t_h_x_c7_valid == 0).item()
        print(f"S2 B11 zero count: {s2_b11_zero_count} in tif {name} in batch {i}")

        # for debugging purposes, if S1 VH contains zero values, print how many
        landsat_b6_zero_count = torch.sum(s_t_h_x_c13_valid == 0).item()
        print(f"Landsat B6 zero count: {landsat_b6_zero_count} in tif {name} in batch {i}")

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
        
        # print unique values count for aspect channel
        aspect_unique_values = torch.unique(sp_x_c2_valid, return_counts=True)
        print(f"Aspect unique values and counts: {aspect_unique_values}")

        # for debugging purposes, if S1 VH contains zero values, print how many
        aspect_zero_count = torch.sum(sp_x_c2_valid == 0).item()
        print(f"Aspect zero count: {aspect_zero_count} in tif {name} in batch {i}")

        # for debugging purposes, if S1 VH contains zero values, print how many
        slope_zero_count = torch.sum(s_t_h_x_c1_valid == 0).item()
        print(f"Slope zero count: {slope_zero_count} in tif {name} in batch {i}")

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
