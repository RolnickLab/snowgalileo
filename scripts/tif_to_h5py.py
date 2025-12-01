import argparse
from pathlib import Path

import numpy as np

from src.data.dataset import Dataset
from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument("--h5py_folder", type=str, default="data/h5pys_pretrain")
argparser.add_argument("--tif_folder", type=str, default="data/tifs_all_bands")

args = argparser.parse_args().__dict__

if __name__ == "__main__":
    dataset = Dataset(
        data_folder=Path(args["tif_folder"]),
        download=False,
        h5py_folder=Path(args["h5py_folder"]),
        h5pys_only=False,
    )

    ndvi_out_of_bounds_count = 0
    ndsi_out_of_bounds_count = 0

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
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
        ) = batch

        ndvi = s_t_l_x[..., -1][valid_data_mask_s_t_l[..., -1].astype(bool)]
        ndsi = s_t_l_x[..., -2][valid_data_mask_s_t_l[..., -2].astype(bool)]

        ndvi_out_of_bounds_count += np.sum((ndvi < -1) | (ndvi > 1))
        ndsi_out_of_bounds_count += np.sum((ndsi < -1) | (ndsi > 1))

    print(f"Total NDVI out of bounds count: {ndvi_out_of_bounds_count}")
    print(f"Total NDSI out of bounds count: {ndsi_out_of_bounds_count}")
    print(f"Total possible NDSI / NDVI values: {len(dataset) * 4 * 8}")
