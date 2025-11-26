from src.data.config import DATA_FOLDER, NORMALIZATION_DICT_FILENAME
from src.utils import config_dir
from src.data.dataset import Dataset, Normalizer
import numpy as np
from pathlib import Path
import argparse
from torch.utils.data import DataLoader

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--h5py_folder", type=str, default="data/h5pys_pretrain"
)
argparser.add_argument(
    "--tif_folder", type=str, default="data/tifs_all_bands"
)

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

    for i in range(len(dataset)):
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            months,
            s_t_h_m,
            s_t_m_m,
            s_t_l_m,
            sp_m,
            t_m,
            st_m
        ) = dataset[i]

        ndvi = s_t_l_x[..., -1][s_t_l_m[..., -1].astype(bool)]
        ndsi = s_t_l_x[..., -2][s_t_l_m[..., -2].astype(bool)]

        ndvi_out_of_bounds_count += np.sum((ndvi < -1) | (ndvi > 1))
        ndsi_out_of_bounds_count += np.sum((ndsi < -1) | (ndsi > 1))

    print(f"Total NDVI out of bounds count: {ndvi_out_of_bounds_count}")
    print(f"Total NDSI out of bounds count: {ndsi_out_of_bounds_count}")
    print(f"Total possible NDSI / NDVI values: {len(dataset)*4*8}")
