from src.data.config import DATA_FOLDER, NORMALIZATION_DICT_FILENAME
from src.utils import config_dir
from src.data.dataset import Dataset, Normalizer
import numpy as np
from pathlib import Path
import argparse

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

    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(normalizing_dict, flush=True)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer

    stats = []

    # create a csv that stores the min and max values for each channel
    for i in range(len(dataset)):
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
        ) = dataset[i]