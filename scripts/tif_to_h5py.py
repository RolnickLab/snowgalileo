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

    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=4)


    for i, batch in enumerate(dataloader):
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
        ) = batch