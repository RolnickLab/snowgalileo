from src.data.dataset import Dataset
from src.data.config import DATA_FOLDER, TIFS_FOLDER
from torch.utils.data import DataLoader
from src.collate_fns import mae_collate_fn
import json
from functools import partial
from src.utils import load_check_config
import torch

config = load_check_config("ai4snow.json")
training_config = config["training"]

dataset = Dataset(
    TIFS_FOLDER,
    download=False,
    h5py_folder=DATA_FOLDER / "h5pys",
    h5pys_only=False,
)

dataloader = DataLoader(
    dataset,
    batch_size=1000,
    shuffle=True,
    num_workers=0,
    collate_fn=partial(
        mae_collate_fn,
        patch_sizes=training_config["patch_sizes"],
        shape_time_combinations=training_config["shape_time_combinations"],
        encode_ratio=training_config["encode_ratio"],
        decode_ratio=training_config["decode_ratio"],
        augmentation_strategies=training_config["augmentation"],
        masking_probabilities=training_config["masking_probabilities"],
        max_unmasking_channels=training_config["max_unmasking_channels"],
        random_masking=training_config["random_masking"],
        unmasking_channels_combo=training_config["unmasking_channels_combo"],
    ),
    pin_memory=True,
)

for i, batch in enumerate(dataloader):
    if i == 3:
        break
    else:
        for b in batch:
            (
                s_t_x,
                sp_x,
                t_x,
                st_x,
                s_t_m,
                sp_m,
                t_m,
                st_m,
                months,
                patch_size,
                c_i,
            ) = b
            print(torch.mean(s_t_x[s_t_x != -9999]))
            print(torch.std(s_t_x[s_t_x != -9999]))
            print(torch.mean(sp_x[sp_x != -9999]))
            print(torch.std(sp_x[sp_x != -9999]))
            print(torch.mean(t_x[t_x != -9999]))
            print(torch.std(t_x[t_x != -9999]))
            print(torch.mean(st_x[st_x != -9999]))
            print(torch.std(st_x[st_x != -9999]))