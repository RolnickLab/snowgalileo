from functools import partial

import torch
from torch.utils.data import DataLoader

from src.collate_fns import mae_collate_fn
from src.data.config import DATA_FOLDER, NO_DATA_VALUE, NORMALIZATION_DICT_FILENAME, TIFS_FOLDER
from src.data.dataset import Dataset, Normalizer
from src.utils import config_dir, load_check_config

config = load_check_config("ai4snow.json")
training_config = config["training"]

dataset = Dataset(
    TIFS_FOLDER,
    download=False,
    h5py_folder=DATA_FOLDER / "h5pys",
    h5pys_only=False,
)

if training_config["normalization"] == "std":
    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(NORMALIZATION_DICT_FILENAME)
    print(normalizing_dict, flush=True)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer
else:
    normalizer = Normalizer(std=False)
    dataset.normalizer = normalizer

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

            valid_mask_s_t = s_t_x != NO_DATA_VALUE
            valid_mask_sp = sp_x != NO_DATA_VALUE
            valid_mask_t = t_x != NO_DATA_VALUE
            valid_mask_st = st_x != NO_DATA_VALUE

            assert s_t_x.shape == valid_mask_s_t.shape
            assert sp_x.shape == valid_mask_sp.shape
            assert t_x.shape == valid_mask_t.shape
            assert st_x.shape == valid_mask_st.shape

            # import pdb; pdb.set_trace()

            # Compute mean and std per channel, excluding NO_DATA_VALUE
            s_t_x_mean = torch.tensor(
                [
                    torch.mean(s_t_x[..., i][valid_mask_s_t[..., i]])
                    if valid_mask_s_t[..., i].any()
                    else float("nan")
                    for i in range(s_t_x.shape[-1])
                ]
            )
            s_t_x_std = torch.tensor(
                [
                    torch.std(s_t_x[..., i][valid_mask_s_t[..., i]])
                    if valid_mask_s_t[..., i].any()
                    else float("nan")
                    for i in range(s_t_x.shape[-1])
                ]
            )

            sp_x_mean = torch.tensor(
                [
                    torch.mean(sp_x[..., i][valid_mask_sp[..., i]])
                    if valid_mask_sp[..., i].any()
                    else float("nan")
                    for i in range(sp_x.shape[-1])
                ]
            )
            sp_x_std = torch.tensor(
                [
                    torch.std(sp_x[..., i][valid_mask_sp[..., i]])
                    if valid_mask_sp[..., i].any()
                    else float("nan")
                    for i in range(sp_x.shape[-1])
                ]
            )

            t_x_mean = torch.tensor(
                [
                    torch.mean(t_x[..., i][valid_mask_t[..., i]])
                    if valid_mask_t[..., i].any()
                    else float("nan")
                    for i in range(t_x.shape[-1])
                ]
            )
            t_x_std = torch.tensor(
                [
                    torch.std(t_x[..., i][valid_mask_t[..., i]])
                    if valid_mask_t[..., i].any()
                    else float("nan")
                    for i in range(t_x.shape[-1])
                ]
            )

            st_x_mean = torch.tensor(
                [
                    torch.mean(st_x[i][valid_mask_st[i]])
                    if valid_mask_st[i].any()
                    else float("nan")
                    for i in range(st_x.shape[-1])
                ]
            )
            st_x_std = torch.tensor(
                [
                    torch.std(st_x[i][valid_mask_st[i]])
                    if valid_mask_st[i].any()
                    else float("nan")
                    for i in range(st_x.shape[-1])
                ]
            )

            for i, (mean, std) in enumerate(zip(s_t_x_mean, s_t_x_std)):
                print(f"s_t_x channel {i}: Mean = {mean.item():.4f}, Std = {std.item():.4f}")

            for i, (mean, std) in enumerate(zip(sp_x_mean, sp_x_std)):
                print(f"sp_x channel {i}: Mean = {mean.item():.4f}, Std = {std.item():.4f}")

            for i, (mean, std) in enumerate(zip(t_x_mean, t_x_std)):
                print(f"t_x channel {i}: Mean = {mean.item():.4f}, Std = {std.item():.4f}")

            for i, (mean, std) in enumerate(zip(st_x_mean, st_x_std)):
                print(f"st_x channel {i}: Mean = {mean.item():.4f}, Std = {std.item():.4f}")
