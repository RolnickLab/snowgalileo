import json
from math import sqrt
from pathlib import Path
from typing import Optional, Tuple, cast

import geobench
import numpy as np
import torch.multiprocessing
from einops import repeat
from torch.utils.data import Dataset as PyTorchDataset

from ..data.dataset import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    normalize_space_time,
)
from ..data.earthengine.s1 import S1_BANDS
from ..data.earthengine.s2 import S2_BANDS
from ..masking import MaskedOutput
from ..utils import masked_output_np_to_tensor

torch.multiprocessing.set_sharing_strategy("file_system")


class GeobenchBaseDataset(PyTorchDataset):
    """
    Class implementation inspired by: https://github.com/vishalned/MMEarth-train/tree/main
    """

    def __init__(
        self,
        dataset_config_file: str,
        split: str = "train",
        num_subtiles_per_image: Optional[int] = 1,
    ):
        with (
            Path(__file__).parents[0] / Path("geobench_configs") / Path(dataset_config_file)
        ).open("r") as f:
            config = json.load(f)

        assert split in ["train", "test"]
        assert config["benchmark_name"] in ["classification_v1.0", "segmentation_v1.0"]

        self.split = split
        self.config = config

        for task in geobench.task_iterator(benchmark_name=self.config["benchmark_name"]):
            if task.dataset_name == self.config["dataset_name"]:
                break

        self.dataset = task.get_dataset(split=self.split, band_names=self.config["band_names"])
        self.label_map = task.get_label_map()
        self.label_stats = task.label_stats()
        self.dataset_dir = task.get_dataset_dir()
        self.tmp_band_names = [
            self.dataset[0].bands[i].band_info.name for i in range(len(self.dataset[0].bands))
        ]
        # get the tmp bands in the same order as the ones present in the band names list
        self.tmp_band_indices = [
            self.tmp_band_names.index(band_name) for band_name in self.config["band_names"]
        ]
        self.in_channels = len(self.tmp_band_indices)

        self.masks = self.create_masks()

        self.num_subtiles_per_image = num_subtiles_per_image
        assert sqrt(cast(float, self.num_subtiles_per_image)).is_integer()

    def create_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config["include_s1"]:
            s_t_channels = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S" in key
            ]
        else:
            s_t_channels = [
                idx for idx, key in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if "S2" in key
            ]

        # everything is masked by default
        s_t_m = np.ones([len(SPACE_TIME_BANDS_GROUPS_IDX)])
        # unmask available s1 and s2 bands
        s_t_m[s_t_channels] = 0
        s_t_m = repeat(
            s_t_m,
            "d -> h w t d",
            h=self.config["input_height_width"],
            w=self.config["input_height_width"],
            t=self.config["num_timesteps"],
        )

        # no static channels are available
        sp_m = np.ones(
            [
                self.config["input_height_width"],
                self.config["input_height_width"],
                len(SPACE_BAND_GROUPS_IDX),
            ]
        )
        t_m = np.ones([self.config["num_timesteps"], len(TIME_BAND_GROUPS_IDX)])
        st_m = np.ones([len(STATIC_BAND_GROUPS_IDX)])

        assert ((s_t_m == 0) | (s_t_m == 1)).all()
        assert (sp_m == 1).all()
        assert (t_m == 1).all()
        assert (st_m == 1).all()

        return (s_t_m, sp_m, t_m, st_m)

    def image_to_space_time_array(self, image: np.ndarray) -> np.ndarray:
        if self.config["include_s1"]:
            kept_dynamic_bands = [
                idx for idx, x in enumerate(SPACE_TIME_BANDS) if (x in S2_BANDS or x in S1_BANDS)
            ]
        else:
            kept_dynamic_bands = [idx for idx, x in enumerate(SPACE_TIME_BANDS) if x in S2_BANDS]

        eo_style_array = np.zeros(
            [
                self.config["input_height_width"],
                self.config["input_height_width"],
                self.config["num_timesteps"],
                len(SPACE_TIME_BANDS),
            ]
        )
        eo_style_array[:, :, :, kept_dynamic_bands] = repeat(
            image, "c h w -> h w t c", t=self.config["num_timesteps"]
        )

        return normalize_space_time(eo_style_array)

    def __getitem__(self, idx) -> Tuple[MaskedOutput, torch.Tensor]:
        img_idx = idx // self.num_subtiles_per_image

        label = self.dataset[img_idx].label

        x = []

        for band_idx in self.tmp_band_indices:
            x.append(self.dataset[img_idx].bands[band_idx].data)

        x = np.stack(x, axis=0)

        s_t_x = self.image_to_space_time_array(x)

        # space only / time only / static bands are not provided by so2sat
        sp_x = np.zeros((s_t_x.shape[0], s_t_x.shape[1], len(SPACE_BANDS)))
        t_x = np.zeros((s_t_x.shape[2], len(TIME_BANDS)))
        st_x = np.zeros((len(STATIC_BANDS)))

        s_t_m, sp_m, t_m, st_m = self.masks
        month = np.zeros((self.config["num_timesteps"],))

        # check if label is an object or a number
        if not (isinstance(label, int) or isinstance(label, list)):
            label = label.data
            # label is a memoryview object, convert it to a list, and then to a numpy array
            label = np.array(list(label))

        targets = torch.tensor(label, dtype=torch.long)

        subtiles_per_dim = int(sqrt(cast(float, self.num_subtiles_per_image)))
        h, w = s_t_x.shape[:2]
        assert h == w  # this is the case for Geobench datasets
        assert h % subtiles_per_dim == 0
        pixels_per_dim = h // subtiles_per_dim
        subtile_idx = idx % self.num_subtiles_per_image

        row_idx = subtile_idx // subtiles_per_dim
        col_idx = subtile_idx % subtiles_per_dim

        if self.config["benchmark_name"] == "segmentation_v1.0":
            targets = (
                targets[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                ],
            )

        return (
            masked_output_np_to_tensor(
                s_t_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                sp_x[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_x,
                st_x,
                s_t_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                    :,
                ],
                sp_m[
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                    :,
                ],
                t_m,
                st_m,
                month,
            ),
            targets,
        )

    def __len__(self) -> int:
        return len(self.dataset) * self.num_subtiles_per_image
