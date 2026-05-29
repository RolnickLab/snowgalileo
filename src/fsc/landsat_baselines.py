import json
import math
import random
from pathlib import Path
from time import time
from typing import Dict, Optional, Union, cast

import joblib
import numpy as np
import rioxarray
import torch
import wandb
import xarray as xr
from einops import rearrange, reduce, repeat
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Subset

from src.config import DEFAULT_SEED
from src.data.config import DATA_FOLDER, RESULTS_FOLDER
from src.data.dataset import Normalizer
from src.data.earthengine.eo_eval import SPACE_TIME_HIGH_RES_BANDS, TIME_BANDS
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset, masked_output_np_to_tensor
from src.fsc.metrics import compute_regression_metrics


class LandsatEvalDatasetSklearn(LandsatEvalDataset):
    """The Random Forest baseline uses the same dataset as the main
    LandsatEvalDataset, but doesn't group channel masks into channel group
    masks, so that we can directly remove masked channels for the Random Forest
    input.
    """

    def __init__(
        self,
        split: str = "train",
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        exclude_prediction_era5: bool = True,
        normalizer: Optional[Normalizer] = None,
        data_config: Dict = {},
        h5pys_only: bool = False,
    ):
        super().__init__(
            split=split,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            normalizer=normalizer,
            data_config=data_config,
            h5pys_only=h5pys_only,
        )

    # NOTE: overwritten because for baselines, we use ungrouped channels
    def mask_prediction_high_res(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # masks the high resolution, optical data in the prediction timestep
        # high resolution channels are: s1, s2, landsat, so we retain the first 3 channels
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_high_res
        assert s_t_h_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS)
        s_t_h_m[:, :, -1, 3:] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    # NOTE: overwritten because for baselines, we use ungrouped channels
    def mask_prediction_sensor_data(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # Masks all sensor channel groups in the prediction timestep
        # This includes all Sentinel-1, Sentinel-2, Landsat, Sentinel-3, MODIS, VIIRS data, as well as the MODIS-derived indeces
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_sensors
        assert t_m.shape[-1] == len(TIME_BANDS)
        s_t_h_m[:, :, -1, :] = 1
        s_t_m_m[:, :, -1, :] = 1
        s_t_l_m[:, :, -1, :] = 1
        t_m[-1, :-5] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    # NOTE: overwritten because for baselines, we use ungrouped channels
    def mask_prediction_era5(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # NOTE: 0 = valid, 1 = masked
        print("Masking ERA5 data in prediction timestep", flush=True)
        assert self.exclude_prediction_era5
        assert t_m.shape[-1] == len(TIME_BANDS)
        t_m[-1, 4:] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    def __getitem__(self, idx):
        if self.h5pys_only:
            h5py_path, _ = self.pairs[idx]
            print(f"Using {h5py_path}", flush=True)
            h5py = self.read_and_slice_h5py_file(h5py_path)
        else:
            h5py = self.load_tif(idx)

        if self.normalizer is None:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                month,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = h5py

        else:
            (
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                month,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            ) = h5py.normalize(self.normalizer)

        s_t_h_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_h))
        s_t_m_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_m))
        s_t_l_m = torch.as_tensor(np.logical_not(valid_data_mask_s_t_l))
        sp_m = torch.as_tensor(np.logical_not(valid_data_mask_sp))
        t_m = torch.as_tensor(np.logical_not(valid_data_mask_t))
        st_m = torch.as_tensor(np.logical_not(valid_data_mask_st))

        # since the prediction timestep function is channel-independent, we don't have to overwrite it
        if self.exclude_prediction_date:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_timestep(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        if self.exclude_prediction_high_res:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_high_res(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        if self.exclude_prediction_sensors:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_sensor_data(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        if self.exclude_prediction_era5:
            (
                s_t_h_m,
                s_t_m_m,
                s_t_l_m,
                sp_m,
                t_m,
                st_m,
            ) = self.mask_prediction_era5(s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m)

        image_path, label_path = self.pairs[idx]
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label_path)) as data:
            label = cast(np.ndarray, data.values)
            # remove first dimension (for shape consistency)
            label = np.squeeze(label, axis=0)

        assert image_path.name.split(".")[0] == label_path.name.split(".")[0], (
            f"Input path {image_path.name} and label path {label_path.name} do not match."
        )

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
                month,
            ),
            label,
            image_path.name,  # for logging purposes
        )


class LandsatEvalSklearn(LandsatEval):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        exclude_prediction_era5: bool = True,
        num_tokens_per_dim: int = 10,
        h5pys_only: bool = False,
        resample: bool = False,
        eval_config: Dict = {},
        model_type: str = "rf",
        normalizing_dict: Optional[Dict] = None,
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.exclude_prediction_sensors = exclude_prediction_sensors
        self.exclude_prediction_era5 = exclude_prediction_era5
        self.num_tokens_per_dim = num_tokens_per_dim
        self.resample = resample
        self.name = "ls_rf"
        self.model_type = model_type
        self.normalizing_dict = normalizing_dict
        self.h5pys_only = h5pys_only

        assert model_type in ["rf", "svr", "mlp"], f"Unknown model type {model_type}"

        super().__init__(
            normalization=normalization,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            resample=resample,
            eval_config=eval_config,
            h5pys_only=self.h5pys_only,
        )

    def forward_filling_masked_data_per_channel_else_aggregate(self, x, m, t, array_type: str):
        """Fills masked values in x by forward-filling along the time dimension
        per channel.

        Remaining NaNs will fall back to aggregation replacement.
        """
        assert x.dim() == 4, f"Expected 4D tensor, got shape {x.shape}"

        x = x.clone()
        m = m.bool()
        t = t.clone()

        x = torch.masked_fill(x, m, float("nan"))
        valid_mask = ~torch.isnan(x)

        # accumulate last valid value along time axis
        timestep_idx = torch.arange(x.size(-1), device=x.device)
        timestep_idx = timestep_idx.view(1, 1, 1, -1).expand_as(x)

        # invalid positions get -1 index, so they don't contribute to cummax
        idx_masked = torch.where(valid_mask, timestep_idx, torch.full_like(timestep_idx, -1))

        # get index of last valid timestep
        last_idx = torch.cummax(idx_masked, dim=-1).values

        # gather forward-filled values, replace with NaN where no valid value has occurred yet (index == -1, after clamp: 0)
        ff = torch.gather(x, dim=-1, index=last_idx.clamp(min=0))
        ff = torch.where(last_idx == -1, torch.tensor(float("nan"), device=x.device), ff)
        x = ff

        # update mask too
        m = torch.where(torch.isnan(x), 1, 0)

        # fill remaining NaNs with medians
        x = self.replace_masked_data_with_aggregate(x, m, array_type=array_type)

        # update time distance: last_idx already encodes last-valid timestep index
        timestep_grid = timestep_idx
        dist = timestep_grid - last_idx
        # TODO: change value if we decide for another placeholder
        dist[last_idx == -1] = -1
        t = dist

        return x, t

    @staticmethod
    def median_replace(x, m, dims):
        """Replaces masked values in x with the median over the specified dims.

        Breaks early if no NaNs are left after a replacement step.
        """
        x = torch.masked_fill(x, m.bool(), float("nan"))
        for d in dims:
            x = torch.where(torch.isnan(x), torch.nanmedian(x, dim=d, keepdim=True).values, x)
            if not torch.isnan(x).any():
                break

        return x

    def replace_masked_data_with_aggregate(self, x, m, array_type: str):
        """Replaces masked values in x with an aggregate.

        First, tries to replace over the time dimension for timeseries
        data, then over the space dimension. For space-only and static
        data, replaces over space dimension. If no replacement is
        possible, fill with the per-channel mean of the pre-training
        data.
        """
        # timeseries data with shape (B, S, C, T)
        if x.dim() == 4:
            dims = [-1, -3]
        # space-only or static data with shape (B, S, C)
        elif x.dim() == 3:
            dims = [-2]
        else:
            raise ValueError(f"Unexpected shape {x.shape}")
        x = LandsatEvalSklearn.median_replace(x, m, dims)

        if torch.isnan(x).any():
            assert self.normalizing_dict is not None, (
                "normalizing_dict must be provided for final fill value replacement."
            )
            # fill remaining NaNs with per-channel mean from normalizing_dict
            if array_type not in self.normalizing_dict:
                raise ValueError(f"Unknown array type: {array_type}")
            channel_means = torch.tensor(
                self.normalizing_dict[array_type]["mean"], device=x.device
            )
            assert channel_means.size(0) == x.size(2)
            for c in range(x.size(2)):
                x[:, :, c] = torch.where(
                    torch.isnan(x[:, :, c]),
                    channel_means[c],
                    x[:, :, c],
                )
        return x

    def aggregate_data_per_output_pixel(
        self,
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
        month,
    ):
        """Aggregates input data per output pixel by bringing all data to the
        output resolution.

        This means upsampling medium and low resolution data, and
        aggregating high resolution data to output pixel resolution.
        """
        # determine upsampling factors for medium and low resolution data
        p_m = self.num_tokens_per_dim // s_t_m_x.shape[1]
        p_l = self.num_tokens_per_dim // s_t_l_x.shape[1]

        # first, bring all data into token resolution (the output resolution)
        # t_h = token height, t_w = token width
        s_t_h_x = rearrange(
            reduce(
                s_t_h_x,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                t_h=self.num_tokens_per_dim,
                t_w=self.num_tokens_per_dim,
                reduction="mean",  # average pooling for input data
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_h_m = rearrange(
            reduce(
                s_t_h_m,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                t_h=self.num_tokens_per_dim,
                t_w=self.num_tokens_per_dim,
                reduction="max",  # if one value is masked, the entire patch should be masked
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        # repeat medium and low resolution tokens over high resolution
        s_t_m_x = rearrange(
            repeat(s_t_m_x, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_m, p_w=p_m),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_m_m = rearrange(
            repeat(s_t_m_m, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_m, p_w=p_m),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        s_t_l_x = rearrange(
            repeat(s_t_l_x, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_l, p_w=p_l),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_l_m = rearrange(
            repeat(s_t_l_m, "b t_h t_w t c -> b (t_h p_h) (t_w p_w) t c", p_h=p_l, p_w=p_l),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )

        # space-only tokens
        sp_x = rearrange(
            reduce(
                sp_x,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                t_h=self.num_tokens_per_dim,
                t_w=self.num_tokens_per_dim,
                reduction="mean",
            ),
            "b t_h t_w c -> b (t_h t_w) c",
        )
        sp_m = rearrange(
            reduce(
                sp_m,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                t_h=self.num_tokens_per_dim,
                t_w=self.num_tokens_per_dim,
                reduction="max",
            ),
            "b t_h t_w c -> b (t_h t_w) c",
        )

        # repeat time-only and static tokens over space
        t_x = repeat(t_x, "b t c -> b s t c", s=sp_x.shape[1])
        t_m = repeat(t_m, "b t c -> b s t c", s=sp_x.shape[1])

        st_x = repeat(st_x, "b c -> b s c", s=sp_x.shape[1])
        st_m = repeat(st_m, "b c -> b s c", s=sp_x.shape[1])

        # also include month as a feature, repeat over space
        month = repeat(month, "b c -> b s c", s=sp_x.shape[1])

        return (
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
            month,
        )

    def replace_masked_data(
        self,
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
        month,
        replace_with="last",
    ):
        assert replace_with in ["last", "median", "zeros", "nan"]

        # create an extra variable for each channel that indicates when the data was acquired
        # per default, everything is from the same timestep, so filled with zeros
        # TODO: Decide if we want to use zeros or placeholder values if data is replaced by an aggregate or fill value
        s_t_h_t = torch.zeros_like(s_t_h_x)
        s_t_m_t = torch.zeros_like(s_t_m_x)
        s_t_l_t = torch.zeros_like(s_t_l_x)
        sp_t = torch.zeros_like(sp_x)
        t_t = torch.zeros_like(t_x)
        st_t = torch.zeros_like(st_x)

        if replace_with == "median":
            # NOTE: for median replacement, we keep the acquisition time variable at zero, as we loose the temporal information
            s_t_h_x = self.replace_masked_data_with_aggregate(
                rearrange(s_t_h_x, "b s t c -> b s c t"),
                rearrange(s_t_h_m, "b s t c -> b s c t"),
                array_type="space_time_high_res",
            )
            s_t_m_x = self.replace_masked_data_with_aggregate(
                rearrange(s_t_m_x, "b s t c -> b s c t"),
                rearrange(s_t_m_m, "b s t c -> b s c t"),
                array_type="space_time_med_res",
            )
            s_t_l_x = self.replace_masked_data_with_aggregate(
                rearrange(s_t_l_x, "b s t c -> b s c t"),
                rearrange(s_t_l_m, "b s t c -> b s c t"),
                array_type="space_time_low_res",
            )
            sp_x = self.replace_masked_data_with_aggregate(
                rearrange(sp_x, "b s c -> b s c"),
                rearrange(sp_m, "b s c -> b s c"),
                array_type="space",
            )
            t_x = self.replace_masked_data_with_aggregate(
                rearrange(t_x, "b s t c -> b s c t"),
                rearrange(t_m, "b s t c -> b s c t"),
                array_type="time",
            )
            st_x = self.replace_masked_data_with_aggregate(
                rearrange(st_x, "b s c -> b s c"),
                rearrange(st_m, "b s c -> b s c"),
                array_type="static",
            )
        if replace_with == "last":
            s_t_h_x, s_t_h_t = self.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_h_x, "b s t c -> b s c t"),
                rearrange(s_t_h_m, "b s t c -> b s c t"),
                rearrange(s_t_h_t, "b s t c -> b s c t"),
                array_type="space_time_high_res",
            )
            s_t_m_x, s_t_m_t = self.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_m_x, "b s t c -> b s c t"),
                rearrange(s_t_m_m, "b s t c -> b s c t"),
                rearrange(s_t_m_t, "b s t c -> b s c t"),
                array_type="space_time_med_res",
            )
            s_t_l_x, s_t_l_t = self.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_l_x, "b s t c -> b s c t"),
                rearrange(s_t_l_m, "b s t c -> b s c t"),
                rearrange(s_t_l_t, "b s t c -> b s c t"),
                array_type="space_time_low_res",
            )
            t_x, t_t = self.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(t_x, "b s t c -> b s c t"),
                rearrange(t_m, "b s t c -> b s c t"),
                rearrange(t_t, "b s t c -> b s c t"),
                array_type="time",
            )
            # NOTE: for space-only and static data, we fall back to median replacement
            sp_x = self.replace_masked_data_with_aggregate(
                rearrange(sp_x, "b s c -> b s c"),
                rearrange(sp_m, "b s c -> b s c"),
                array_type="space",
            )
            # set all masked timestamps to -1
            sp_t = sp_t.masked_fill(sp_m.bool(), -1)
            st_x = self.replace_masked_data_with_aggregate(
                rearrange(st_x, "b s c -> b s c"),
                rearrange(st_m, "b s c -> b s c"),
                array_type="static",
            )
            st_t = st_t.masked_fill(st_m.bool(), -1)

        elif replace_with == "nan":
            # NOTE: for NaN replacement, we keep the acquisition time variable at zero, as we don't replace any data
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), float("nan"))
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), float("nan"))
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), float("nan"))
            sp_x = sp_x.masked_fill(sp_m.bool(), float("nan"))
            t_x = t_x.masked_fill(t_m.bool(), float("nan"))
            st_x = st_x.masked_fill(st_m.bool(), float("nan"))
        elif replace_with == "zeros":
            # NOTE: for zero replacement, we keep the acquisition time variable at zero, as we don't replace any data
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), 0.0)
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), 0.0)
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), 0.0)
            sp_x = sp_x.masked_fill(sp_m.bool(), 0.0)
            t_x = t_x.masked_fill(t_m.bool(), 0.0)
            st_x = st_x.masked_fill(st_m.bool(), 0.0)

        return (
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
            s_t_h_t,
            s_t_m_t,
            s_t_l_t,
            sp_t,
            t_t,
            st_t,
            month,
        )

    @staticmethod
    def concatenate_features_per_output_pixel(
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
        s_t_h_t,
        s_t_m_t,
        s_t_l_t,
        sp_t,
        t_t,
        st_t,
        month,
    ):
        """Concatenates all data of a single image into shapes of [B, S, N],
        where for RF, S is n_samples and N is n_features.
        """
        # for timeseries data, flatten time and channel dimension
        s_t_h_x = rearrange(s_t_h_x, "b s t c -> b s (t c)")
        s_t_h_m = rearrange(s_t_h_m, "b s t c -> b s (t c)")
        s_t_h_t = rearrange(s_t_h_t, "b s t c -> b s (t c)")
        s_t_m_x = rearrange(s_t_m_x, "b s t c -> b s (t c)")
        s_t_m_m = rearrange(s_t_m_m, "b s t c -> b s (t c)")
        s_t_m_t = rearrange(s_t_m_t, "b s t c -> b s (t c)")
        s_t_l_x = rearrange(s_t_l_x, "b s t c -> b s (t c)")
        s_t_l_m = rearrange(s_t_l_m, "b s t c -> b s (t c)")
        s_t_l_t = rearrange(s_t_l_t, "b s t c -> b s (t c)")
        t_x = rearrange(t_x, "b s t c -> b s (t c)")
        t_m = rearrange(t_m, "b s t c -> b s (t c)")
        t_t = rearrange(t_t, "b s t c -> b s (t c)")

        # concatenate all data along feature dimension
        x = torch.cat(
            [
                s_t_h_x,
                s_t_m_x,
                s_t_l_x,
                sp_x,
                t_x,
                st_x,
                s_t_h_t,
                s_t_m_t,
                s_t_l_t,
                sp_t,
                t_t,
                st_t,
                month,
            ],
            dim=2,
        )  # B, S, N
        m = torch.cat(
            [s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m, torch.zeros_like(month)], dim=2
        )  # B, S, N

        # check that no no-data values (-9999) are left
        assert not (x == -9999).any(), "No-data values (-9999) left in input."
        return x, m

    def fit_sklearn(
        self,
        id: str = "",
        seed: int = DEFAULT_SEED,
        hyperparameters: Dict = {},
        save_results: bool = False,
        normalization: str = "std",
        dataset_subset_size: int = 0,
        sweep_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ) -> Dict[str, float]:
        assert normalization in ["std", ""], f"Unknown normalization {normalization}"

        start_time = time()

        if hyperparameters == {}:
            hyperparameters = self.eval_config[f"hyperparameters_{self.model_type}"]

        assert (
            self.eval_config["cloud_generation"]["cloud_prob_pred_day"] == 0.0 or self.h5pys_only
        ), "Cloud generation is only supported with h5pys to this point."

        train_data_checkpoint_path = (
            Path(DATA_FOLDER) / self.eval_config["data"]["sklearn_train_data_checkpoint_folder"]
        )

        if train_data_checkpoint_path == "":
            train_data_checkpoint_path = None

        # see if checkpoint path has data, if so, load it and skip training preparation
        if (
            train_data_checkpoint_path is not None
            and (train_data_checkpoint_path / "sklearn_model_input.npy").exists()
            and (train_data_checkpoint_path / "sklearn_model_labels.npy").exists()
        ):
            print(
                f"Loading preprocessed training data from {train_data_checkpoint_path}", flush=True
            )
            model_input = np.load(train_data_checkpoint_path / "sklearn_model_input.npy")
            model_labels = np.load(train_data_checkpoint_path / "sklearn_model_labels.npy")

        else:
            train_ds = LandsatEvalDatasetSklearn(
                split="train",
                exclude_prediction_date=self.exclude_prediction_date,
                exclude_prediction_high_res=self.exclude_prediction_high_res,
                exclude_prediction_sensors=self.exclude_prediction_sensors,
                exclude_prediction_era5=self.exclude_prediction_era5,
                data_config=self.data_config,
                h5pys_only=self.h5pys_only,
            )

            if normalization == "std":
                normalizer = Normalizer(std=True, normalizing_dicts=self.normalizing_dict)
                train_ds.normalizer = normalizer

            if dataset_subset_size > 0:
                indices = random.sample(range(len(train_ds)), dataset_subset_size)
                subset_ds = Subset(train_ds, indices)

                train_dl = DataLoader(
                    subset_ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                )

            else:
                train_dl = DataLoader(
                    train_ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                )

            all_samples = []
            all_labels = []

            for input, label, _ in train_dl:
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
                    month,
                ) = input

                input = torch.squeeze(
                    self.concatenate_features_per_output_pixel(
                        *self.replace_masked_data(
                            *self.aggregate_data_per_output_pixel(
                                s_t_h_x=s_t_h_x,
                                s_t_m_x=s_t_m_x,
                                s_t_l_x=s_t_l_x,
                                sp_x=sp_x,
                                t_x=t_x,
                                st_x=st_x,
                                s_t_h_m=s_t_h_m,
                                s_t_m_m=s_t_m_m,
                                s_t_l_m=s_t_l_m,
                                sp_m=sp_m,
                                t_m=t_m,
                                st_m=st_m,
                                month=month,
                            ),
                        )
                    )[0]
                )  # (N, num_features)
                label = torch.squeeze(label).flatten()  # (N,)
                all_samples.append(input)
                all_labels.append(label)

            model_input = torch.cat(all_samples, dim=0).numpy()
            model_labels = torch.cat(all_labels, dim=0).numpy()

            if train_data_checkpoint_path is not None:
                train_data_checkpoint_path.mkdir(parents=True, exist_ok=True)
                np.save(train_data_checkpoint_path / "sklearn_model_input.npy", model_input)
                np.save(train_data_checkpoint_path / "sklearn_model_labels.npy", model_labels)

        if self.model_type == "rf":
            print("Training Random Forest Regressor...", flush=True)

            model = RandomForestRegressor(
                n_estimators=hyperparameters.get("n_estimators", 400),
                min_samples_leaf=hyperparameters.get("min_samples_leaf", 2),
                max_features=math.ceil(model_input.shape[-1] / 3)
                if hyperparameters.get("max_features") == "feature_dependent"
                else hyperparameters.get("max_features", "sqrt"),
                min_samples_split=hyperparameters.get("min_samples_split", 2),
                max_depth=hyperparameters.get("max_depth", 30),
                random_state=seed,
            )

        elif self.model_type == "svr":
            print("Training Support Vector Regressor...", flush=True)
            gamma = hyperparameters.get("gamma_base", 2) ** hyperparameters.get(
                "gamma_exponent", -5
            )
            degree = hyperparameters.get("degree", 3)
            C = hyperparameters.get("C_base", 2) ** hyperparameters.get("C_exponent", 0)
            print(f"Using gamma={gamma}, degree={degree}", flush=True)
            model = SVR(
                kernel=hyperparameters.get("kernel", "rbf"),
                gamma=gamma,
                degree=degree,
                C=C,
                max_iter=hyperparameters.get("max_iter", 5000),
                epsilon=hyperparameters.get("epsilon", 0.1),
            )

        elif self.model_type == "mlp":
            print("Training Multi-layer Perceptron Regressor...", flush=True)
            model = MLPRegressor(
                activation=hyperparameters.get("activation", "logistic"),
                hidden_layer_sizes=tuple(
                    hyperparameters.get("hidden_layer_sizes", [256, 128, 64])
                ),
                batch_size=hyperparameters.get("batch_size", 128),
                solver=hyperparameters.get("solver", "adam"),
                alpha=hyperparameters.get("alpha", 0.00001),
                random_state=seed,
                learning_rate_init=hyperparameters.get("learning_rate_init", 0.001),
                max_iter=hyperparameters.get("max_iter", 1000),
                early_stopping=True,
                n_iter_no_change=20,
            )

        else:
            raise ValueError(f"Unknown model type {self.model_type}")

        if hyperparameters.get("bagging", False):
            model_composed = BaggingRegressor(estimator=model)
        else:
            model_composed = model

        model_composed.fit(model_input, model_labels)

        if self.model_type == "mlp":
            print(
                f"MLP training stopped after {model_composed.n_iter_} iterations with training loss {model_composed.loss_:.4f}",
                flush=True,
            )

        end_time = time()
        print(f"Training time: {end_time - start_time:.2f} seconds", flush=True)

        test_ds = LandsatEvalDatasetSklearn(
            split="test",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            exclude_prediction_era5=self.exclude_prediction_era5,
            data_config=self.data_config,
            h5pys_only=self.h5pys_only,
        )

        if normalization == "std":
            normalizer = Normalizer(std=True, normalizing_dicts=self.normalizing_dict)
            test_ds.normalizer = normalizer

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        all_preds = []
        all_test_labels = []

        for input, label, _ in test_dl:
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
                month,
            ) = input
            input = torch.squeeze(
                self.concatenate_features_per_output_pixel(
                    *self.replace_masked_data(
                        *self.aggregate_data_per_output_pixel(
                            s_t_h_x=s_t_h_x,
                            s_t_m_x=s_t_m_x,
                            s_t_l_x=s_t_l_x,
                            sp_x=sp_x,
                            t_x=t_x,
                            st_x=st_x,
                            s_t_h_m=s_t_h_m,
                            s_t_m_m=s_t_m_m,
                            s_t_l_m=s_t_l_m,
                            sp_m=sp_m,
                            t_m=t_m,
                            st_m=st_m,
                            month=month,
                        ),
                    )
                )[0]
            )  # (N, num_features)
            preds = model_composed.predict(input.numpy())
            all_preds.append(torch.as_tensor(preds))
            all_test_labels.append(torch.squeeze(label).flatten())

        test_preds = torch.cat(all_preds, dim=0).numpy()
        test_labels = torch.cat(all_test_labels, dim=0).numpy()

        results = compute_regression_metrics(preds=test_preds, target=test_labels)

        if sweep_run is not None:
            sweep_run.log(results)

        if save_results:
            # model checkpoint
            try:
                model_path = Path(f"./landsat_{self.model_type}_model_{id}.joblib")
                joblib.dump(model_composed, model_path)
                print(f"Saved {self.model_type} model to {model_path}", flush=True)
            except Exception as e:
                print(f"Could not save {self.model_type} model due to {e}", flush=True)

            # results
            results_path = Path(f"./landsat_{self.model_type}_results_{id}.json")
            with results_path.open("w") as f:
                json.dump(results, f)

        return results

    def predict_only(
        self,
        model: Union[RandomForestRegressor, SVR, MLPRegressor],
        id: str = "",
        save_results: bool = False,
        normalization: str = "std",
    ) -> Dict[str, float]:
        assert normalization in ["std", ""], f"Unknown normalization {normalization}"

        test_ds = LandsatEvalDatasetSklearn(
            split="test",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            exclude_prediction_sensors=self.exclude_prediction_sensors,
            exclude_prediction_era5=self.exclude_prediction_era5,
            data_config=self.data_config,
            h5pys_only=self.h5pys_only,
        )

        if normalization == "std":
            normalizer = Normalizer(std=True, normalizing_dicts=self.normalizing_dict)
            test_ds.normalizer = normalizer

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        all_preds = []
        all_test_labels = []

        if save_results:
            # create a csv to store results
            results_folder = RESULTS_FOLDER
            results_path = results_folder / id
            results_csv_path = results_folder / f"{self.model_type}_{id}.csv"

            results_path.mkdir(parents=True, exist_ok=True)
            results_csv_path.touch(exist_ok=True)

            # create header if file is empty
            if results_csv_path.stat().st_size == 0:
                with open(results_csv_path, "w") as f:
                    f.write("filename,rmse\n")

        for input, label, filename in test_dl:
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
                month,
            ) = input
            input = torch.squeeze(
                self.concatenate_features_per_output_pixel(
                    *self.replace_masked_data(
                        *self.aggregate_data_per_output_pixel(
                            s_t_h_x=s_t_h_x,
                            s_t_m_x=s_t_m_x,
                            s_t_l_x=s_t_l_x,
                            sp_x=sp_x,
                            t_x=t_x,
                            st_x=st_x,
                            s_t_h_m=s_t_h_m,
                            s_t_m_m=s_t_m_m,
                            s_t_l_m=s_t_l_m,
                            sp_m=sp_m,
                            t_m=t_m,
                            st_m=st_m,
                            month=month,
                        ),
                    )
                )[0]
            )  # (N, num_features)
            preds = model.predict(input.numpy())
            all_preds.append(torch.as_tensor(preds))
            all_test_labels.append(torch.squeeze(label).flatten())

            label_to_save = torch.squeeze(label).numpy()
            pred_to_save = torch.as_tensor(preds).numpy().reshape(label_to_save.shape)

            # save predictions and labels for each sample
            if save_results:
                run_folder = Path(f"./{id}")
                run_folder.mkdir(exist_ok=True)
                sample_id = filename[0].split(".tif")[0]
                sample_preds_path = Path(f"./{run_folder}/{sample_id}_{self.model_type}_preds.npy")
                sample_labels_path = Path(
                    f"./{run_folder}/{sample_id}_{self.model_type}_labels.npy"
                )
                np.save(sample_preds_path, pred_to_save)
                np.save(sample_labels_path, label_to_save)

                rmse = root_mean_squared_error(label_to_save.flatten(), pred_to_save.flatten())

                # append results to csv with filename, r2, rmse
                with open(results_csv_path, "a") as f:
                    f.write(f"{filename[0]},{rmse}\n")

        test_preds = torch.cat(all_preds, dim=0).numpy()
        test_labels = torch.cat(all_test_labels, dim=0).numpy()

        results = compute_regression_metrics(preds=test_preds, target=test_labels)

        if save_results:
            # results
            results_path = Path(f"./{self.model_type}_{id}.json")
            with results_path.open("w") as f:
                json.dump(results, f)

        return results
