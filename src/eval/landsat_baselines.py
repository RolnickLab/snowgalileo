import json
import math
from pathlib import Path
from typing import Dict, Optional, Union, cast

import joblib
import numpy as np
import rioxarray
import torch
import xarray as xr
from einops import rearrange, reduce, repeat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from torch.utils.data import DataLoader

from src.config import DEFAULT_SEED
from src.data.config import NORMALIZATION_DICT_FILENAME
from src.data.dataset import Normalizer
from src.data.earthengine.eo_eval import (
    SPACE_TIME_HIGH_RES_BANDS,
)
from src.eval.landsat_eval import LandsatEval, LandsatEvalDataset, masked_output_np_to_tensor
from src.utils import config_dir
from src.data.config import NORMALIZATION_DICT_FILENAME


class LandsatEvalDatasetSklearn(LandsatEvalDataset):
    """
    The Random Forest baseline uses the same dataset as the main LandsatEvalDataset,
    but doesn't group channel masks into channel group masks, so that we can directly
    remove masked channels for the Random Forest input.
    """

    def __init__(
        self,
        split: str = "train",
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        normalizer: Optional[Normalizer] = None,
        data_config: Dict = {},
    ):
        super().__init__(
            split=split,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            normalizer=normalizer,
            data_config=data_config,
        )

    def mask_prediction_high_res(self, s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m):
        # masks the high resolution, optical data in the prediction timestep
        # high resolution channels are: s1, s2, landsat, so we retain the first 3 channels
        # NOTE: 0 = valid, 1 = masked
        print("Masking high resolution data in prediction timestep", flush=True)
        assert self.exclude_prediction_high_res
        assert s_t_h_m.shape[-1] == len(SPACE_TIME_HIGH_RES_BANDS)
        s_t_h_m[:, :, -1, 3:] = 1
        return s_t_h_m, s_t_m_m, s_t_l_m, sp_m, t_m, st_m

    def __getitem__(self, idx):
        h5py = self.load_tif(idx)
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

        label = self.label_tifs[idx]
        # TODO: optinally add conversion to h5pys for labels
        with cast(xr.Dataset, rioxarray.open_rasterio(label)) as data:
            label = cast(np.ndarray, data.values)
            # remove first dimension
            label = np.squeeze(label, axis=0)
            print(f"Label shape: {label.shape}", flush=True)

        # if assertion is triggered, go to the next tif file
        try:
            assert self.input_tifs[idx].name == self.label_tifs[idx].name, (
                f"Input path {self.input_tifs[idx].name} and label path {self.label_tifs[idx].name} do not match."
            )
        except AssertionError:
            print(
                f"Label shape {label.shape} does not match expected shape ({self.label_height_width}, {self.label_height_width}) for {self.label_tifs[idx].name}"
            )
            self.label_tifs[idx] = (
                self.label_tifs[idx + 1]
                if idx < len(self.label_tifs) - 1
                else self.label_tifs[idx - 1]
            )
            return self.__getitem__(idx)

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
            self.input_tifs[idx].name,  # for logging purposes
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
        num_tokens_per_dim: int = 10,
        resample: bool = False,
        eval_config: Dict = {},
        model_type: str = "rf",
        normalizing_dict: Optional[Dict] = None,
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.num_tokens_per_dim = num_tokens_per_dim
        self.resample = resample
        self.name = "ls_rf"
        self.model_type = model_type
        self.normalizing_dict = normalizing_dict

        assert model_type in ["rf", "svr", "mlp"], f"Unknown model type {model_type}"

        super().__init__(
            normalization=normalization,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            resample=resample,
            eval_config=eval_config,
        )

    @staticmethod
    def forward_filling_masked_data_per_channel_else_aggregate(x, m, t, array_type: str, normalizing_dict=None):
        """Fills masked values in x by forward-filling along the time dimension per channel.
        Remaining NaNs will fall back to aggregation replacement."""

        assert x.dim() == 4, f"Expected 4D tensor, got shape {x.shape}"

        x = x.clone()
        m = m.bool()
        t = t.clone()

        x = torch.masked_fill(x, m, float("nan"))
        valid_mask = ~torch.isnan(x)

        # accumulate last valid value along time axis
        # use "expanding max index" trick
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
        x = LandsatEvalSklearn.replace_masked_data_with_aggregate(x, m, array_type=array_type, normalizing_dict=normalizing_dict)

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
        Breaks early if no NaNs are left after a replacement step."""
        x = torch.masked_fill(x, m.bool(), float("nan"))
        for d in dims:
            x = torch.where(torch.isnan(x), torch.nanmedian(x, dim=d, keepdim=True).values, x)
            if not torch.isnan(x).any():
                break

        return x

    @staticmethod
    def replace_masked_data_with_aggregate(x, m, array_type: str, normalizing_dict=None):
        """Replaces masked values in x with an aggregate.
        First, tries to replace over the time dimension for timeseries data,
        then over the space dimension. For space-only and static data, replaces over space dimension.
        If no replacement is possible, fill with the per-channel mean of the pre-training data."""
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
            assert normalizing_dict is not None, "normalizing_dict must be provided for final fill value replacement."
            # fill remaining NaNs with per-channel mean from normalizing_dict
            if array_type not in normalizing_dict:
                raise ValueError(f"Unknown array type: {array_type}")
            channel_means = torch.tensor(normalizing_dict[array_type]["mean"], device=x.device)
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
        """
        Aggregates input data per output pixel by bringing all data to the output resolution.
        This means upsampling medium and low resolution data,
        and aggregating high resolution data to output pixel resolution.
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

    @classmethod
    def replace_masked_data(
        cls,
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
        normalizing_dict=None,
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
            s_t_h_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(s_t_h_x, "b s t c -> b s c t"), rearrange(s_t_h_m, "b s t c -> b s c t"), array_type="space_time_high_res", normalizing_dict=normalizing_dict
            )
            s_t_m_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(s_t_m_x, "b s t c -> b s c t"), rearrange(s_t_m_m, "b s t c -> b s c t"), array_type="space_time_med_res", normalizing_dict=normalizing_dict
            )
            s_t_l_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(s_t_l_x, "b s t c -> b s c t"), rearrange(s_t_l_m, "b s t c -> b s c t"), array_type="space_time_low_res", normalizing_dict=normalizing_dict
            )
            sp_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(sp_x, "b s c -> b s c"), rearrange(sp_m, "b s c -> b s c"), array_type="space", normalizing_dict=normalizing_dict
            )
            t_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(t_x, "b s t c -> b s c t"), rearrange(t_m, "b s t c -> b s c t"), array_type="time", normalizing_dict=normalizing_dict
            )
            st_x = LandsatEvalSklearn.replace_masked_data_with_aggregate(
                rearrange(st_x, "b s c -> b s c"), rearrange(st_m, "b s c -> b s c"), array_type="static", normalizing_dict=normalizing_dict
            )
        if replace_with == "last":
            s_t_h_x, s_t_h_t = cls.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_h_x, "b s t c -> b s c t"),
                rearrange(s_t_h_m, "b s t c -> b s c t"),
                rearrange(s_t_h_t, "b s t c -> b s c t"),
                array_type="space_time_high_res",
                normalizing_dict=normalizing_dict,
            )
            s_t_m_x, s_t_m_t = cls.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_m_x, "b s t c -> b s c t"),
                rearrange(s_t_m_m, "b s t c -> b s c t"),
                rearrange(s_t_m_t, "b s t c -> b s c t"),
                array_type="space_time_med_res",
                normalizing_dict=normalizing_dict,
            )
            s_t_l_x, s_t_l_t = cls.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(s_t_l_x, "b s t c -> b s c t"),
                rearrange(s_t_l_m, "b s t c -> b s c t"),
                rearrange(s_t_l_t, "b s t c -> b s c t"),
                array_type="space_time_low_res",
                normalizing_dict=normalizing_dict,
            )
            t_x, t_t = cls.forward_filling_masked_data_per_channel_else_aggregate(
                rearrange(t_x, "b s t c -> b s c t"),
                rearrange(t_m, "b s t c -> b s c t"),
                rearrange(t_t, "b s t c -> b s c t"),
                array_type="time",
                normalizing_dict=normalizing_dict,    
            )
            # NOTE: for space-only and static data, we fall back to median replacement
            sp_x = cls.replace_masked_data_with_aggregate(
                rearrange(sp_x, "b s c -> b s c"),
                rearrange(sp_m, "b s c -> b s c"),
                array_type="space",
                normalizing_dict=normalizing_dict,
            )
            # set all masked timestamps to -1
            sp_t = sp_t.masked_fill(sp_m.bool(), -1)
            st_x = cls.replace_masked_data_with_aggregate(
                rearrange(st_x, "b s c -> b s c"),
                rearrange(st_m, "b s c -> b s c"),
                array_type="static",
                normalizing_dict=normalizing_dict,
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
        """
        Concatenates all data of a single image into shapes of [B, S, N], where for RF, S is n_samples and N is n_features.
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
        hyperparameters: Dict = {},
        save_results: bool = False,
        normalization: str = "std",
    ) -> Dict[str, float]:
        assert normalization in ["std", ""], f"Unknown normalization {normalization}"

        if hyperparameters == {}:
            hyperparameters = self.eval_config[f"hyperparameters_{self.model_type}"]

        train_ds = LandsatEvalDatasetSklearn(
            split="train",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            data_config=self.data_config,
        )

        # we use the normalization values for missing data imputation so we load it independently
        normalizing_dict = train_ds.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
        )

        if normalization == "std":
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
            train_ds.normalizer = normalizer

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
                        normalizing_dict=normalizing_dict,
                    )
                )[0]
            )  # (N, num_features)
            label = torch.squeeze(label).flatten()  # (N,)
            all_samples.append(input)
            all_labels.append(label)

        model_input = torch.cat(all_samples, dim=0).numpy()
        model_labels = torch.cat(all_labels, dim=0).numpy()

        if self.model_type == "rf":
            print("Training Random Forest Regressor...", flush=True)
            model = RandomForestRegressor(
                n_estimators=hyperparameters["n_estimators"],
                min_samples_leaf=hyperparameters["min_samples_leaf"],
                max_features=math.ceil(
                    model_input.shape[-1] / 3
                ),  # fixed here, since depends on input shape
                random_state=DEFAULT_SEED,
            )

        elif self.model_type == "svr":
            print("Training Support Vector Regressor...", flush=True)
            gamma = hyperparameters["gamma_base"] ** hyperparameters["gamma_exponent"]
            degree = hyperparameters["degree"]
            C = hyperparameters["C_base"] ** hyperparameters["C_exponent"]
            print(f"Using gamma={gamma}, degree={degree}", flush=True)
            model = SVR(kernel=hyperparameters["kernel"], gamma=gamma, degree=degree, C=C)

        elif self.model_type == "mlp":
            print("Training Multi-layer Perceptron Regressor...", flush=True)
            model = MLPRegressor(
                hidden_layer_sizes=(model_input.shape[-1],),
                random_state=DEFAULT_SEED,
                learning_rate_init=hyperparameters["learning_rate_init"],
            )

        else:
            raise ValueError(f"Unknown model type {self.model_type}")

        model.fit(model_input, model_labels)

        test_ds = LandsatEvalDatasetSklearn(
            split="test",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            data_config=self.data_config,
        )

        if normalization == "std":
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
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
                        normalizing_dict=normalizing_dict,
                    )
                )[0]
            )  # (N, num_features)
            preds = model.predict(input.numpy())
            all_preds.append(torch.as_tensor(preds))
            all_test_labels.append(torch.squeeze(label).flatten())

        test_preds = torch.cat(all_preds, dim=0).numpy()
        test_labels = torch.cat(all_test_labels, dim=0).numpy()

        rmse = root_mean_squared_error(test_labels, test_preds)
        r2 = r2_score(test_labels, test_preds)

        results = {
            "rmse": float(rmse),
            "r2": float(r2),
        }

        if save_results:
            # model checkpoint
            try:
                model_path = Path(f"./landsat_{self.model_type}_model_{id}.joblib")
                joblib.dump(model, model_path)
                print(f"Saved {self.model_type} model to {model_path}", flush=True)
            except Exception as e:
                print(f"Could not save {self.model_type} model due to {e}", flush=True)

            # results
            results_path = Path(f"./landsat_{self.model_type}_results_{id}.json")
            with results_path.open("w") as f:
                json.dump(results, f)

        return results


if __name__ == "__main__":
    id = "test"
    with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval_5_95.json")).open(
        "r"
    ) as f:
        config = json.load(f)

    rf = LandsatEvalSklearn(
        normalization="std",
        exclude_prediction_date=False,
        exclude_prediction_high_res=False,
        resample=False,
        eval_config=config,
        model_type="rf",
    )
    rf.fit_sklearn(id)
