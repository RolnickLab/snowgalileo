import json
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
from torch.utils.data import DataLoader

from src.data.dataset import Normalizer
from src.data.earthengine.eo_eval import (
    SPACE_TIME_HIGH_RES_BANDS,
)
from src.eval.landsat_eval import LandsatEval, LandsatEvalDataset, masked_output_np_to_tensor


class LandsatEvalDatasetRandomForest(LandsatEvalDataset):
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


class LandsatEvalRandomForest(LandsatEval):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        normalization: Union[str, Normalizer] = "std",  # or "scaling"
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        resample: bool = False,
        eval_config: Dict = {},
    ):
        self.normalization = normalization
        self.exclude_prediction_date = exclude_prediction_date
        self.exclude_prediction_high_res = exclude_prediction_high_res
        self.resample = resample
        self.name = "ls_rf"

        super().__init__(
            normalization=normalization,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            resample=resample,
            eval_config=eval_config,
        )

    def remove_masked_data_and_flatten(
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
        # returns: (N) where N is the number of unmasked values
        assert s_t_h_x.shape == s_t_h_m.shape
        assert s_t_m_x.shape == s_t_m_m.shape
        assert s_t_l_x.shape == s_t_l_m.shape
        assert sp_x.shape == sp_m.shape
        assert t_x.shape == t_m.shape
        assert st_x.shape == st_m.shape
        x = torch.cat(
            [
                s_t_h_x.flatten(),
                s_t_m_x.flatten(),
                s_t_l_x.flatten(),
                sp_x.flatten(),
                t_x.flatten(),
                st_x.flatten(),
                month.flatten(),
            ]
        )
        m = torch.cat(
            [
                s_t_h_m.flatten(),
                s_t_m_m.flatten(),
                s_t_l_m.flatten(),
                sp_m.flatten(),
                t_m.flatten(),
                st_m.flatten(),
                torch.zeros_like(month).flatten(),  # month is never masked
            ]
        )
        assert x.shape == m.shape
        return x[m == 0]

    # TODO: TEST THIS FUNCTION!
    # TODO: what to do if the first value is masked?
    def forward_filling_masked_data_per_channel_else_median(
        self,
        x,
        m,
        t,
    ):
        # shape: (B, (S), C, (T))
        # for timeseries data:
        # for each channel, replaces masked values with the last unmasked value over timestep for this channel
        # for space-only and static data:
        # replaces masked values with the last unmasked value over all channels in the same data group
        # perform forward filling per channel

        x = torch.masked_fill(x, m.bool(), float("nan"))

        for i in range(x.shape[-2]):
            if x.dim() == 3:
                # space-only or static data
                channel_data = x[..., i]
                channel_mask = m[..., i]
                # as we don't have a time dimension here, we take the median over all channels in the same data group
                x[..., i] = torch.nanmedian(channel_data, dim=-1, keepdim=True)[0][..., i]
            else:
                channel_data = x[..., i, :]
                channel_mask = m[..., i, :]
                channel_time_distance = t[..., i, :]
                if torch.all(channel_mask):
                    # all values are masked, replace with median per channel group
                    x[..., i, :] = torch.nanmedian(x, dim=-1, keepdim=True)[0][..., i, :]
                else:
                    last_valid_timestep = torch.nan
                    current_timestep = 0
                    last_valid_value = torch.nan
                    for timestep in range(channel_data.shape[-1]):
                        if not channel_mask[..., timestep]:
                            last_valid_value = channel_data[..., timestep]
                            last_valid_timestep = current_timestep
                            current_timestep += 1
                        else:
                            if torch.isnan(last_valid_value):
                                # no valid value found yet, replace with median per channel group
                                channel_data[..., timestep] = torch.nanmedian(
                                    x, dim=-1, keepdim=True
                                )[0][..., i, :]
                            else:
                                channel_data[..., timestep] = last_valid_value
                                channel_time_distance[..., timestep] = (
                                    current_timestep - last_valid_timestep
                                )
                            current_timestep += 1

                    x[..., i, :] = channel_data
                    t[..., i, :] = channel_time_distance

        # assert there are no NaNs left
        assert not torch.isnan(x).any(), "There are still NaNs left after forward filling."
        return x, t

    def replace_masked_data_with_median_per_channel(
        self,
        x,
        m,
    ):
        x = torch.masked_fill(x, m.bool(), float("nan"))

        # for timeseries data:
        # for each channel, replaces NaNs with mean over timestep for this channel
        # for space-only and static data:
        # replaces NaNs with mean over all channels in the same data group
        x = torch.where(torch.isnan(x), torch.nanmedian(x, dim=-1, keepdim=True), x)

        # if there are still NaNs (all values in the timeseries (for timeseries) or
        # data group (for space-only and static data) were masked):
        # replace mean over timesteps (all channels in the same data group)
        # or replace with spatial mean (for space-only and static data)
        x = torch.where(torch.isnan(x), torch.nanmedian(x, dim=-2, keepdim=True), x)

        return x

    # this option ends up in shapes of [B, S, N] where for RF, S is n_samples and n_features.
    # the dimension of N is (C * T) for space-time tokens, C for space tokens, (C * T) for time tokens, and C for static tokens
    # concatenated
    # Next step would we to concatenate along S
    def aggregate_per_output_pixel_and_replace_masked_data(
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
        # replace masked data with (A) the last timestep of this sensor, (B) the median over time of this sensor, (C) zeros, (D) NaNs to be handled by RF.
        # RF computes median for missing values (?)
        # TODO: replace all invalid with mean per timestep
        assert replace_with in ["last", "median", "zeros", "nan"]

        # TODO: make this more dynamic
        patch_size_high_res = 10
        p_m = patch_size_high_res // s_t_m_x.shape[1]
        p_l = patch_size_high_res // s_t_l_x.shape[1]

        # first, bring all data into token resolution (the output resolution)
        # t_h = token height, t_w = token width
        s_t_h_x = rearrange(
            reduce(
                s_t_h_x,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="mean",
            ),
            "b t_h t_w t c -> b (t_h t_w) t c",
        )
        s_t_h_m = rearrange(
            reduce(
                s_t_h_m,
                "b (t_h p_h) (t_w p_w) t c -> b t_h t_w t c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="max",  # if one value is masked, the entire patch is masked
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

        sp_x = rearrange(
            reduce(
                sp_x,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="mean",
            ),
            "b t_h t_w c -> b (t_h t_w) c",
        )
        sp_m = rearrange(
            reduce(
                sp_m,
                "b (t_h p_h) (t_w p_w) c -> b t_h t_w c",
                p_h=patch_size_high_res,
                p_w=patch_size_high_res,
                reduction="max",  # if one value is masked, the entire patch is masked
            ),
            "b t_h t_w c -> b (t_h t_w) c",
        )

        # repeat time tokens over space
        t_x = repeat(t_x, "b t c -> b s t c", s=sp_x.shape[1])
        t_m = repeat(t_m, "b t c -> b s t c", s=sp_x.shape[1])

        st_x = repeat(st_x, "b c -> b s c", s=sp_x.shape[1])
        st_m = repeat(st_m, "b c -> b s c", s=sp_x.shape[1])

        # also include month as a feature, repeat over space
        month = repeat(month, "b c -> b s c", s=sp_x.shape[1])

        assert s_t_h_x.shape[1] == 100

        # create an extra variable for each channel that indicates when the data was acquired
        # per default, everything is from the same timestep, so filled with zeros
        s_t_h_t = torch.zeros_like(s_t_h_x)
        s_t_m_t = torch.zeros_like(s_t_m_x)
        s_t_l_t = torch.zeros_like(s_t_l_x)
        sp_t = torch.zeros_like(sp_x)
        t_t = torch.zeros_like(t_x)
        st_t = torch.zeros_like(st_x)

        if replace_with == "median":
            # NOTE: for median replacement, we set the acquisition time to zero, as we lose the temporal information
            s_t_h_x = self.replace_masked_data_with_median_per_channel(
                rearrange(s_t_h_x, "b s t c -> b s c t"), rearrange(s_t_h_m, "b s t c -> b s c t")
            )
            s_t_m_x = self.replace_masked_data_with_median_per_channel(
                rearrange(s_t_m_x, "b s t c -> b s c t"), rearrange(s_t_m_m, "b s t c -> b s c t")
            )
            s_t_l_x = self.replace_masked_data_with_median_per_channel(
                rearrange(s_t_l_x, "b s t c -> b s c t"), rearrange(s_t_l_m, "b s t c -> b s c t")
            )
            sp_x = self.replace_masked_data_with_median_per_channel(
                rearrange(sp_x, "b s c -> b s c"), rearrange(sp_m, "b s c -> b s c")
            )
            t_x = self.replace_masked_data_with_median_per_channel(
                rearrange(t_x, "b s t c -> b s c t"), rearrange(t_m, "b s t c -> b s c t")
            )
            st_x = self.replace_masked_data_with_median_per_channel(
                rearrange(st_x, "b s c -> b s c"), rearrange(st_m, "b s c -> b s c")
            )
        if replace_with == "last":
            s_t_h_x, s_t_h_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(s_t_h_x, "b s t c -> b s c t"),
                rearrange(s_t_h_m, "b s t c -> b s c t"),
                rearrange(s_t_h_t, "b s t c -> b s c t"),
            )
            s_t_m_x, s_t_m_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(s_t_m_x, "b s t c -> b s c t"),
                rearrange(s_t_m_m, "b s t c -> b s c t"),
                rearrange(s_t_m_t, "b s t c -> b s c t"),
            )
            s_t_l_x, s_t_l_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(s_t_l_x, "b s t c -> b s c t"),
                rearrange(s_t_l_m, "b s t c -> b s c t"),
                rearrange(s_t_l_t, "b s t c -> b s c t"),
            )
            sp_x, sp_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(sp_x, "b s c -> b s c"),
                rearrange(sp_m, "b s c -> b s c"),
                rearrange(sp_t, "b s c -> b s c"),
            )
            t_x, t_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(t_x, "b s t c -> b s c t"),
                rearrange(t_m, "b s t c -> b s c t"),
                rearrange(t_t, "b s t c -> b s c t"),
            )
            st_x, st_t = self.forward_filling_masked_data_per_channel_else_median(
                rearrange(st_x, "b s c -> b s c"),
                rearrange(st_m, "b s c -> b s c"),
                rearrange(st_t, "b s c -> b s c"),
            )
        elif replace_with == "nan":
            # NOTE: for NaN replacement, we keep the acquisition time as is, as we don't change the data
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), float("nan"))
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), float("nan"))
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), float("nan"))
            sp_x = sp_x.masked_fill(sp_m.bool(), float("nan"))
            t_x = t_x.masked_fill(t_m.bool(), float("nan"))
            st_x = st_x.masked_fill(st_m.bool(), float("nan"))
        elif replace_with == "zeros":
            # NOTE: for zero replacement, we set the acquisition time to zero, as we lose the temporal information
            s_t_h_x = s_t_h_x.masked_fill(s_t_h_m.bool(), 0.0)
            s_t_m_x = s_t_m_x.masked_fill(s_t_m_m.bool(), 0.0)
            s_t_l_x = s_t_l_x.masked_fill(s_t_l_m.bool(), 0.0)
            sp_x = sp_x.masked_fill(sp_m.bool(), 0.0)
            t_x = t_x.masked_fill(t_m.bool(), 0.0)
            st_x = st_x.masked_fill(st_m.bool(), 0.0)

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
        assert not (x == -9999).any(), (
            "No-data values (-9999) left in input after replacing masked data"
        )

        return x, m

    def fit_random_forest(self, id: str):
        train_ds = LandsatEvalDatasetRandomForest(
            split="train",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            data_config=self.data_config,
        )
        # NOTE: no normalization here, since RF works better without normalization!
        # NOTE (Update): our experiments show that normalization helps RF as well

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
            input, _ = torch.squeeze(
                self.aggregate_per_output_pixel_and_replace_masked_data(
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
                )
            )  # (N, num_features)
            label = torch.squeeze(label).flatten()  # (N,)
            all_samples.append(input)
            all_labels.append(label)

        rf_input = torch.cat(all_samples, dim=0).numpy()
        rf_labels = torch.cat(all_labels, dim=0).numpy()

        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(rf_input, rf_labels)

        # save the model
        try:
            model_path = Path(f"./landsat_rf_model_{id}.joblib")
            joblib.dump(regr, model_path)
            print(f"Saved Random Forest model to {model_path}", flush=True)
        except Exception as e:
            print(f"Could not save Random Forest model due to {e}", flush=True)

        test_ds = LandsatEvalDatasetRandomForest(
            split="test",
            exclude_prediction_date=self.exclude_prediction_date,
            exclude_prediction_high_res=self.exclude_prediction_high_res,
            data_config=self.data_config,
        )

        # TODO: make sure that really no normalization takes place.

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
                self.aggregate_per_output_pixel_and_replace_masked_data(
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
                )[0]
            )  # (N, num_features)
            preds = regr.predict(input.numpy())
            all_preds.append(torch.as_tensor(preds))
            all_test_labels.append(torch.squeeze(label).flatten())

        test_preds = torch.cat(all_preds, dim=0).numpy()
        test_labels = torch.cat(all_test_labels, dim=0).numpy()

        rmse = root_mean_squared_error(test_labels, test_preds)
        print(f"Test RMSE: {rmse}", flush=True)
        r2 = r2_score(test_labels, test_preds)
        print(f"Test R2: {r2}", flush=True)

        print("Training pipeline complete.", flush=True)

        # store results as json
        results = {
            "test_rmse": float(rmse),
            "test_r2": float(r2),
        }
        results_path = Path(f"./landsat_rf_results_{id}.json")
        with results_path.open("w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    id = "test"
    with (Path(__file__).parents[0] / Path("eval_configs") / Path("landsat_eval_5_95.json")).open(
        "r"
    ) as f:
        config = json.load(f)
    rf = LandsatEvalRandomForest(
        normalization="std",
        exclude_prediction_date=False,
        exclude_prediction_high_res=False,
        resample=False,
        eval_config=config,
    )
    rf.fit_random_forest(id)
