import random
from pathlib import Path
from typing import Dict, Union, cast

import numpy as np
import psutil
import rioxarray
import satellite_cloud_generator as scg
import torch
import xarray as xr
from einops import rearrange

from src.config import DEFAULT_SEED
from src.data.config import (
    DATASET_OUTPUT_HW_HIGH_RES,
    MODALITIES,
    MODIS_FILL_VALUE,
    NDI_VALID_DATA_BOUNDS,
    NO_DATA_VALUE,
    NORMALIZATION_DICT_FILENAME,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_TIMESTEPS,
)
from src.data.dataset import DatasetOutput, Normalizer, to_cartesian
from src.data.earthengine.eo_eval import (
    CLOUD_BANDS,
    EE_SPACE_BANDS,
    EE_WC_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    EO_SPACE_TIME_LOW_RES_BANDS,
    ESA_WORLDCOVER_BAND_INDEX,
    SPACE_TIME_LOW_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS,
    STATIC_BANDS,
    TIME_BANDS,
)
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import config_dir, seed_everything

seed_everything(DEFAULT_SEED)
process = psutil.Process()

# NOTE: Scaling factors according to Earthengine documentation for the specific bands
CHANNEL_WISE_CLOUD_PARAMETERS: Dict[str, Dict] = {
    "s_t_h_x": {
        "S1": {
            "band_names": ["VV", "VH", "angle"],
            "apply_clouds": [False, False, False],
            "channel_magnitudes": [0.0, 0.0, 0.0],
            "scaling_factors": [1.0, 1.0, 1.0],
        },
        "S2": {
            "band_names": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "apply_clouds": [True, True, True, True, True, True],
            "channel_magnitudes": [0.3252, 0.3036, 0.3235, 0.3716, 0.2770, 0.2563],
            "scaling_factors": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        },
        "Landsat": {
            "band_names": [
                "B2_landsat",
                "B3_landsat",
                "B4_landsat",
                "B5_landsat",
                "B6_landsat",
                "B7_landsat",
            ],
            "apply_clouds": [True, True, True, True, True, True],
            "channel_magnitudes": [0.5157, 0.4304, 0.4384, 0.4473, 0.1760, 0.2105],
            "scaling_factors": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
    },
    "s_t_m_x": {
        "S3": {
            "band_names": ["Oa17_radiance", "Oa21_radiance"],
            "apply_clouds": [True, True],
            "channel_magnitudes": [1.7589,  0.3781],
            "scaling_factors": [0.00493004, 0.00324118],
        },
    },
    # NOTE: Indeces computation happens after cloud generation
    "s_t_l_x": {
        "MODIS": {
            "band_names": [
                "sur_refl_b01",
                "sur_refl_b02",
                "sur_refl_b03",
                "sur_refl_b04",
                "sur_refl_b05",
                "sur_refl_b06",
                "sur_refl_b07",
            ],
            "apply_clouds": [True, True, True, True, True, True, True],
            "channel_magnitudes": [0.8411, 0.8400, 0.7846, 0.8301, 0.7175, 0.5384, 0.4205],
            "scaling_factors": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        },
        "VIIRS": {
            "band_names": ["I1", "I3"],
            "apply_clouds": [True, True],
            "channel_magnitudes": [1.1923, 0.6415],
            "scaling_factors": [1.0, 1.0],
        },
    },
    "t_x": {
        "VIIRS": {
            "band_names": ["M5", "M7", "M10", "M11"],
            "apply_clouds": [True, True, True, True],
            "channel_magnitudes": [0.3277, 0.4289, 0.1601, 0.1406],
            "scaling_factors": [1.0, 1.0, 1.0, 1.0],
        },
        "ERA5": {
            "band_names": [
                "skin_temperature",
                "temperature_2m",
                "total_precipitation_sum",
                "u_component_of_wind_10m",
                "v_component_of_wind_10m",
            ],
            "apply_clouds": [False, False, False, False, False],
            "channel_magnitudes": [0.0, 0.0, 0.0, 0.0, 0.0],
            "scaling_factors": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
    },
}

FULL_CONFIG={'min_lvl': [0.5,0.9],
             'max_lvl': 1.0,
             'const_scale':True,
             'decay_factor':1.0,
             'clear_threshold':0.0,
             'locality_degree':1,
             'cloud_color':[True,False],
             'channel_offset':2,
             'blur_scaling':2
            }


def generate_clouds(band_stack, band_weights, scaling_factors, cloud_type="random", cloud_prob=0.0, shadow_prob=0.0):
    """Function to generate clouds. Input image should be in shape [B,C,H,W]. Band weights should be in shape [B,C,1,1]."""

    # the generator function takes reflectance values, but some inputs are in DN format.
    # we handle this by temporarily scaling to reflectance values
    band_stack *= scaling_factors.view(1, -1, 1, 1)
    print(f"Using cloud type: {cloud_type}")

    if cloud_type == "random":
        cfgs = [scg.WIDE_CONFIG, scg.BIG_CONFIG, scg.LOCAL_CONFIG, scg.FOG_CONFIG]
    elif cloud_type == "big":
        cfgs = [scg.BIG_CONFIG]
    elif cloud_type == "wide":
        cfgs = [scg.WIDE_CONFIG]
    elif cloud_type == "full":
        cfgs = [FULL_CONFIG]

    gens = []

    for cfg in cfgs:
        gens.append(scg.CloudGenerator(cfg, cloud_p=cloud_prob, shadow_p=shadow_prob))

    gen = random.choice(gens)
    out, cloud_mask, _ = gen(band_stack, channel_magnitude=band_weights, return_cloud=True)

    # cloud generations brightens the images, we clamp to get physically consistent outputs
    out = torch.clamp(out, 0.0, 1.0)

    # scale back to not disturb the input distribution of the model
    out = out / scaling_factors.view(1, -1, 1, 1)

    return out, cloud_mask


class CloudGeneratorMetaDataset(LandsatEvalDataset):
    def __init__(
        self,
        augmentation,
        data_config={},
        split="train",
        h5pys_only=False,
        eval_config=None,
        exclude_prediction_date=False,
        exclude_prediction_high_res=False,
        exclude_prediction_sensors=False,
        exclude_prediction_era5=False,
    ):
        super().__init__(
            data_config=data_config,
            split=split,
            h5pys_only=h5pys_only,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            augmentation=augmentation,
        )
        self.eval_config = eval_config
        assert self.eval_config is not None, "eval_config must be provided for cloud generation"
        assert "cloud_generation" in self.eval_config, "cloud_generation config missing"

    def _apply_cloud_augmentation(
        self, space_time_high_res_x, space_time_med_res_x, space_time_low_res_x, time_x
    ):
        # Create copies of the arrays to later compute valid masks on, since invalid data values will be
        # changed by cloud generation
        space_time_high_res_x_no_clouds_added = space_time_high_res_x.copy()
        space_time_med_res_x_no_clouds_added = space_time_med_res_x.copy()
        space_time_low_res_x_no_clouds_added = space_time_low_res_x.copy()
        time_x_no_clouds_added = time_x.copy()

        cloud_mask_s_t_h = np.zeros_like(space_time_high_res_x)
        cloud_mask_s_t_m = np.zeros_like(space_time_med_res_x)
        cloud_mask_s_t_l = np.zeros_like(space_time_low_res_x)
        cloud_mask_t = np.zeros_like(time_x)

        space_time_names = ["s_t_h_x", "s_t_m_x", "s_t_l_x", "t_x"]
        space_time_vars = [
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            time_x,
        ]
        space_time_cloud_masks = [
            cloud_mask_s_t_h,
            cloud_mask_s_t_m,
            cloud_mask_s_t_l,
            cloud_mask_t,
        ]

        channel_meta = []
        band_weights = []
        scaling_factors = []

        for name, var, cl_mask in zip(space_time_names, space_time_vars, space_time_cloud_masks):
            config = CHANNEL_WISE_CLOUD_PARAMETERS[name]
            apply_mask = []

            for sensor_cfg in config.values():
                for apply, magnitude, scaling in zip(
                    sensor_cfg["apply_clouds"],
                    sensor_cfg["channel_magnitudes"],
                    sensor_cfg["scaling_factors"],
                ):
                    apply_mask.append(apply)
                    if apply:
                        band_weights.append(magnitude)
                        scaling_factors.append(scaling)

            apply_mask = np.array(apply_mask, dtype=bool)
            channel_meta.append((var, apply_mask, cl_mask))

        if len(band_weights) == 0:
            return (
                space_time_high_res_x,
                space_time_med_res_x,
                space_time_low_res_x,
                time_x,
                space_time_high_res_x_no_clouds_added,
                space_time_med_res_x_no_clouds_added,
                space_time_low_res_x_no_clouds_added,
                time_x_no_clouds_added,
            )

        band_weights_tensor = torch.tensor(band_weights).float()
        scaling_factors_tensor = torch.tensor(scaling_factors).float()

        def apply_clouds_at_timestep(t_idx, cloud_prob, cloud_type):
            to_cloud = []

            for var, apply_mask, _ in channel_meta:
                x_slice = var[:, :, t_idx, :]
                x_cloud_channels = x_slice[:, :, apply_mask]
                to_cloud.append(x_cloud_channels)

            to_cloud_combined = np.concatenate(to_cloud, axis=-1)
            x_cloud_in = np.transpose(to_cloud_combined, (2, 0, 1))[None]

            x_cloud_in_tensor = torch.from_numpy(x_cloud_in).float()

            x_clouded, cloud_mask = generate_clouds(
                band_stack=x_cloud_in_tensor,
                band_weights=band_weights_tensor,
                scaling_factors=scaling_factors_tensor,
                cloud_type=cloud_type,
                cloud_prob=cloud_prob,
                shadow_prob=self.eval_config["cloud_generation"]["shadow_prob"],
            )

            x_clouded_hw_c = np.transpose(x_clouded[0], (1, 2, 0))
            cloud_mask_hw_c = np.transpose(cloud_mask[0], (1, 2, 0))

            # redistribute channels
            c_start = 0
            for var, apply_mask, cl_mask in channel_meta:
                c_count = apply_mask.sum()
                c_end = c_start + c_count

                if c_count > 0:
                    var[:, :, t_idx, apply_mask] = x_clouded_hw_c[:, :, c_start:c_end]
                    cl_mask[:, :, t_idx, apply_mask] = cloud_mask_hw_c[:, :, c_start:c_end]

                c_start = c_end

        cloud_type = self.eval_config.get("cloud_generation", {}).get("cloud_type", "random")

        if self.eval_config["cloud_generation"]["cloud_prob_pred_day"] > 0.0:
            apply_clouds_at_timestep(
                -1,
                self.eval_config["cloud_generation"]["cloud_prob_pred_day"],
                cloud_type
            )

        if self.eval_config["cloud_generation"]["cloud_prob_timeseries"] > 0.0:
            prob = self.eval_config["cloud_generation"]["cloud_prob_timeseries"]
            for T in range(time_x.shape[-2]):
                apply_clouds_at_timestep(T, prob, cloud_type)

        return (
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            time_x,
            space_time_high_res_x_no_clouds_added,
            space_time_med_res_x_no_clouds_added,
            space_time_low_res_x_no_clouds_added,
            time_x_no_clouds_added,
            cloud_mask_s_t_l,
        )

    def _tif_to_array(self, tif_path: Path) -> DatasetOutput:
        """
        Loads a spatiotemporal tif file, divides it into different array groups, and creates valid data masks.

        The different array types are:
        space_time_high_res_x: (H, W, T, C_STH)
        space_time_med_res_x: (3, 3, T, C_STM)
        space_time_low_res_x: (2, 2, T, C_STL)
        space_x: (H, W, C_SP)
        time_x: (T, C_T)
        static_x: (C_ST)

        space_time_med_res_x and space_time_low_res_x are created by taking the block mean of their high res version.
        valid data masks are created by masking out values below a channel-specific threshold (0: invalid, 1: valid).
        """
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)

            # extract lat, lon in EPSG:4326 from tif_path
            # TODO: make this dynamic in case the tif_path has a different naming convention
            parts = tif_path.stem.split("_")

            if self.split == "inference":
                lat = float(parts[2])
                lon = float(parts[3])

            else:
                lat = float(parts[3])
                lon = float(parts[4])

        num_timesteps = (values.shape[0] - len(EE_SPACE_BANDS)) / len(EO_ALL_DYNAMIC_IN_TIME_BANDS)
        assert (values.shape[0] - len(EE_SPACE_BANDS)) % len(EO_ALL_DYNAMIC_IN_TIME_BANDS) == 0, (
            f"{tif_path} has incorrect number of channels"
        )
        assert num_timesteps == NUM_TIMESTEPS, f"{tif_path} has incorrect number of timesteps"
        dynamic_in_time_x = rearrange(
            values[: -(len(EE_SPACE_BANDS))],
            "(t c) h w -> h w t c",
            c=len(EO_ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = self._check_and_fillna(
            dynamic_in_time_x, EO_ALL_DYNAMIC_IN_TIME_BANDS_NP
        )
        space_time_high_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            : -(
                len(SPACE_TIME_MED_RES_BANDS)
                + len(EO_SPACE_TIME_LOW_RES_BANDS)
                + len(TIME_BANDS)
                + len(CLOUD_BANDS)
            ),
        ]
        space_time_med_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(
                len(SPACE_TIME_MED_RES_BANDS)
                + len(EO_SPACE_TIME_LOW_RES_BANDS)
                + len(TIME_BANDS)
                + len(CLOUD_BANDS)
            ) : -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)),
        ]
        space_time_low_res_x = dynamic_in_time_x[
            :,
            :,
            :,
            -(len(EO_SPACE_TIME_LOW_RES_BANDS) + len(TIME_BANDS) + len(CLOUD_BANDS)) : -(
                len(TIME_BANDS) + len(CLOUD_BANDS)
            ),
        ]
        time_x = dynamic_in_time_x[
            :, :, :, -(len(TIME_BANDS) + len(CLOUD_BANDS)) : -len(CLOUD_BANDS)
        ]

        (
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            time_x,
            space_time_high_res_x_no_clouds_added,
            space_time_med_res_x_no_clouds_added,
            space_time_low_res_x_no_clouds_added,
            time_x_no_clouds_added,
            cloud_mask_s_t_l,
        ) = self._apply_cloud_augmentation(
            space_time_high_res_x, space_time_med_res_x, space_time_low_res_x, time_x
        )

        time_x = np.nanmean(time_x, axis=(0, 1))
        time_x_no_clouds_added = np.nanmean(time_x_no_clouds_added, axis=(0, 1))

        # NDSI = (Green - SWIR) / (Green + SWIR)
        if MODALITIES["ndsi"].get("active"):
            # base on array without clouds added, because no data values will be changed
            # cloud dependent computation will be taken care of with cloud mask
            ndsi = self.calculate_ndi(
                space_time_low_res_x_no_clouds_added,
                band_1="sur_refl_b04",
                band_2="sur_refl_b06",
                cloud_mask=cloud_mask_s_t_l,
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndsi), axis=-1)
            space_time_low_res_x_no_clouds_added = np.concatenate(
                (space_time_low_res_x_no_clouds_added, ndsi), axis=-1
            )
            assert (ndsi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDSI for {tif_path}"
            )
            assert (
                (ndsi >= NDI_VALID_DATA_BOUNDS[0]) & (ndsi <= NDI_VALID_DATA_BOUNDS[1])
                | (ndsi == NO_DATA_VALUE)
            ).all(), f"NDI values out of bounds {NDI_VALID_DATA_BOUNDS} for {tif_path}"

        # NDVI = (NIR - Red) / (NIR + Red)
        if MODALITIES["ndvi"].get("active"):
            ndvi = self.calculate_ndi(
                space_time_low_res_x_no_clouds_added,
                band_1="sur_refl_b02",
                band_2="sur_refl_b01",
                cloud_mask=cloud_mask_s_t_l,
            )
            space_time_low_res_x = np.concatenate((space_time_low_res_x, ndvi), axis=-1)
            space_time_low_res_x_no_clouds_added = np.concatenate(
                (space_time_low_res_x_no_clouds_added, ndvi), axis=-1
            )
            assert (ndvi != MODIS_FILL_VALUE).any(), (
                f"MODIS fill values encountered in NDVI for {tif_path}"
            )
            assert (
                (ndvi >= NDI_VALID_DATA_BOUNDS[0]) & (ndvi <= NDI_VALID_DATA_BOUNDS[1])
                | (ndvi == NO_DATA_VALUE)
            ).all(), f"NDI values out of bounds {NDI_VALID_DATA_BOUNDS} for {tif_path}"

        space_x = rearrange(
            values[-len(EE_SPACE_BANDS) :],
            "c h w -> h w c",
        )
        space_x = self._check_and_fillna(space_x, np.array(EE_SPACE_BANDS))

        # one-hot encode ESA Worldcover band
        esa_wc = self.one_hot_encode_esa_worldcover(space_x[:, :, ESA_WORLDCOVER_BAND_INDEX])
        assert np.isin(esa_wc, [0, 1, NO_DATA_VALUE]).all(), (
            f"Unexpected values in ESA Worldcover for {tif_path}"
        )
        space_x = np.concatenate((space_x[:, :, : (-len(EE_WC_BANDS))], esa_wc), axis=-1)

        static_x = to_cartesian(lat, lon)
        static_x = self._check_and_fillna(static_x, np.array(STATIC_BANDS))

        months = self.month_array_from_file(tif_path, int(num_timesteps))

        (
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
            months,
        ) = self.subset_image(
            space_time_high_res_x,
            space_time_med_res_x,
            space_time_low_res_x,
            space_x,
            time_x,
            static_x,
            months,
            size=DATASET_OUTPUT_HW_HIGH_RES,
            num_timesteps=NUM_TIMESTEPS,
        )
        # base on array without clouds added, because cloud generation modified original no data values
        (
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = self.create_valid_mask(
            space_time_high_res_x_no_clouds_added,
            space_time_med_res_x_no_clouds_added,
            space_time_low_res_x_no_clouds_added,
            space_x,
            time_x_no_clouds_added,
            static_x,
        )

        # for downsampling, the arrays need to be in divisible shape so we do it after cropping
        space_time_med_res_x, valid_data_mask_s_t_m = self.downsample_dynamic_in_time_with_mean(
            space_time_med_res_x,
            valid_data_mask_s_t_m,
            target_shape=(NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM),
        )
        space_time_low_res_x, valid_data_mask_s_t_l = self.downsample_dynamic_in_time_with_mean(
            space_time_low_res_x,
            valid_data_mask_s_t_l,
            target_shape=(NUM_LOW_RES_PIXELS_PER_DIM, NUM_LOW_RES_PIXELS_PER_DIM),
        )

        try:
            assert not np.isnan(space_time_high_res_x).any(), f"NaNs in s_t_h_x for {tif_path}"
            assert not np.isnan(space_time_med_res_x).any(), f"NaNs in s_t_m_x for {tif_path}"
            assert not np.isnan(space_time_low_res_x).any(), f"NaNs in s_t_l_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_high_res_x).any(), f"Infs in s_t_h_x for {tif_path}"
            assert not np.isinf(space_time_med_res_x).any(), f"Infs in s_t_m_x for {tif_path}"
            assert not np.isinf(space_time_low_res_x).any(), f"Infs in s_t_l_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_high_res_x.astype(np.half),
                space_time_med_res_x.astype(np.half),
                space_time_low_res_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
                valid_data_mask_s_t_h,
                valid_data_mask_s_t_m,
                valid_data_mask_s_t_l,
                valid_data_mask_sp,
                valid_data_mask_t,
                valid_data_mask_st,
            )
        except AssertionError as e:
            raise e

    @staticmethod
    def calculate_ndi(
        input_array: np.ndarray, band_1: str, band_2: str, cloud_mask: np.ndarray | None = None
    ) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """


        for b in [band_1, band_2]:
            assert b in SPACE_TIME_LOW_RES_BANDS

        band_1_idx = SPACE_TIME_LOW_RES_BANDS.index(band_1)
        band_2_idx = SPACE_TIME_LOW_RES_BANDS.index(band_2)

        band_1_np = input_array[:, :, :, band_1_idx].copy()
        band_2_np = input_array[:, :, :, band_2_idx].copy()

        if cloud_mask is not None and not np.all(cloud_mask == 0):
            cloud_pixels = cloud_mask[:, :, :, band_1_idx] > 0
            band_1_np[cloud_pixels] = NO_DATA_VALUE
            band_2_np[cloud_pixels] = NO_DATA_VALUE

        invalid = (
            (band_1_np == NO_DATA_VALUE)
            | (band_1_np == MODIS_FILL_VALUE)
            | (band_2_np == NO_DATA_VALUE)
            | (band_2_np == MODIS_FILL_VALUE)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            # suppress the following warning
            # RuntimeWarning: invalid value encountered in divide
            # for cases where near_infrared + red == 0
            # since this is handled in the where condition
            ndi = np.expand_dims(
                np.where(
                    ((band_1_np + band_2_np) > 0) & (~invalid),
                    (band_1_np - band_2_np) / (band_1_np + band_2_np),
                    NO_DATA_VALUE,
                ),
                -1,
            )
        # when the input bands have different signs, NDI can be outside [-1, 1]
        # set values outside valid range to NO_DATA_VALUE (will be masked out later)
        ndi[(ndi < NDI_VALID_DATA_BOUNDS[0]) | (ndi > NDI_VALID_DATA_BOUNDS[1])] = NO_DATA_VALUE
        return ndi


class CloudGeneratorEval(LandsatEval):
    regression = True
    spatial_token_prediction = True
    multilabel = False

    def __init__(
        self,
        exclude_prediction_date: bool = False,
        exclude_prediction_high_res: bool = False,
        exclude_prediction_sensors: bool = False,
        exclude_prediction_era5: bool = False,
        h5pys_only: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = {},
        job_id="",
        seed=DEFAULT_SEED,
    ):
        super().__init__(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            h5pys_only=h5pys_only,
            num_finetune_epochs=num_finetune_epochs,
            decoder_mode=decoder_mode,
            eval_config=eval_config,
        )

    def _get_dataset(
        self,
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        exclude_prediction_sensors: bool,
        exclude_prediction_era5: bool,
        split: str,
        augmentation,
        h5pys_only: bool = False,
        data_config: Dict = {},
        normalization: Union[str, Normalizer] = "std",
    ):
        ds = CloudGeneratorMetaDataset(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            exclude_prediction_era5=exclude_prediction_era5,
            split=split,
            h5pys_only=h5pys_only,
            augmentation=augmentation,
            data_config=data_config,
            eval_config=self.eval_config,
        )

        if normalization == "std":
            normalizing_dict = ds.load_normalization_values(
                path=config_dir / NORMALIZATION_DICT_FILENAME
            )
            normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
        else:
            normalizer = Normalizer(std=False)
        ds.normalizer = normalizer

        return ds
