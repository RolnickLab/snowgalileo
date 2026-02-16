import satellite_cloud_generator as scg
from src.config import DEFAULT_SEED
from src.data import config
from src.fsc.landsat_eval import LandsatEval, LandsatEvalDataset
from src.utils import masked_output_np_to_tensor, seed_everything
from src.data.dataset import Normalizer
from src.utils import config_dir
from src.data.config import NORMALIZATION_DICT_FILENAME
from typing import Union, Dict, cast
from einops import rearrange
import numpy as np
import psutil
import random
from pathlib import Path
import rioxarray
import torch
import torch.nn.functional as F
from satellite_cloud_generator.noise import generate_perlin, flex_noise
from satellite_cloud_generator.CloudSimulator import mix
from satellite_cloud_generator.CloudSimulator import KT as KT
import xarray as xr
from src.data.config import (
    DATA_FOLDER,
    MODALITIES,
    MODIS_FILL_VALUE,
    NDI_VALID_DATA_BOUNDS,
    NO_DATA_VALUE,
    NORMALIZATION_DICT_FILENAME,
    NUM_LOW_RES_PIXELS_PER_DIM,
    NUM_MED_RES_PIXELS_PER_DIM,
    NUM_TIMESTEPS,
    RESULTS_FOLDER,
    DATASET_OUTPUT_HW_HIGH_RES,
)
from src.data.dataset import Dataset as BaseDataset
from src.data.dataset import DatasetOutput, Normalizer, to_cartesian
from src.data.earthengine.eo_eval import (
    CLOUD_BANDS,
    EE_SPACE_BANDS,
    EE_WC_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS,
    EO_ALL_DYNAMIC_IN_TIME_BANDS_NP,
    EO_SPACE_TIME_LOW_RES_BANDS,
    ESA_WORLDCOVER_BAND_INDEX,
    SPACE_BAND_GROUPS_IDX,
    SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX,
    SPACE_TIME_MED_RES_BANDS,
    SPACE_TIME_HIGH_RES_BANDS,
    SPACE_TIME_MED_RES_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BANDS,
    TIME_BANDS_GROUPS_IDX,
    SPACE_TIME_LOW_RES_BANDS,
)

seed_everything(DEFAULT_SEED)
process = psutil.Process()

EO_ALL_DYNAMIC_IN_TIME_BANDS = (
    SPACE_TIME_HIGH_RES_BANDS
    + SPACE_TIME_MED_RES_BANDS
    + EO_SPACE_TIME_LOW_RES_BANDS
    + TIME_BANDS
    + CLOUD_BANDS
)

CHANNEL_WISE_CLOUD_PARAMETERS: Dict[str, Dict] = {
    "s_t_h_x": {
        "S1": {
            "band_names": ["VV", "VH", "angle"],
            "apply_clouds": [False, False, False],
            "channel_magnitudes": [0.0, 0.0, 0.0],
        },
        "S2": {
            "band_names": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "apply_clouds": [True, True, True, True, True, True],
            "channel_magnitudes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
            "channel_magnitudes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
    "s_t_m_x": {
        "S3": {
            "band_names": ["Oa17_radiance", "Oa21_radiance"],
            "apply_clouds": [True, True],
            "channel_magnitudes": [0.0, 0.0],
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
            "channel_magnitudes": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "VIIRS": {
            "band_names": ["I1", "I3"],
            "apply_clouds": [True, True],
            "channel_magnitudes": [0.0, 0.0],
        },
    },
    "t_x": {
        "VIIRS": {
            "band_names": ["M5", "M7", "M10", "M11"],
            "apply_clouds": [True, True, True, True],
            "channel_magnitudes": [0.0, 0.0, 0.0, 0.0],
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
        },
    },
}

class CostumCloudGenerator(scg.CloudGenerator):
    def __init__(self, config, cloud_p=1.0, shadow_p=1.0):
        super().__init__(config, cloud_p=cloud_p, shadow_p=shadow_p)

    def add_cloud(input,
                max_lvl=(0.95,1.0),
                min_lvl=(0.0, 0.05),
                channel_magnitude=None,
                clear_threshold=0.0,
                noise_type = 'perlin',
                const_scale=True,
                decay_factor=1,
                locality_degree=1,
                invert=False,
                channel_magnitude_shift=0.05,
                channel_offset=2,
                blur_scaling=2.0,
                cloud_color=True,
                return_cloud=False
                ):
        """ Takes an input image of shape [batch, channels, height, width]        
            and returns a generated cloudy version of the input image
        
        Args:
            input (Tensor) : input image in shape [B,C,H,W]
        
            max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
            
            min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
            channel_magnitude (Tensor) : cloud magnitudes in each channel, shape [B,C,1,1]
            
            clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
            
            noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
            
            const_scale (bool): If True, the spatial frequencies of the cloud shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)
            
            decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
            
            locality degree (int): more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)
            
            invert (bool) : for some applications, the cloud can be inverted to effectively decrease the level of reflected power (see thermal example in the notebook)
            
            channel_offset (int): optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)
            
            channel_magniutde_shift (float): optional offset from the reference cloud mask magnitude for individual channels, if non-zero, then each channel will have a cloud magnitude uniformly sampled from C+-channel_magnitude, where C is the reference cloud mask
            
            blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.
            
            cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
            
            return_cloud (bool): If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image
            
        Returns:
        
            Tensor: Tensor containing a generated cloudy image (and a cloud mask if return_cloud == True)
    
        """  
        
        if not torch.is_tensor(input):
            input = torch.FloatTensor(input)
        
        while len(input.shape) < 4:
            input = input.unsqueeze(0)  
        
        b,c,h,w = input.shape
        device=input.device
        
        # --- Potential Sampling of Parameters (if provided as a range)
        min_lvl=torch.tensor(min_lvl, device=device)
        max_lvl=torch.tensor(max_lvl, device=device)
        
        if len(min_lvl.shape) != 0:
            min_lvl = min_lvl[0] +(min_lvl[1]-min_lvl[0])*torch.rand([b,1,1,1], device=device)
            
        # max_lvl is dependent on min_lvl (cannot be less than min_lvl)
        if len(max_lvl.shape) != 0:        
            max_floor=min_lvl+F.relu(max_lvl[0]-min_lvl)
            max_lvl = max_floor + (max_lvl[1]-max_floor)*torch.rand([b,1,1,1], device=device)
            
        # ensure max_lvl does not go below min_lvl
        max_lvl=min_lvl+F.relu(max_lvl-min_lvl)
            
        # clear_threshold
        if isinstance(clear_threshold, tuple) or isinstance(clear_threshold, list):
            clear_threshold = clear_threshold[0] +(clear_threshold[1]-clear_threshold[0])*torch.rand([b,1,1], device=device)
            
        # decay_factor
        if isinstance(decay_factor, tuple) or isinstance(decay_factor, list):
            decay_factor = float(decay_factor[0] +(decay_factor[1]-decay_factor[0])*torch.rand([1,1]))

        # locality_degree
        if isinstance(locality_degree, tuple) or isinstance(locality_degree, list):
            locality_degree = int(locality_degree[0]+torch.randint(1+locality_degree[1]-locality_degree[0],(1,1)))
        
        # --- End of Parameter Sampling
        locality_degree=max([1, int(locality_degree)])
        
        net_noise_shape=torch.ones((b,h,w),device=device)
        for idx in range(locality_degree):
            # generate noise shape
            if noise_type == 'perlin':
                noise_shape=generate_perlin(shape=(h,w), batch=b, device=device, const_scale=const_scale, decay_factor=decay_factor)     
            elif noise_type == 'flex':
                noise_shape = flex_noise(h,w, const_scale=const_scale, decay_factor=decay_factor)
            else:
                raise NotImplementedError

            noise_shape -= noise_shape.min()
            noise_shape /= noise_shape.max()
            
            net_noise_shape*=noise_shape
            
        # apply non-linearities and rescale
        net_noise_shape[net_noise_shape < clear_threshold] = 0.0
        net_noise_shape -= clear_threshold  
        net_noise_shape = net_noise_shape.clip(0,1)    
        if not net_noise_shape.max()==0:
            net_noise_shape /= net_noise_shape.max()

        # channel-wise mask
        cloud=(net_noise_shape.unsqueeze(1)*(max_lvl-min_lvl) + min_lvl).expand(b,c,h,w)
        
        # channel-wise thickness difference
        if channel_magnitude_shift != 0.0:
            channel_magnitude_shift=abs(channel_magnitude_shift)
            weights=channel_magnitude_shift*(2*torch.rand(c, device=device)-1)+1
            cloud=(weights[:,None,None]*cloud)
        
        # channel offset (optional)
        if channel_offset != 0:
            offsets = torch.randint(-channel_offset, channel_offset+1, (2,c))
            
            crop_val = offsets.max().abs()
            if crop_val != 0:
                for ch in range(cloud.shape[1]):
                    cloud[:,ch] = torch.roll(cloud[:,ch], offsets[0,ch].item(),dims=-2)
                    cloud[:,ch] = torch.roll(cloud[:,ch], offsets[1,ch].item(),dims=-1)                    

                    cloud = KT.resize(cloud[:,:,crop_val:-crop_val-1, crop_val:-crop_val-1],
                                    (h,w),
                                    interpolation='nearest',
                                    align_corners=True)     
        
        # transparency between 0 and 1
        cloud=cloud.clip(0,1)
        
        if channel_magnitude is None:
            channel_magnitude=torch.ones(*input.shape[:-2],1,1,device=input.device)
                    
        output = mix(input, cloud, channel_magnitude=channel_magnitude, blur_scaling=blur_scaling, cloud_color=cloud_color, invert=invert)
        
        if not return_cloud:
            return output
        else:
            return output, cloud# if not invert else 1-cloud


def generate_clouds(band_stack, band_weights, cloud_prob=0.0, shadow_prob=0.0):
    """Function to generate clouds. Input image should be in shape [B,C,H,W]. Band weights should be in shape [B,C,1,1]."""

    cfgs = [scg.WIDE_CONFIG, scg.BIG_CONFIG, scg.LOCAL_CONFIG, scg.FOG_CONFIG]

    gens = []

    for cfg in cfgs:
        gens.append(CostumCloudGenerator(cfg, cloud_p=cloud_prob, shadow_p=shadow_prob))

    gen = random.choice(gens)
    out, cloud_mask, _ = gen(band_stack, channel_magnitude=band_weights, return_cloud=True)

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
    ):
        super().__init__(
            data_config=data_config,
            split=split,
            h5pys_only=h5pys_only,
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            augmentation=augmentation,
        )
        self.eval_config = eval_config
        assert self.eval_config is not None, "eval_config must be provided for cloud generation"
        assert "cloud_generation" in self.eval_config, "cloud_generation config missing"

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
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
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

        # ---------- Cloud Generation ------------------------------------

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

        # TODO:
        # Test NDSI / NDVI
        # Cloud mask: 0 is no cloud, but why are the others <1 ?
        # Test no data values
        # Test output visually
        if self.eval_config["cloud_generation"]["cloud_prob_pred_day"] != 0.0:
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
            to_cloud = []
            channel_slices = []
            band_weights = []

            for name, var, cl_mask in zip(
                space_time_names, space_time_vars, space_time_cloud_masks
            ):
                config = CHANNEL_WISE_CLOUD_PARAMETERS[name]
                apply_clouds_mask = []
                for sensor_cfg in config.values():
                    for apply, magnitude in zip(
                        sensor_cfg["apply_clouds"], sensor_cfg["channel_magnitudes"]
                    ):
                        apply_clouds_mask.append(apply)
                        if apply:
                            band_weights.append(magnitude)

                apply_clouds_mask = np.array(apply_clouds_mask, dtype=bool)
                # var shape: [H, W, T, C]
                x_last = var[:, :, -1, :]  # [H, W, C]
                x_cloud_channels = x_last[:, :, apply_clouds_mask]  # [H, W, C_cloud]
                to_cloud.append(x_cloud_channels)

                # Save where these channels came from
                channel_slices.append((var, apply_clouds_mask, cl_mask))

        to_cloud_combined = np.concatenate(to_cloud, axis=-1)
        x_cloud_in = np.transpose(to_cloud_combined, (2, 0, 1))[None]

        # to tensor
        x_cloud_in_tensor = torch.from_numpy(x_cloud_in).float()
        band_weights_tensor = torch.tensor(band_weights).float()

        x_clouded, cloud_mask = generate_clouds(
            band_stack=x_cloud_in_tensor,
            band_weights=band_weights_tensor,
            cloud_prob=self.eval_config["cloud_generation"]["cloud_prob_pred_day"],
            shadow_prob=self.eval_config["cloud_generation"]["shadow_prob"],
        )
        x_clouded_hw_c = np.transpose(x_clouded[0], (1, 2, 0))
        cloud_mask_hw_c = np.transpose(cloud_mask[0], (1, 2, 0))

        c_start = 0
        for array, clouds_applied, cl_mask in channel_slices:
            c_count = clouds_applied.sum().item()
            c_end = c_start + c_count

            cloud_chunk = x_clouded_hw_c[:, :, c_start:c_end]
            array[:, :, -1, clouds_applied] = cloud_chunk
            cl_mask[:, :, -1, clouds_applied] = cloud_mask_hw_c[:, :, c_start:c_end]

            c_start = c_end

        # ----------------------------------------------------------------------

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
        assert esa_wc.all() in [0, 1, NO_DATA_VALUE], (
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
        input_array: np.ndarray, band_1: str, band_2: str, cloud_mask: np.ndarray
    ) -> np.ndarray:
        r"""
        Given an input array of shape [h, w, t, bands]
        where bands == len(EO_DYNAMIC_IN_TIME_BANDS_NP), returns an array of shape
        [h, w, t, 1] representing NDI,
        (band_1 - band_2) / (band_1 + band_2)
        """

        for b in [band_1, band_2]:
            assert b in SPACE_TIME_LOW_RES_BANDS

        band_1_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_1)]
        band_2_np = input_array[:, :, :, SPACE_TIME_LOW_RES_BANDS.index(band_2)]

        if not np.all(cloud_mask == 0):
            band_1_np[:, :, -1] = NO_DATA_VALUE
            band_2_np[:, :, -1] = NO_DATA_VALUE

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
        h5pys_only: bool = False,
        num_finetune_epochs: int = 50,
        decoder_mode: str = "attention_probe",
        eval_config: Dict = {},
    ):
        super().__init__(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
            h5pys_only=h5pys_only,
            num_finetune_epochs=num_finetune_epochs,
            decoder_mode=decoder_mode,
            eval_config=eval_config,
        )

    def _get_dataset(
        self,
        augmentation,
        exclude_prediction_date: bool,
        exclude_prediction_high_res: bool,
        exclude_prediction_sensors: bool,
        split: str,
        h5pys_only: bool = False,
        data_config: Dict = {},
        normalization: Union[str, Normalizer] = "std",
    ) -> CloudGeneratorMetaDataset:
        ds = CloudGeneratorMetaDataset(
            exclude_prediction_date=exclude_prediction_date,
            exclude_prediction_high_res=exclude_prediction_high_res,
            exclude_prediction_sensors=exclude_prediction_sensors,
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
