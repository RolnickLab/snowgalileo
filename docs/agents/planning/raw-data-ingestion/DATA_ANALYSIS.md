# Data Ingestion Analysis

This repository ingests Earth observation inputs from Google Earth Engine in
`src/data/earthengine`. The Earth Engine layer exports one multiband GeoTIFF per
area and time window; `src/data/dataset.py` then reshapes that GeoTIFF into the
model's tensor groups, adds derived channels, creates masks, and applies
normalization.

## End-to-end flow

1. Export windows are sampled around latitude/longitude points.
   `EarthEngineExporter.export_for_latlons` constructs a square polygon around
   each point. The polygon side length is `EXPORTED_HEIGHT_WIDTH_METRES = 1000 m`
   (built from `surrounding_metres = EXPORTED_HEIGHT_WIDTH_METRES / 2 = 500 m`
   on each side of the centre). For every point, the exporter iterates the three
   hemisphere-specific seasons defined in `NORTH_HEM_SEASONS` /
   `SOUTH_HEM_SEASONS` (`early`, `mid`, `late`). For each season it samples a
   random year in `[START_YEAR, END_YEAR]` via `sample_season_year`, then samples
   a contiguous `NUM_TIMESTEPS = 8` day window inside that season via
   `sample_time_window`. One GeoTIFF is exported per (point, season).
2. `create_ee_image` iterates over the sampled window using
   `DAYS_PER_TIMESTEP = 1`. For each day it calls every active time-varying
   image function in `TIME_IMAGE_FUNCTIONS` (in the fixed order asserted in
   `eo.py`), clips the result to the polygon, concatenates the bands, then
   reduces all timesteps into one multiband image with `imcoll.iterate`.
3. Static spatial layers from `SPACE_IMAGE_FUNCTIONS` (DEM + ESA WorldCover) are
   appended once after the time-varying stack.
4. Export uses `crs="EPSG:4326"`, `scale=10`,
   `formatOptions={"noData": NO_DATA_VALUE}` with `NO_DATA_VALUE = -9999`, and
   <!-- NOTE 2026-06-04: this `crs="EPSG:4326"` is `create_ee_image`/label-path's
   DEFAULT. The Bow Valley **inference** path (`export_from_csv_utm`) overrides it
   with the CSV's `crs=EPSG:32611`, so the reference patches and our per-cell grid
   are UTM 11N @ 10 m (100×100), NOT 4326. See PLAN §3 Grid+CRS table / KNOWLEDGE.md. -->

   `img.unmask(-9999)` before export. All sources are resampled by Earth Engine
   onto the 10 m export grid, even when their native resolution is coarser, so
   the GeoTIFF is approximately `100 x 100` pixels per band per timestep.
5. The dataset loader reads the GeoTIFF, reshapes the dynamic bands from
   `(timestep * channel, height, width)` to `(height, width, timestep, channel)`,
   crops to `DATASET_OUTPUT_HW_HIGH_RES = 100` pixels via `subset_image`, splits
   channels into model groups, computes derived low-resolution indices
   (`NDSI`, `NDVI`), one-hot encodes WorldCover, creates valid-data masks,
   block-mean downsamples medium and low resolution groups to
   `NUM_MED_RES_PIXELS_PER_DIM = 5` and `NUM_LOW_RES_PIXELS_PER_DIM = 2`
   respectively, spatially averages the time-only group, and optionally
   normalizes valid pixels.

### Key constants (`src/data/config.py`)

| Constant | Value | Purpose |
| --- | --- | --- |
| `DAYS_PER_TIMESTEP` | `1` | Day stride between consecutive timesteps. |
| `NUM_TIMESTEPS` | `8` | Number of days in each exported window. |
| `EXPORTED_HEIGHT_WIDTH_METRES` | `1000` | Side length of the export polygon. |
| `DATASET_OUTPUT_HW_HIGH_RES` | `100` | High-res H/W after cropping. |
| `DATASET_OUTPUT_HW_MED_RES` | `200` | Effective metres per pixel for med-res; yields `5 x 5` after downsampling. |
| `DATASET_OUTPUT_HW_LOW_RES` | `500` | Effective metres per pixel for low-res; yields `2 x 2` after downsampling. |
| `NO_DATA_VALUE` | `-9999` | Sentinel value for missing pixels and acquisitions. |
| `MODIS_FILL_VALUE` | `-28672` | Native MODIS fill value, additionally checked by derived-index code. |
| `NDI_VALID_DATA_BOUNDS` | `(-1, 1)` | Valid range used to clamp NDSI/NDVI. |
| `START_YEAR`, `END_YEAR` | `2022`, `2023` | Inclusive sampling range; constrained by Landsat 9 availability. |

Current exported dynamic order per timestep is:

```text
S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5
+ MODIS cloud flag + S2 cloud flag + Landsat cloud flag
```

Static Earth Engine order is:

```text
Copernicus DEM: DEM, slope, aspect
ESA WorldCover: Map
```

## Model tensor groups

| Group | Sources (band count) | Loader shape |
| --- | --- | --- |
| `space_time_high_res_x` | S1 `[VV, VH, angle]` (3) + S2 `[B2, B3, B4, B8, B11, B12]` (6) + Landsat `[B2..B7]_landsat` (6) | `100 x 100 x 8 x 15` |
| `space_time_med_res_x` | S3 `[Oa17_radiance, Oa21_radiance]` (2) | Downsampled to `5 x 5 x 8 x 2` |
| `space_time_low_res_x` | MODIS `sur_refl_b01..b07` (7) + VIIRS fine `[I1, I3]` (2) + derived `NDSI` (1) + derived `NDVI` (1) | Downsampled to `2 x 2 x 8 x 11` |
| `time_x` | VIIRS coarse `[M5, M7, M10, M11]` (4) + ERA5 `[skin_temperature, temperature_2m, total_precipitation_sum, u_component_of_wind_10m, v_component_of_wind_10m]` (5) | Spatial mean, `8 x 9` |
| `space_x` | DEM `[DEM, slope, aspect]` (3) + one-hot WorldCover (11) | `100 x 100 x 14` |
| `static_x` | Location only, not exported from Earth Engine | Cartesian `[x, y, z]` from filename lat/lon |

Cloud flag bands are exported and included in the dynamic channel count, but the
main dataset loader drops them when returning model tensors. They are not parsed
into cloud masks in `Dataset._tif_to_array`.

## Assembly for Training and Evaluation

The raw Earth Engine stack is not passed directly to the model. Both the
pretraining and downstream fractional snow cover pipelines consume the processed
`DatasetOutput` groups created from the GeoTIFF or HDF5 cache.

### Shared Dataset Output Contract

After `Dataset._tif_to_array` or `LandsatEvalDataset._tif_to_array`, every
sample is represented as:

| Field | Contents | Used as |
| --- | --- | --- |
| `space_time_high_res_x` | Sentinel-1, Sentinel-2, Landsat | High-resolution spatiotemporal tokens |
| `space_time_med_res_x` | Sentinel-3 | Medium-resolution spatiotemporal tokens |
| `space_time_low_res_x` | MODIS, VIIRS fine, NDSI, NDVI | Low-resolution spatiotemporal tokens |
| `space_x` | DEM, slope, aspect, one-hot WorldCover | Static spatial tokens |
| `time_x` | VIIRS coarse, ERA5 | Time-only tokens |
| `static_x` | Cartesian location `x, y, z` | Static non-spatial tokens |
| `months` | Month index from filename | Temporal positional/context input |
| `valid_data_mask_*` | Per-channel valid-data masks | Converted into model masks |

The encoder does not operate on individual raw bands directly. The pipeline
aggregates bands into channel groups defined by the `*_GROUPS_IDX` ordered
dicts in `src/data/earthengine/eo.py`. Masks are applied at the group level —
a group is masked if any source band inside that group is invalid.

The groups, in the order they are concatenated along the channel axis:

- High-resolution (7 groups, `SPACE_TIME_HIGH_RES_BANDS_GROUPS_IDX`):
  `S1` `[VV, VH, angle]`, `S2_RGB` `[B2, B3, B4]`, `S2_NIR` `[B8]`,
  `S2_SWIR` `[B11, B12]`, `L_RGB` `[B2_landsat, B3_landsat, B4_landsat]`,
  `L_NIR` `[B5_landsat]`, `L_SWIR` `[B6_landsat, B7_landsat]`.
- Medium-resolution (1 group, `SPACE_TIME_MED_RES_BANDS_GROUPS_IDX`):
  `S3_NIR` `[Oa17_radiance, Oa21_radiance]`.
- Low-resolution (up to 7 groups, `SPACE_TIME_LOW_RES_BANDS_GROUPS_IDX`):
  `MODIS_RGB` `[sur_refl_b01, sur_refl_b03, sur_refl_b04]`,
  `MODIS_NIR` `[sur_refl_b02]`,
  `MODIS_SWIR` `[sur_refl_b05, sur_refl_b06, sur_refl_b07]`,
  `VIIRS_RGB_FINE` `[I1]`, `VIIRS_VNIR_FINE` `[I3]`, `NDSI`, `NDVI`. The last
  two are appended only when `MODALITIES["ndsi"]` and `MODALITIES["ndvi"]` are
  active (both are active by default).
- Time-only (4 groups, `TIME_BANDS_GROUPS_IDX`):
  `VIIRS_RGB_COARSE` `[M5, M7]`, `VIIRS_VNIR_COARSE` `[M10]`,
  `VIIRS_SWIR_COARSE` `[M11]`, `ERA5` (all five meteorological bands).
- Space-only (2 groups, `SPACE_BAND_GROUPS_IDX`): `DEM` (elevation, slope,
  aspect), `WC` (11 one-hot WorldCover channels).
- Static (1 group, `STATIC_BAND_GROUPS_IDX`): `location` `[x, y, z]`.

### Pretraining Pipeline

Entry points:

- Export: `scripts/export_for_pretrain.py`.
- Training: `scripts/pretrain.py`.
- Dataset: `src.data.Dataset`.
- Collate and masking: `mae_collate_fn` in `src/collate_fns.py`.

Pretraining assembly flow:

1. `scripts/export_for_pretrain.py` reads sampling points, builds
   `EarthEngineExporter`, and exports 8-day Earth Engine windows into the
   configured TIFF folder.
2. `scripts/pretrain.py` loads `Dataset(data_folder=DATA_FOLDER / tifs_folder)`.
   It can read GeoTIFFs directly or use an HDF5 cache through `h5py_folder` and
   `h5pys_only`.
3. If config normalization is `"std"`, the script loads
   `configs/normalizing_dict_december.json`, constructs `Normalizer(std=True)`,
   and assigns it to the dataset. Each `__getitem__` normalizes only valid
   pixels and preserves `-9999` for invalid pixels.
4. The `DataLoader` uses `mae_collate_fn`. This function default-collates the
   dataset fields and calls `batch_subset_mask_galileo` four times, producing
   four independently masked views of the same batch.
5. `batch_subset_mask_galileo` optionally applies data augmentation, then calls
   `batch_mask_random`.
6. `batch_mask_random` computes token counts for every model group:
   high-resolution spatial-temporal tokens, medium-resolution tokens,
   low-resolution tokens, space-only tokens, time-only tokens, and static
   tokens.
7. Random mask values are assigned with this meaning:
   - `0`: token is visible to the encoder.
   - `1`: token is hidden and ignored by the decoder/loss.
   - `2`: token is hidden from the encoder but reconstructed by the decoder.
8. Invalid-data masks from the dataset override random masking. Any invalid
   channel group is forced to mask value `1`, so no loss is computed for missing
   or out-of-range source data.
9. The training loop sends each masked view through `Encoder`, then
   `GalileoPixelDecoder`, and computes masked reconstruction loss with
   `do_loss`.

In pretraining, all source groups are therefore assembled into a single
multi-resolution masked-autoencoder objective. The model learns to reconstruct
masked sensor, meteorological, terrain, land-cover, and location tokens from the
visible subset.

### Fractional Snow Cover Finetuning Pipeline

Entry points:

- Finetuning: `scripts/finetune.py`.
- Evaluation-only: `scripts/eval_only.py`.
- Dataset: `LandsatEvalDataset` in `src/fsc/landsat_eval.py`.
- Task wrapper: `LandsatEval` in `src/fsc/landsat_eval.py`.
- Prediction head: `EncoderWithHead` in `src/fsc/patch_predict.py`.

Finetuning assembly flow:

1. `scripts/finetune.py` loads a pretrained encoder checkpoint or creates a
   randomly initialized encoder.
2. It loads a config from `configs/finetune`. The `data` block defines the input
   Earth Engine TIFF folder, optional HDF5 folder, label folder, label
   resolution, timestep count, and split layout.
3. `LandsatEval._get_dataset` constructs `LandsatEvalDataset` for the `train`
   and `test` splits.
4. `LandsatEvalDataset` reads the same Earth Engine band stack structure as
   pretraining, but it parses dates and coordinates from evaluation-specific
   filename conventions.
5. It pairs each input image with a label TIFF of the same name. The label is a
   fractional snow cover mask loaded with `rioxarray`, squeezed to 2D, and
   returned with the input tensors.
6. The dataset builds channel-group masks deterministically instead of using the
   random masked-autoencoder collate function:
   - all valid source groups start with mask `0`;
   - invalid source groups from `valid_data_mask_*` become mask `1`;
   - optional prediction-day ablations can additionally mask selected groups.
7. Optional ablations change the last timestep input:
   - `exclude_prediction_date` masks all dynamic inputs for the prediction day;
   - `exclude_prediction_sensors` masks all observational sensor groups for the
     prediction day while keeping weather handling separate;
   - `exclude_prediction_high_res` masks high-resolution optical groups on the
     prediction day but keeps Sentinel-1;
   - `exclude_prediction_era5` masks the ERA5 group on the prediction day.
8. The dataset returns a `MaskedOutput`, the label raster, and the filename.
   The finetuning `DataLoader` uses PyTorch's default collation.
9. `EncoderWithHead` sends the grouped tensors and masks through the encoder.
   It then maps encoder tokens to fractional snow cover predictions with either
   a spatial-mean linear head or an attention probe.
10. With the default `patch_size_high_res=10` and a `100 x 100` high-resolution
    input, the model predicts a `10 x 10` fractional snow cover map. The sigmoid
    head constrains predictions to `[0, 1]`.
11. `finetune_seg` trains with mean squared error against the label map.

The downstream task therefore reuses the same source assembly as pretraining,
but converts masks from "random reconstruction targets" into "which sensor
groups are available to the predictor." The label is external to Earth Engine
input assembly and is aligned by filename.

### Evaluation-only Pipeline

`scripts/eval_only.py` loads an `EncoderWithHead` checkpoint and constructs one
of several `LandsatEval` variants:

- `LandsatEval` for normal fractional snow cover evaluation.
- `TimeseriesAblationsEval` or `SensorAblationsEval` when the eval config
  requests ablations.
- `CloudGeneratorEval` when the eval config requests synthetic cloud generation.

These variants still start from the same assembled tensors:
`space_time_high_res_x`, `space_time_med_res_x`, `space_time_low_res_x`,
`space_x`, `time_x`, `static_x`, group masks, and months. Evaluation then runs
the checkpointed `EncoderWithHead`, compares predicted fractional snow cover to
the label masks, and reports regression, binned-classification, and segmentation
metrics.

### Sklearn Downstream Mode

When `decoder_mode="sklearn"`, `LandsatEval.train_sklearn_model` uses the same
dataset and grouped masks, but does not train a neural prediction head. Instead:

1. The pretrained encoder produces token embeddings for each sample.
2. `apply_mask_and_average_tokens_per_highres_spatial_patch` aggregates encoder
   outputs per high-resolution spatial patch.
3. Fractional snow cover labels are rearranged into the same token sequence.
4. A scikit-learn regressor, such as linear regression, random forest, or KNN,
   is fit on encoder embeddings and evaluated with the same downstream metrics.

## Common compatibility transformations

- Missing acquisitions: most time-varying loaders call `create_placeholder`,
  which creates constant `-9999` bands clipped to the requested region.
- Missing pixels in exports: the exporter calls `unmask(-9999)` before export.
- NaN and infinity handling: `_check_and_fillna` replaces NaN/inf with per-band
  means where possible, or `0` for lower-dimensional data.
- Valid-data masks: `Dataset.create_valid_mask` starts from `value != -9999` and
  then applies per-channel lower-bound thresholds from
  `CHANNEL_WISE_INVALID_DATA_THRESHOLDS`.
- Normalization: `Normalizer._normalize` applies `(x - shift) / div` only where
  the valid-data mask is true; invalid pixels remain `-9999`.
- With precomputed normalization dictionaries, the hardcoded shift/div constants
  for standard numeric bands are replaced by `mean - 2 * std` and `4 * std`.
  NDSI, NDVI, ESA WorldCover, and location use fixed identity-style values.
- Spatial sizing: all exported dynamic spatial bands initially arrive on the
  high-resolution export grid. Medium and low resolution groups are block-mean
  downsampled after cropping, and their masks use the block minimum so any
  invalid source pixel invalidates the block.

## Source-by-source mapping

### Sentinel-1 GRD

- Earth Engine collection: `COPERNICUS/S1_GRD`.
- Bands: `VV`, `VH`, `angle`.
- Shape group: `space_time_high_res_x`.
- Query transformation:
  - Filters by date, region, and `instrumentMode == "IW"`.
  - Applies `mask_edge`, which masks pixels where the image value is less than
    `-30.0`.
  - Selects the three project bands and takes the first image in the filtered
    collection.
  - Emits a `-9999` placeholder if the collection is empty.
- Compatibility transformation:
  - Exported at 10 m with all other layers.
  - Valid thresholds are `VV >= -50`, `VH >= -50`, `angle >= 0`.
  - Baseline normalization constants are `shift=[25, 25, 0]` and
    `div=[25, 25, 90]`.

### Sentinel-2 Harmonized

- Earth Engine collection: `COPERNICUS/S2_HARMONIZED`.
- Bands: `B2`, `B3`, `B4`, `B8`, `B11`, `B12`.
- Cloud flag band: `QA60`, exported separately.
- Shape group: `space_time_high_res_x`.
- Query transformation:
  - Uses the harmonized collection to avoid the post-2022 processing-baseline
    digital number offset.
  - Filters by date and region.
  - Selects the snow-relevant optical bands and takes the first image.
  - Emits a `-9999` placeholder if no image exists.
- Compatibility transformation:
  - Bands are grouped as RGB (`B2`, `B3`, `B4`), NIR (`B8`), and SWIR
    (`B11`, `B12`).
  - Baseline normalization divides by `10000`.
  - Valid threshold is `>= -1` for each selected band.
  - `QA60` is exported but not used by the main dataset tensor output.

### Landsat 8/9 TOA

- Earth Engine collections:
  - `LANDSAT/LC09/C02/T1_TOA`
  - `LANDSAT/LC08/C02/T1_TOA`
- Bands: original `B2`, `B3`, `B4`, `B5`, `B6`, `B7`.
- Renamed bands: `B2_landsat`, `B3_landsat`, `B4_landsat`,
  `B5_landsat`, `B6_landsat`, `B7_landsat`.
- Cloud flag band: `QA_PIXEL`, exported separately.
- Shape group: `space_time_high_res_x`.
- Query transformation:
  - Tries Landsat 9 first for the date/region.
  - Falls back to Landsat 8 if Landsat 9 is unavailable.
  - Selects the six optical bands and renames them to avoid collisions with
    Sentinel-2 band names.
  - Emits a `-9999` placeholder using the renamed band names if no image exists.
- Compatibility transformation:
  - Baseline normalization divides by `10000`.
  - Valid threshold is `>= 0.0000001`, treating zero as invalid/no-data.
  - `QA_PIXEL` is exported but not used by the main dataset tensor output.

### Sentinel-3 OLCI

- Earth Engine collection: `COPERNICUS/S3/OLCI`.
- Bands: `Oa17_radiance`, `Oa21_radiance`.
- Shape group: `space_time_med_res_x`.
- Query transformation:
  - Filters by date and region.
  - Selects the two radiance bands and takes the first image.
  - Emits a `-9999` placeholder if no image exists.
- Compatibility transformation:
  - Exported on the 10 m grid, cropped with the high-resolution tensors, then
    block-mean downsampled to `5 x 5`.
  - Baseline normalization is identity: `shift=[0, 0]`, `div=[1, 1]`.
  - Valid threshold is `>= -1`.
  - The code has a `TODO` noting that the shift/div constants should be changed.

### MODIS Terra Surface Reflectance

- Earth Engine collection: `MODIS/061/MOD09GA`.
- Bands: `sur_refl_b01`, `sur_refl_b02`, `sur_refl_b03`, `sur_refl_b04`,
  `sur_refl_b05`, `sur_refl_b06`, `sur_refl_b07`.
- Cloud flag band: `state_1km`, exported separately.
- Shape group: `space_time_low_res_x`.
- Query transformation:
  - Filters by date and region.
  - Selects the seven surface-reflectance bands and takes the first image.
  - Emits a `-9999` placeholder if no image exists.
- Compatibility transformation:
  - Exported on the 10 m grid, cropped with the high-resolution tensors, then
    block-mean downsampled to `2 x 2`.
  - Baseline normalization uses `shift=-7950` and `div=8050` per band, so the
    implemented formula is `(x + 7950) / 8050`.
  - Valid threshold is `>= -100`, which masks the MODIS fill value.
  - `NDSI` is derived as `(sur_refl_b04 - sur_refl_b06) /
    (sur_refl_b04 + sur_refl_b06)`.
  - `NDVI` is derived as `(sur_refl_b02 - sur_refl_b01) /
    (sur_refl_b02 + sur_refl_b01)`.
  - Derived indices become `-9999` when either source band is `-9999`, equals the
    configured MODIS fill value, has a non-positive denominator, or produces a
    result outside `[-1, 1]`.
  - `state_1km` is exported but not used by the main dataset tensor output.

### VIIRS Surface Reflectance

- Earth Engine collection: `NASA/VIIRS/002/VNP09GA`.
- Fine bands: `I1`, `I3`.
- Coarse bands: `M5`, `M7`, `M10`, `M11`.
- Cloud flag band available in module: `QF1`, but it is not active in
  `MODALITIES` and is not exported by `eo.py`.
- Shape groups:
  - `I1`, `I3` go to `space_time_low_res_x`.
  - `M5`, `M7`, `M10`, `M11` go to `time_x`.
- Query transformation:
  - Fine and coarse loaders each filter by date and region, select their bands,
    and take the first image.
  - Each emits a `-9999` placeholder if no image exists.
- Compatibility transformation:
  - Fine bands are exported on the 10 m grid and then block-mean downsampled
    with the low-resolution group to `2 x 2`.
  - Coarse bands are spatially averaged over the exported patch to produce a
    per-timestep vector.
  - Baseline normalization uses `shift=-0.795` and `div=0.805`, so the
    implemented formula is `(x + 0.795) / 0.805`.
  - Valid threshold is `>= -0.01`.

### ERA5-Land Daily Aggregates

- Earth Engine collection: `ECMWF/ERA5_LAND/DAILY_AGGR`.
- Bands: `skin_temperature`, `temperature_2m`, `total_precipitation_sum`,
  `u_component_of_wind_10m`, `v_component_of_wind_10m`.
- Shape group: `time_x`.
- Query transformation:
  - Filters by date and region.
  - Selects the five meteorological bands and takes the first daily aggregate.
  - Emits a `-9999` placeholder if no image exists.
- Compatibility transformation:
  - Exported on the 10 m grid, then spatially averaged over the patch to produce
    a per-timestep vector.
  - Baseline normalization constants are:
    - temperatures: `shift=-272.15`, `div=35`
    - precipitation: `shift=0`, `div=0.03`
    - wind components: `shift=0`, `div=10000`
  - Because normalization is implemented as `(x - shift) / div`, the current
    temperature formula is `(kelvin + 272.15) / 35`, despite the code comment
    saying the intent is to shift to Celsius.
  - Valid thresholds are `temperature >= 184 K`, `precipitation >= -1`, and
    wind components `>= -53`.
  - **ERA5-Land accumulation / day-shift gotcha (`total_precipitation_sum` only).**
    `total_precipitation` is a forecast **accumulation** field (`GRIB_stepType = accum`,
    units = m), not an instantaneous value, and ERA5-Land stamps the accumulation that
    *closes* day `i` (the 00→24 h total) at **`00:00` of day `i+1`**. Therefore the daily
    precip total for day `i` is read from the `i+1` `00:00` slice — **not** the slice
    labelled `i`. Verified in `data/bow_valley_selection_raw/era5/*_totalprecip.nc`:
    `tp` has `valid_time` of length = days-in-month, each stamped `YYYY-MM-DDT00:00`,
    `GRIB_stepType=accum`, `units=m`. The instantaneous variables
    (`temperature_2m`, `skin_temperature`, `u/v_component_of_wind_10m`) are **not**
    accumulations and carry **no** day shift — they align to their own label. A naive
    label-based precip read attributes every day's rain to the wrong (previous) day, a
    silent off-by-one. (See `CLIPPING_PLAN.md`/adapter rules; SPEC FR-14/AC-20.)

### Copernicus DEM

- Earth Engine collection: `COPERNICUS/DEM/GLO30`.
- Bands created: `DEM`, `slope`, `aspect`.
- Shape group: `space_x`.
- Query transformation:
  - Filters by region and selects the first `DEM` image.
  - Computes slope with `ee.Terrain.slope`.
  - Computes aspect with `ee.Terrain.aspect`.
  - Concatenates elevation, slope, and aspect into one static image.
- Compatibility transformation:
  - Appended once after all dynamic bands.
  - Exported on the 10 m grid with the rest of the image.
  - Baseline normalization is identity for all three bands.
  - Valid thresholds are `DEM >= 0.0000001`, `slope >= 0`, and `aspect >= 0`.

### ESA WorldCover

- Earth Engine collection: `ESA/WorldCover/v200`.
- Exported band: `Map`.
- Loader bands after transformation: `WC_tree_cover`, `WC_shrubland`,
  `WC_grassland`, `WC_cropland`, `WC_built_up`,
  `WC_bare_sparse_vegetation`, `WC_snow_and_ice`,
  `WC_permanent_water_bodies`, `WC_herbaceous_wetland`, `WC_mangroves`,
  `WC_moss_and_lichen`.
- Shape group: `space_x`.
- Query transformation:
  - Filters by region, selects `Map`, and takes the first image.
  - Does not apply the requested date window because WorldCover is static in
    this pipeline.
- Compatibility transformation:
  - The loader one-hot encodes the class values
    `[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]` into 11 channels.
  - Class `0`, `-9999`, or unexpected class values become `-9999` across all
    one-hot channels.
  - Baseline normalization is identity for the resulting one-hot channels.

### Cloud Flag Bands

- MODIS: `MODIS/061/MOD09GA`, band `state_1km`.
- Sentinel-2: `COPERNICUS/S2_HARMONIZED`, band `QA60`.
- Landsat: Landsat 9/8 TOA collections, band `QA_PIXEL`.
- Query transformation:
  - Each cloud flag loader uses the same date/region selection as its source.
  - Landsat cloud flags prefer Landsat 9 and fall back to Landsat 8.
  - Missing flags become `-9999` placeholders.
- Compatibility transformation:
  - These bands are included in the exported dynamic stack for every timestep.
  - The main dataset loader excludes them from `space_time_*` and `time_x`
    tensors. No bit decoding is performed in the primary ingestion path.

### Location

- Source: filename latitude/longitude, not Earth Engine.
- Shape group: `static_x`.
- Transformation:
  - The loader extracts latitude and longitude from the GeoTIFF filename.
  - Coordinates are asserted to be EPSG:4326 ranges.
  - Latitude/longitude are converted to Cartesian unit-sphere coordinates
    `[x, y, z]`.
  - Baseline normalization is identity.

## Raw Archive Directory Formats and Structures (Direct-Source)

Based on the raw data archive under `data/bow_valley_selection_raw/`, direct-source ingestion must handle the following structures, files, and nested formats for the 9 modalities:

- **dem** (Copernicus DEM GLO-30):
  - Path: `data/bow_valley_selection_raw/dem/DEM1_SAR_DGE_30_[meta]/Copernicus_DSM_10_[tile]/`
  - Format: Nested SAFE directory. Under each tile's main directory, there is a `DEM/` subfolder containing a single GeoTIFF file (`..._DEM.tif`) representing elevation in meters above EGM2008 geoid.
- **era5** (ERA5-Land Daily Aggregates):
  - Path: `data/bow_valley_selection_raw/era5/`
  - Format: NetCDF (`.nc`) files. Monthly folders named `YYYYMM_ERA5LAND/` contain separate daily average files for wind and temperature variables: `10m_u_component_of_wind_0_daily-mean.nc`, `10m_v_component_of_wind_0_daily-mean.nc`, `2m_temperature_0_daily-mean.nc`, and `skin_temperature_0_daily-mean.nc`. Daily accumulated precipitation is stored as monthly files in the parent folder, e.g., `YYYYMM_ERA5LAND_totalprecip.nc`.
- **landsat8** / **landsat9** (Landsat Collection 2 Level 1 TOA):
  - Path: `data/bow_valley_selection_raw/landsat8/` and `data/bow_valley_selection_raw/landsat9/`
  - Format: `.tar` files or extracted directories containing individual band GeoTIFF files (`_B2.TIF` through `_B7.TIF` and `_B11.TIF`), a pixel QA band (`_QA_PIXEL.TIF`), and metadata text/JSON/XML files (`_MTL.json`, `_MTL.txt`, `_MTL.xml`) defining scaling coefficients.
- **sentinel1** (Sentinel-1 GRD):
  - Path: `data/bow_valley_selection_raw/sentinel1/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure. Inside the archive, measurements are in `.tiff` files (under `measurement/`) and metadata in `.xml` files.
- **sentinel2** (Sentinel-2 Level-1C):
  - Path: `data/bow_valley_selection_raw/sentinel2/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure. Granules contain JPEG2000 (`.jp2`) band files under `GRANULE/[granule_id]/IMG_DATA/`.
- **sentinel3** (Sentinel-3 OLCI Level-1 EFR):
  - Path: `data/bow_valley_selection_raw/sentinel3/`
  - Format: Standard `.zip` archives containing the Sentinel SAFE directory structure for OL_1_EFR products. The radiance bands (e.g. `Oa17_radiance.nc`, `Oa21_radiance.nc`) and coordinate tie-points (`geo_coordinates.nc`) are stored as separate NetCDF files.
- **modis** (MOD09GA daily surface reflectance):
  - Path: `data/bow_valley_selection_raw/modis/`
  - Format: Standard HDF4 (`.hdf`) files representing MOD09GA tiles (e.g., `h10v03`) containing sinusoidal grid subdatasets.
- **worldcover** (ESA WorldCover v200):
  - Path: `data/bow_valley_selection_raw/worldcover/ESA_WorldCover_10m_2021_v200_[tile]_Map/`
  - Format: Categorical GeoTIFF file (`..._Map.tif`) under its respective tile directory.

## Verified Raw Data Catalog (Bow Valley Selection)

Detailed inventory and parsed metadata of raw assets under `data/bow_valley_selection_raw` (target: `/archive/data/ai4snow/bow_valley_selection_raw`). Spans **~325.8 GB** across **10 datasets**.

### Summary Matrix

| Dataset | Subdirectory | File Count | Total Size | Primary Format | Coordinate System (CRS) | Spatial Resolution | Temporal Frequency |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **DEM** | `dem/` | 196 files / **9 `*_DEM.tif` tiles** | 632 MB | GeoTIFF / KML (nested SAFE) | `EPSG:4326` (WGS 84) | ~30m (1 arc-sec) | Static |
| **WorldCover** | `worldcover/` | 8 files / **4 `*_Map.tif` tiles** | 377 MB | GeoTIFF | `EPSG:4326` (WGS 84) | 10m (~8.33e-5°) | Static (2021) |
| **ERA5** | `era5/` | 15 | 4.4 MB | NetCDF-4 (`.nc`) | `EPSG:4326` (WGS 84) | 0.1° (~10km) | Daily Aggregated |
| **Landsat 8** | `landsat8/` | 19 | 24 GB | `.tar` (GeoTIFFs) | `EPSG:32612` (UTM 12N) | 30m | 16-day Revisit |
| **Landsat 9** | `landsat9/` | 30 | 36 GB | `.tar` (GeoTIFFs) | `EPSG:32612` (UTM 12N) | 30m | 16-day Revisit |
| **MODIS** | `modis/` | 93 | 12 GB | HDF4 (`.hdf`) | Custom Sinusoidal | 500m / 1km | Daily |
| **Sentinel-1** | `sentinel1/` | 32 | 53 GB | `.zip` (SAFE/TIFF) | *Swath / Sensor* | 10m | 6 to 12 days |
| **Sentinel-2** | `sentinel2/` | 116 | 75 GB | `.zip` (SAFE/JP2) | `EPSG:32611` (UTM 11N) | 10m / 20m / 60m | 5-day Revisit |
| **Sentinel-3** | `sentinel3/` | 125 | 112 GB | `.zip` (SEN3/NetCDF)| *Swath / Sensor* | ~300m | Daily |
| **VIIRS** | `viirs/` | 93 | 13 GB | HDF5 (`.h5`) | Custom Sinusoidal | 500m / 1km | Daily |

### Spatial-Temporal Characteristics

- **DEM (Digital Elevation Model):** Copernicus GLO-30 / GLO-10. Single band. Shape `(3601, 2401)` per 1°×1° tile. **9 `*_DEM.tif` tiles** (`N50–N52 × W115–W117`); each tile sits in a nested SAFE folder alongside KML/XML/PDF and auxiliary FLM/EDM/HEM/WBM rasters (~196 files total, 99 tifs — only the `*_DEM.tif` are elevation). **Verified mosaic extent `lon[-117,-114] lat[50,53]`** ⊇ AOI to `lat_max = 52.31`. `float32`.
- **WorldCover:** ESA WorldCover 10m. Categorical landcover. Shape `(36000, 36000)` **per 3°×3° tile**. **4 `*_Map.tif` tiles** (+ 4 `*_InputQuality.tif` companions = 8 tifs; clip only `*_Map.tif`). **Verified mosaic extent `lon[-120,-114] lat[48,54]`** ⊇ AOI to lat 52.31. `uint8`.
- **ERA5-Land:** Already-daily aggregates in NetCDF (`h5netcdf`/`h5py`), one slice per day. Precip lives in `YYYYMM_ERA5LAND_totalprecip.nc`: var `tp`, dims `(valid_time, latitude, longitude)` = `(days-in-month, 61, 61)`, **`GRIB_stepType=accum`, `units=m`**, `valid_time` stamped `YYYY-MM-DDT00:00`. Instantaneous vars in `YYYYMM_ERA5LAND/` (`t2m`, `skt`, `u10`/`v10`) as `*_daily-mean.nc`. Extent `[-120.0, -114.0, 48.0, 54.0]`. Archive span 2025-03 → 2025-05. **Day-shift (precip only): day `i`'s total is in the `i+1` `00:00` slice — see the ERA5-Land accumulation gotcha above.**
- **Landsat 8 & 9:** L1TP Collection 2 TOA reflectance. Scene shape `(8191, 8101)` (L8), `(8181, 8111)` (L9). Extent around UTM Zone 12N `[176080, 5607800, 420900, 5853300]`. 11 spectral bands + QA_PIXEL + QA_RADSAT. `uint16`.
- **MODIS:** MOD09GA daily surface reflectance HDF4 (tile `h10v03`). **Two co-registered sinusoidal grids per file** (verified via `gdalinfo`): a 1 km grid (`MODIS_Grid_1km_2D`, **1200×1200**, holds `state_1km` + geometry) and a 500 m grid (`MODIS_Grid_500m_2D`, **2400×2400**, holds the science bands `sur_refl_b01`–`b07`). 22 subdatasets total. `uint16`. **Clipping must index each grid at its own resolution — see `CLIPPING_PLAN.md §2.7`.**
- **Sentinel-1:** C-band GRD dual-pol (VV + VH). Swath range geometry. Scene shape `(16708, 26079)`. `uint16`.
- **Sentinel-2:** MultiSpectral Instrument Level-1C. Tile shape `(10980, 10980)` (tiles `T11UPS`, `T11UPT`, `T11UNS`, `T11UNT`). `uint16`.
- **Sentinel-3:** OLCI Level-1 EFR radiance NetCDF. Shape `(4091, 4865)` per orbit segment. 21 radiance bands. `uint16`.
- **VIIRS:** VNP09GA daily surface reflectance HDF5 (tile `h10v03`). **Two co-registered grids** (verified via `gdalinfo`): `VIIRS_Grid_1km_2D` (**1200×1200**, holds the coarse M-bands `M5/M7/M10/M11` → `time_x`) and `VIIRS_Grid_500m_2D` (**2400×2400**, holds the fine I-bands `I1/I3` → `space_time_low_res_x`). 67 datasets. `int16`/`uint16`. The I-bands carry `_FillValue = -28672` (same native fill as MODIS); preserve it through clipping for the same loader-sentinel reason. **Per-grid clipping — see `CLIPPING_PLAN.md §2.7`.**

### Strategic Processing Recommendations

1. **Common UTM Target Grid:** Project all inputs to local **UTM Zone 11N (EPSG:32611)**. Preserves metrics and aligns exactly with downstream fractional snow cover (FSC) labels in the Bow Valley catchment.
2. **8-Day Temporal compositing:** Map daily observational platforms (MODIS, VIIRS, S3, ERA5) to 8-day composited stats. Align Sentinel-2 and Landsat observations into identical temporal slots. Use **`-9999`** flag for all missing or cloud-contaminated pixel windows to activate model mask mechanics.
3. **Out-of-Core Processing (COGs/Zarr):** Convert processed, aligned multi-modal inputs to Cloud Optimized GeoTIFFs (COGs) or Zarr array. Allows streaming `100 x 100` pixel chips on-the-fly without loading massive scene-wide grids into RAM.
4. **Atmospheric Correction (Level 2):** Rescale Landsat and Sentinel-2 to Level-2 surface reflectance (L2A) to ensure spectral consistency and physical matching with MODIS and VIIRS.
5. **Spatial Indexing:** Maintain lightweight spatial indices in **GeoParquet** or file-based STAC catalogs for fast querying over the Bow Valley grid.

## Direct-source Interchangeability Requirements

If this repository stops using Google Earth Engine and downloads products from
Copernicus, USGS, NASA, ECMWF, or ESA directly, the replacement pipeline must
produce the same logical raster stack that `create_ee_image` currently exports.
Interchangeability means the downstream dataset code can read the output without
changing band order, tensor shapes, masks, normalization constants, or label
alignment.

At minimum, a direct-source pipeline must reproduce these global conventions:

- Use the same spatial extent and timestep windows as the current exporter.
- Use the same temporal selection rule: one acquisition per source per timestep.
  The current Earth Engine code uses `.first()` after date and region filtering,
  so a direct pipeline should define and test an equivalent deterministic sort,
  usually ascending acquisition start time within each daily window.
- Emit the exact dynamic band order per timestep:
  `S1 + S2 + Landsat + S3 + MODIS + VIIRS fine + VIIRS coarse + ERA5 + cloud flags`.
- Append static bands once, after all timesteps: `DEM`, `slope`, `aspect`, then
  WorldCover `Map`.
- Use `-9999` as the nodata value for missing acquisitions and missing pixels.
- Match the export grid used by the caller. The pretraining exporter uses
  `EPSG:4326` at `scale=10`; the evaluation exporter can use the label raster's
  CRS. The output transform, dimensions, CRS, and pixel alignment must be
  identical for paired image and label files.
- Match Earth Engine's resampling behavior closely enough for downstream masks
  and normalization to remain valid. Continuous sources should be compared
  numerically against current GEE exports before switching. Categorical and QA
  sources must use nearest-neighbor resampling.
- Preserve the current value domains. Several constants in this repository are
  tuned to Earth Engine-exported values rather than physically ideal units.

### Sentinel-1 GRD from Copernicus

Target GEE product: `COPERNICUS/S1_GRD`.

Direct-source requirements:

- Download Sentinel-1 Ground Range Detected products from the Copernicus archive.
- Restrict to Interferometric Wide swath (`IW`) and to the polarizations needed
  by the project, currently `VV` and `VH`.
- Apply the same preprocessing class as Earth Engine's GRD collection: orbit
  metadata application, thermal/border noise handling, radiometric calibration,
  and terrain correction to the target map grid.
- Convert calibrated backscatter to the same decibel value domain used by Earth
  Engine. The repository expects roughly `[-50, 1]` for `VV` and `VH` before
  normalization.
- Produce or preserve an incidence angle band named `angle` in degrees.
- Apply the repository's edge mask rule by invalidating pixels below `-30.0` in
  the SAR image bands.
- Reproject/crop to the export grid, fill missing pixels with `-9999`, and emit
  bands as `VV`, `VH`, `angle`.

Interchangeability risk: Sentinel-1 preprocessing choices materially change
pixel values. A direct pipeline should validate against existing GEE exports at
several sites before replacing this source.

### Sentinel-2 from Copernicus

Target GEE product: `COPERNICUS/S2_HARMONIZED`.

Direct-source requirements:

- Download Sentinel-2 Level-1C products if the goal is to match the current GEE
  collection. The current code uses harmonized top-of-atmosphere values, not
  Sentinel-2 Level-2A surface reflectance.
- Select bands `B2`, `B3`, `B4`, `B8`, `B11`, and `B12`.
- Preserve the Earth Engine harmonized digital-number convention. Sentinel-2 data in GEE (both Level-1C via `COPERNICUS/S2_HARMONIZED` and Level-2A via `COPERNICUS/S2_SR_HARMONIZED`) are harmonized to correct for the processing baseline baseline 04.00+ offset (+1000 DN). Direct Copernicus products do NOT have this harmonization. The local adapter must read the granule metadata, check the processing baseline version, and subtract 1000 from the digital numbers if the baseline is `04.00` or later to ensure a harmonized time series.
- Keep the current scaled-integer reflectance convention expected by the code:
  values are normalized later by division by `10000`.
- Resample 20 m SWIR bands (`B11`, `B12`) onto the target export grid in a way
  that matches the GEE export as closely as possible.
- Preserve or reconstruct `QA60` when exporting cloud flags. Do not decode it in
  the ingestion layer because the current main dataset path does not decode it.
- Emit `-9999` placeholders for missing daily acquisitions.

Interchangeability risk: using Level-2A surface reflectance would be a better
remote-sensing product for many analyses, but it is not interchangeable with the
current GEE `S2_HARMONIZED` values without changing normalization and model
expectations.

### Landsat 8/9 from USGS

Target GEE products: `LANDSAT/LC09/C02/T1_TOA`, then
`LANDSAT/LC08/C02/T1_TOA` as fallback.

Direct-source requirements:

- Download USGS Landsat Collection 2 Tier 1 Level-1 products for Landsat 9 and
  Landsat 8.
- Reproduce the current fallback rule: use Landsat 9 when available for the
  date/region, otherwise use Landsat 8.
- Convert raw optical digital numbers to top-of-atmosphere reflectance using the
  scene metadata coefficients and sun-angle correction. Do not substitute
  Level-2 surface reflectance unless the downstream normalization and thresholds
  are changed.
- Select original bands `B2`, `B3`, `B4`, `B5`, `B6`, `B7` and rename them to
  `B2_landsat`, `B3_landsat`, `B4_landsat`, `B5_landsat`, `B6_landsat`, and
  `B7_landsat`.
- Preserve `QA_PIXEL` as the Landsat cloud flag band when exporting cloud flags.
- Reproject/crop to the target grid, fill missing pixels with `-9999`, and use
  `0` or near-zero values consistently with the current invalid-data threshold.

Interchangeability risk: GEE's `T1_TOA` collection already applies a TOA
conversion. A direct USGS path must compare output reflectance against GEE rather
than passing raw Level-1 DNs through the existing loader.

### Sentinel-3 OLCI from Copernicus

Target GEE product: `COPERNICUS/S3/OLCI`.

Direct-source requirements:

- Download Sentinel-3 OLCI products that expose radiance bands corresponding to
  `Oa17_radiance` and `Oa21_radiance`.
- Preserve radiance units and scaling as exported by GEE. The repository applies
  identity baseline normalization for these bands, so any source-side scale
  factor will flow directly into the model.
- Use the Sentinel-3 OLCI SAFE format tie-point grids (latitude and longitude coordinate datasets stored in separate NetCDF files) to precisely georeference and reproject the radiance bands onto the target cell grid. Naive geolocation or ignoring the tie-point interpolation will cause significant pixel coordinate misalignment relative to GEE's orthorectified OLCI collection.
- Emit only `Oa17_radiance` and `Oa21_radiance` in the Sentinel-3 slot.
- Fill missing acquisitions with `-9999` and allow the dataset loader to
  downsample this group to `5 x 5` after cropping.

Interchangeability risk: the code has a TODO for Sentinel-3 normalization, so a
source switch is a good time to validate the radiance distribution explicitly.

### MODIS Terra Surface Reflectance from NASA LP DAAC

Target GEE product: `MODIS/061/MOD09GA`.

Direct-source requirements:

- Download MOD09GA Collection 6.1 daily Terra surface reflectance granules.
- Mosaic tiles when an exported patch crosses tile boundaries.
- Select `sur_refl_b01` through `sur_refl_b07` and preserve the value convention
  used by Earth Engine for these bands. The repository's baseline constants
  expect the integer-like MODIS range, not necessarily physical reflectance in
  `[0, 1]`.
- Preserve MODIS fill values so the current lower-bound thresholds and
  `MODIS_FILL_VALUE` checks continue to work.
- Preserve `state_1km` as the cloud flag bitfield when exporting cloud flags.
  Do not decode or remap the bitfield in the interchange format.
- Reproject from the MODIS sinusoidal grid to the target grid, then let the
  repository perform its existing crop and block-mean downsampling to `2 x 2`.
- Ensure the derived `NDSI` and `NDVI` computed by the dataset loader remain in
  `[-1, 1]` or `-9999`.

Interchangeability risk: applying the MODIS scale factor during download would
change the numeric domain relative to the current normalization constants.

### VIIRS Surface Reflectance from NASA LP DAAC

Target GEE product: `NASA/VIIRS/002/VNP09GA`.

Direct-source requirements:

- Download VNP09GA Version 2 daily VIIRS surface reflectance products.
- Mosaic tiles when necessary.
- Select fine-resolution bands `I1` and `I3` for the low-resolution spatial
  group.
- Select moderate-resolution bands `M5`, `M7`, `M10`, and `M11` for the
  time-only group.
- Preserve the scaled reflectance value domain expected by the current code.
  The baseline normalization uses `shift=-0.795` and `div=0.805`.
- Reproject/crop all selected bands to the target grid. The loader will
  downsample `I1` and `I3` spatially and average `M*` bands over the patch.
- If VIIRS cloud flags are later activated, preserve `QF1` as a raw bitfield;
  currently it is defined in code but not wired into `MODALITIES`.

Interchangeability risk: VIIRS has separate native resolutions for I-bands and
M-bands. The direct pipeline must keep the project split between spatial
low-resolution bands and time-only coarse bands.

### ERA5-Land from ECMWF CDS

Target GEE product: `ECMWF/ERA5_LAND/DAILY_AGGR`.

Direct-source requirements:

- Download ERA5-Land variables from the ECMWF Climate Data Store or another
  authoritative ECMWF endpoint.
- **The archive on disk is already daily-aggregated** (`YYYYMM_ERA5LAND_totalprecip.nc`
  + per-variable `*_daily-mean.nc`), so this archive's adapter does **not** re-aggregate
  hourly data — it reads one slice per day. (The CDS hourly→daily aggregation below
  describes how such daily files are *produced*, for reference / re-download only.)
  If producing daily files from hourly CDS data: daily **mean** over the UTC day
  (00:00–23:00) for `skin_temperature`, `temperature_2m`, `u_component_of_wind_10m`,
  `v_component_of_wind_10m`; for `total_precipitation` (a forecast accumulation) the
  daily total is the accumulation valid at the **end** of the day — i.e. take the
  `00:00` accumulation of the **following** day (or difference consecutive hourly
  accumulations and sum the hourly rates). Do **not** naively sum the 24 accumulation
  values (double-counts) and do **not** stop at 23:00 (drops the closing step).
- **Day-shift on read (`total_precipitation_sum` only) — load-bearing.** Because the
  accumulation closing day `i` is stamped at `00:00` of day `i+1`, the adapter must read
  precip for day `i` from the **`i+1` `00:00` slice**, equivalently `tp[index] → precip
  for day (index − 1)`. The instantaneous temp/wind variables carry **no** shift. Getting
  this wrong is a silent off-by-one that passes shape/type checks. Verified file facts:
  `tp` dims `(valid_time=days, latitude, longitude)`, `GRIB_stepType=accum`, `units=m`,
  `valid_time` stamped `YYYY-MM-DDT00:00`.
- Emit `skin_temperature`, `temperature_2m`, `total_precipitation_sum`,
  `u_component_of_wind_10m`, and `v_component_of_wind_10m`.
- Preserve units expected by the current code: temperature in Kelvin,
  precipitation as daily total depth, and wind components in native wind-speed
  units.
- Reproject/interpolate the coarse ERA5 grid onto the export grid, then let the
  loader spatially average over the patch into `time_x`.
- Use `-9999` for missing days.

Interchangeability risk: the existing ERA5 temperature normalization constant is
inconsistent with its comment. A direct-source pipeline should first match the
current GEE numeric values, then fix normalization as a separate model-change
migration if needed.

### Copernicus DEM from Copernicus Data Space

Target GEE product: `COPERNICUS/DEM/GLO30`.

Direct-source requirements:

- Download Copernicus DEM GLO-30 tiles that cover the exported patch.
- Mosaic and crop tiles before writing the final stack.
- Preserve elevation values in meters with the same vertical datum convention as
  the GEE product.
- Reproject the elevation DEM to the target cell grid (10 m scale, e.g. EPSG:4326) *first* before computing terrain derivatives. Terrain slope and aspect are highly scale-sensitive; computing them on the native 30 m grid and then reprojecting will yield mismatched gradients.
- Compute `slope` and `aspect` in degrees on the reprojected 10 m grid using Horn's algorithm or an equivalent gradient method that closely matches GEE's `ee.Terrain.slope` and `ee.Terrain.aspect` to satisfy downstream masks and normalization.
- Emit bands as `DEM`, `slope`, `aspect`.
- Preserve invalid elevation as `-9999` or values that are masked by the current
  `DEM >= 0.0000001` threshold.

Interchangeability risk: slope and aspect are sensitive to reprojection order.
Compute terrain derivatives in a consistent projected grid, then validate
against GEE-derived patches.

### ESA WorldCover from ESA

Target GEE product: `ESA/WorldCover/v200`.

Direct-source requirements:

- Download ESA WorldCover version 200, which corresponds to the 2021 map used
  by the current GEE collection.
- Preserve class codes exactly: `10`, `20`, `30`, `40`, `50`, `60`, `70`, `80`,
  `90`, `95`, and `100`.
- Mosaic tiles when needed and use nearest-neighbor resampling only.
- Emit one categorical band named `Map` in the static-space slot.
- Do not one-hot encode before writing the interchange GeoTIFF. The current
  dataset loader expects the single `Map` band and performs one-hot encoding
  itself.
- Use `0` or `-9999` only for nodata/unknown pixels, because the loader maps
  those to `-9999` across all one-hot channels.

Interchangeability risk: any class remapping before the loader will break the
expected WorldCover one-hot channel order.

### Cloud Flag Bitfields from Original Products

Target GEE cloud flag bands: MODIS `state_1km`, Sentinel-2 `QA60`, and Landsat
`QA_PIXEL`.

Direct-source requirements:

- Preserve raw QA bitfields rather than decoded cloud masks.
- Keep the current cloud flag order after the science bands for each timestep:
  MODIS, Sentinel-2, then Landsat.
- Use the same fallback logic as the science source for Landsat cloud flags:
  Landsat 9 first, Landsat 8 if Landsat 9 is unavailable.
- Use nearest-neighbor resampling for all QA bands.
- Fill missing QA bands with `-9999` placeholders.

Interchangeability risk: the main dataset loader currently drops cloud flag
bands from model tensors. If a future pipeline starts using them, it should add
explicit bit decoding in one place rather than silently changing the interchange
format.


## Compatibility caveats

- Temporal compositing is minimal: each loader takes `.first()` after filtering
  by date and region. There is no explicit sorting, cloud scoring, median
  composite, or quality mosaic in the main Earth Engine ingestion code.
- Sentinel-2 and Landsat cloud QA bands are exported but not applied as masks in
  the Earth Engine loaders.
- Cloud flag bitfields are not decoded by the main dataset loader.
- All modalities are exported at 10 m in EPSG:4326. Coarser sources therefore
  rely on Earth Engine's export-time resampling before the repository performs
  its own block-mean downsampling.
- The ERA5 temperature baseline shift constant appears inconsistent with the
  comment that says temperatures are shifted to Celsius. With
  `shift=-272.15` and `div=35`, the implemented formula is
  `(K + 272.15) / 35`, not the Celsius conversion `(K - 273.15) / 35`.
- `VIIRS_CLOUD_FLAG_BANDS` and `get_viirs_cloud_flag` exist, but VIIRS cloud
  flags are not wired into `MODALITIES` or `eo.py`.
- The docstring of `Dataset._tif_to_array` (and `LandsatEvalDataset._tif_to_array`)
  reports the medium-resolution output shape as `(3, 3, T, C_STM)`, but the
  actual target is `(NUM_MED_RES_PIXELS_PER_DIM, NUM_MED_RES_PIXELS_PER_DIM, T, C_STM)
  = (5, 5, T, 2)`. The docstring is stale; the code is correct.
- The downstream `LandsatEval` and its ablation subclasses default
  `exclude_prediction_era5=True`, so the ERA5 group is masked on the prediction
  day even when no other ablation flag is set. Override explicitly to keep ERA5.
- `EXPORTED_HEIGHT_WIDTH_METRES = 1000` is the polygon's side length, not its
  half-width. `EarthEngineExporter` derives `surrounding_metres = 500` and
  builds the bounding box symmetrically around the centre point.
- The Earth Engine exporter's window is sampled per (point, season) triple, so
  each input point produces three GeoTIFFs (`early`, `mid`, `late`) per
  exporter run, not one.
- **AOI Coverage and Scene Heterogeneity**: Large fixed-extent AOI mosaics (e.g. composed of a 2x2 grid of ~4 Sentinel-2 or Landsat scenes) suffer from incomplete daily coverage. On any specific date, full AOI coverage is impossible due to varying orbit paths, swathes, and scene boundaries. 
- **Swath Boundary Nodata and Mosaicing**: Scenes/products near orbit edges or swath boundaries often contain significant nodata regions. In Earth Engine, naive `.first()` scene selection on the collection filtered by date and region is sufficient for a single small footprint but fails on cells intersecting scene boundaries or swath edges. A direct-source pipeline must mosaic all valid overlapping scenes/granules acquired on the target day prior to cell cropping to maximize pixel coverage and avoid artificial nodata boundaries within the 1 km grid cells.
- **Same-tile/date multi-product overlap (DISTINCT from cross-tile mosaicing)**: For a *single* reference grid tile on a *single* date, the archive can contain **more than one product** — different relative orbits (e.g. S2 `R070` vs `R113` over the same tile), different satellites (`S2A`/`S2B`), or reprocessing duplicates (same orbit, different PDGS processing time). Each product spans the same tile extent but has **different nodata footprints** (swath-edge geometry, per-product cloud masking). GEE's `.first()` picks exactly one, so a pixel that is nodata in the chosen product but **valid in another same-tile-date product is silently emitted as `-9999`** — a false nodata indistinguishable from a real gap. Verified in `data/bow_valley_selection_raw`: **S2** has ≥7 same-(date,tile) groups with 2–3 products (e.g. `20250331 T11UNT` = R113×2 reprocessing + R070; `20250420 T11UNT`, `20250510 T11UNS` = R113 vs R070); **Landsat 9** has `20250425` path/row `044024` twice. A direct-source pipeline must **coalesce all products sharing the same (tile, date)** per pixel: take the first product with a valid (non-nodata, in-threshold) value at that pixel, falling through to the next where it is nodata; emit `-9999` only where **every** same-tile-date product is nodata. Deterministic product order (e.g. latest processing time first) settles ties. This is per-pixel valid coalescing, NOT averaging (no value blending → preserves the GEE value domain). It is orthogonal to and runs *before* cross-tile mosaicing.

