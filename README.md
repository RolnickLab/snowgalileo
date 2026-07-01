# SnowGalileo: A Pre-trained Transformer for Snow Cover Mapping

This repository contains the code for pre-training, fine-tuning, and evaluating the ESA AI4Snow model "SnowGalileo". SnowGalileo is a transformer model for daily fractional snow cover (FSC) mapping at 100 m resolution, based on multi-sensor Earth observation data.

To reproduce the figures in the accompanying paper, please visit [this Colab notebook](https://colab.research.google.com/drive/1kdggrEBsr7J-ByTAQ-Ga73Z5uCqzSuGe?usp=sharing)

## Python Version

This project uses Python 3.11 and relies on a `Makefile` for standardized, reproducible commands.

## Package & Environment Management

- **Environment & Dependency Management**: [uv](https://docs.astral.sh/uv/) is the recommended default tool for fast, reliable dependency installation and virtual environment creation. It can be configured to use [Poetry](https://python-poetry.org/docs/) or `conda` via `Makefile.variables`.
  - When we mention `conda` in this project, we generally mean `mamba` or `micromamba` See Mamba documentation

Configuration: Review the project-level configurations in Makefile.variables or set individual preferences in Makefile.private.

## Quickstart

### Description of the configs

Also refer to the data retrieval section in data/.

### How to Run Pre-training

For pre-training SnowGalileo, [data] is required. Then run [config] config file.

### How to Run Fine-Tuning

### How to Run Evaluation Experiments

### How to Run Inference on your own Points (preliminary)

Parts of the code require a WandB account to function entirely. If you would like to make use of this, please set the variable [WANDB_ENTITY] in "src/snow_galileo/data/config.py" to your Belieben.

### Detailed Description

#### File Structure

Information about input data export using Google Earth Engine can be found in `data/README.md`.

Pre-training Execution:

- `scripts/export_for_pretrain.py`: Export pre-training data from Google Earth Engine based on specified sampling points (stored in `data/pretraining_points`).
- `scripts/pretrain.py`: Snowgalileo pre-training
  - Setup (wandb, hyperparameters, etc.)
  - Dataloader collate function: creates masks for pre-training
  - Pre-train model for e epochs
  - Evaluate model pre-training on validation task (encoder, with KNN)

Evaluation Execution:

- `scripts/export_for_eval.py`: Export data from Google Earth Engine for evaluation purposes. More post-processing is necessary (TODO: document what exactly)
- `scripts/predict_and_generate_output`: Generates output GeoTIFFs including model input and predictions. Currently only works with already exported data (data paths are specified in the eval config to be passed as argument)
- `scripts/finetune.py`: Main entrypoint for finetuning
- `scripts/finetune_sweeps.py`: Hyperparameter sweeps for finetuning
- `scripts/eval_only.py`: Evaluates finetuned model from checkpoint. Will be main entrypoint for analyzes experiments
- `scripts/visualize.py`: Used to plot qualitative predictions

Data Export:

- `src/snow_galileo/data/earthengine/`
  - contains all code specific to Google Earthengine: sensor-specific export scripts, as well as export files
- `src/snow_galileo/data/dataset.py`
  - contains the pre-training dataset class

Snowgalileo Model:

- `src/snow_galileo/snowgalileo.py`
  - Encoder:
    - divides images into patches
    - projects patches to per-channel-group tokens
    - adds embeddings (e.g., where is space is the token, or where in time)
    - removes masked tokens
    - Applies attention
    - adds masked tokens
  - Pixel Decoder (used for pre-training):
    - gets embedded images
    - Applies attention
    - bring back into pixel space
- `src/snow_galileo/masking.py`
  - creates token masks for pre-training
- `src/snow_galileo/embedding.py`
  - the embeddings that add contextual information to tokens

Finetuning/ Evaluation Setup:

- `src/snow_galileo/eval/patch_predict.py`
  - contains the Finetuning head and functions for finetuning and evaluating the model
- `src/snow_galileo/eval/landsat_eval.py`
  - prepares the Landsat evaluation dataset, and wraps the Landsat-specific evaluation process

#### Disclaimer about Variable Names

To be able to perform sensor fusion of remote sensing data of different spatial and temporal resolutions, this project lives from grouping data with similar resolutions into distinct data types, and processing these as individual variables throughout the different stages of the algorithm. To increase readability, we use shortcuts as identifier for these data types, and define them in this section:

- `s_t_h_x`: All data stemming from high resolution (10m-30m) satellite imagery, that vary over a timespan of 8 days. These include: Sentinel-1, Sentinel-2, Landsat. The initial shape of these array type will be `(height=100, width=100, timesteps=8, channels=15)`.
- `s_t_m_x`: All data stemming from medium resolution (300m) satellite imagery, varying over time. This includes Sentinel-3. Shape: `(height=5, width=5, timesteps=8, channels=2)`.
- `s_t_l_x`: All data stemming from low resolution (500m) satellite imagery, varying over time. This includes MODIS, 500m-VIIRS (RGB and VNIR), NDSI, NDVI. Shape: `(height=2, width=2, timesteps=8, channels=11)`.
- `sp_x`: All data stemming from high resolution satellite imagery, that is static in time. This includes DEM and WC. Shape: `(height=100, width=100, channels=14)`.
- `t_x`: All time-varying data that is static in space. This includes 1km-VIIRS (RGB, VNIR, SWIR) and ERA5. Shape: `(timesteps=8, channels=9)`.
- `st_x`: All data static over space and time. This includes coordinate information. Shape: `(channels=3)`.

Throughout the processing, the spatial (pixel) dimension gets reduced to a token dimension, and channels (referring to satellite bands, or distinct variables of auxiliary data, e.g. topography elevation and slope are distinct channels) are grouped into channel groups that include data with similar characteristics. For example, all Sentinel-2 RGB channels are grouped in one channel group.

### Funding
We are greatful to the ESA AI4Science 4000143295/23/I-DT grant that made this project possible.

### Credits
SatelliteCloudGenerator

More information: Tseng, G., Fuller, A., Reil, M., Herzog, H., Beukema, P., Bastani, F., ... & Rolnick, D. (2025). Galileo: Learning global and local features in pretrained remote sensing models. arXiv e-prints, arXiv-2502.

Original repo: https://github.com/nasaharvest/galileo

This README file is inspired by the advanced lab template of Francis Pelletier.
