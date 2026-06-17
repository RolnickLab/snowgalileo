## A Pretrained Remote Sensing model

### TO-DO: Installation info

### TO-DO: README about the configs


### How to Run Pre-training
For pre-training SnowGalileo, [data] is required. Then run [config] config file.

### How to Run Fine-Tuning


### How to Run Evaluation Experiments


### How to Run Inference on your own Points (preliminary)


### A note about Datasets and adding datasets
Very manual at this point. To start with adding datasets, the following points should be considered:

- Order into a modality group (one of ...,...,...,...,... -- a description is provided in...)

The no-data bounds are based on whether Earthengine has flagged this data as no data (e.g., see s2 script), or manually identified bounds. More information in "data/config.py"


### Other Disclaimers
Some functions in this codebase have been generated, or is inspired with the help of AI. Thorough manual verification and automatic tests have been used to review and verify the code.

Parts of the code require a WandB account to function entirely. If you would like to make use of this, please set the variable [WANDB_ENTITY] in "src/data/config.py" to your Belieben.


### File Structure

Information about input data export and data distributions can be found in `data/README.md`.

Pre-training Execution:
- ```scripts/export_for_pretrain.py```: Export pre-training data from Google Earth Engine based on specified sampling points (stored in ```data/pretraining_points```).
- ```scripts/pretrain.py```: Snowgalileo pre-training
    - Setup (wandb, hyperparameters, etc.)
    - Dataloader collate function: creates masks for pre-training
    - Pre-train model for e epochs
    - Evaluate model pre-training on validation task (encoder, with KNN)

Evaluation Execution:
- ```scripts/export_for_eval.py```: Export data from Google Earth Engine for evaluation purposes. More post-processing is necessary (TODO: document what exactly)
- ```scripts/predict_and_generate_output```: Generates output GeoTIFFs including model input and predictions. Currently only works with already exported data (data paths are specified in the eval config to be passed as argument)
- ```scripts/finetune.py```: Main entrypoint for finetuning
- ```scripts/finetune_sweeps.py```: Hyperparameter sweeps for finetuning
- ```scripts/eval_only.py```: Evaluates finetuned model from checkpoint. Will be main entrypoint for analyzes experiments
- ```scripts/visualize.py```: Used to plot qualitative predictions

Data Export:
- ```src/data/earthengine/```
    - contains all code specific to Google Earthengine: sensor-specific export scripts, as well as export files
- ```src/data/dataset.py```
    - contains the pre-training dataset class

Snowgalileo Model:
- ```src/snowgalileo.py```
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
- ```src/masking.py```
    - creates token masks for pre-training
- ```src/embedding.py```
    - the embeddings that add contextual information to tokens

Finetuning/ Evaluation Setup:
- ```src/eval/patch_predict.py```
    - contains the Finetuning head and functions for finetuning and evaluating the model
- ```src/eval/landsat_eval.py```
    - prepares the Landsat evaluation dataset, and wraps the Landsat-specific evaluation process

#### TO-DO: Describe entire pipeline from input data export to output generation

### Disclaimer about Definitions

To be able to perform sensor fusion of remote sensing data of different spatial and temporal resolutions, this project lives from grouping data with similar resolutions into distinct data types, and processing these as individual variables throughout the different stages of the algorithm. To increase readability, we use shortcuts as identifier for these data types, and define them in this section:

- `s_t_h_x`: All data stemming from high resolution (10m-30m) satellite imagery, that vary over a timespan of 8 days. These include: Sentinel-1, Sentinel-2, Landsat. The initial shape of these array type will be `(height=100, width=100, timesteps=8, channels=15)`.
- `s_t_m_x`: All data stemming from medium resolution (300m) satellite imagery, varying over time. This includes Sentinel-3. Shape: `(height=5, width=5, timesteps=8, channels=2)`.
- `s_t_l_x`: All data stemming from low resolution (500m) satellite imagery, varying over time. This includes MODIS, 500m-VIIRS (RGB and VNIR), NDSI, NDVI. Shape: `(height=2, width=2, timesteps=8, channels=11)`.
- `sp_x`: All data stemming from high resolution satellite imagery, that is static in time. This includes DEM and WC. Shape: `(height=100, width=100, channels=14)`.
- `t_x`: All time-varying data that is static in space. This includes 1km-VIIRS (RGB, VNIR, SWIR) and ERA5. Shape: `(timesteps=8, channels=9)`.
- `st_x`: All data static over space and time. This includes coordinate information. Shape: `(channels=3)`.

Throughout the processing, the spatial (pixel) dimension gets reduced to a token dimension, and channels (referring to satellite bands, or distinct variables of auxiliary data, e.g. topography elevation and slope are distinct channels) are grouped into channel groups that include data with similar characteristics. For example, all Sentinel-2 RGB channels are grouped in one channel group.

ESA AI4Snow contract number: ...

### Credits

This repo inherits from the Galileo model.

More information: Tseng, G., Fuller, A., Reil, M., Herzog, H., Beukema, P., Bastani, F., ... & Rolnick, D. (2025). Galileo: Learning global and local features in pretrained remote sensing models. arXiv e-prints, arXiv-2502.

Original repo: https://github.com/nasaharvest/galileo
