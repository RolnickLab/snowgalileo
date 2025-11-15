## A Pretrained Remote Sensing model

Disclaimer about Definitions:

To be able to perform sensor fusion of remote sensing data of different spatial and temporal resolutions, this project lives from grouping data with similar resolutions into distinct data types, and processing these as individual variables throughout the different stages of the algorithm. To increase readability, we use shortcuts as identifier for these data types, and define them in this section:

- `s_t_h_x`: All data stemming from high resolution (10m-30m) satellite imagery, that vary over a timespan of 8 days. These include: Sentinel-1, Sentinel-2, Landsat. The initial shape of these array type will be `(height=100, width=100, timesteps=8, channels=15)`.
- `s_t_m_x`: All data stemming from medium resolution (300m) satellite imagery, varying over time. This includes Sentinel-3. Shape: `(height=5, width=5, timesteps=8, channels=2)`.
- `s_t_l_x`: All data stemming from low resolution (500m) satellite imagery, varying over time. This includes MODIS, 500m-VIIRS (RGB and VNIR), NDSI, NDVI. Shape: `(height=2, width=2, timesteps=8, channels=11)`.
- `sp_x`: All data stemming from high resolution satellite imagery, that is static in time. This includes DEM and WC. Shape: `(height=100, width=100, channels=4)`.
- `t_x`: All time-varying data that is static in space. This includes 1km-VIIRS (RGB, VNIR, SWIR) and ERA5. Shape: `(timesteps=8, channels=9)`.
- `st_x`: All data static over space and time. This includes coordinate information. Shape: `(channels=3)`.

Throughout the processing, the spatial (pixel) dimension gets reduced to a token dimension, and channels (referring to satellite bands, or distinct variables of auxiliary data, e.g. topography elevation and slope are distinct channels) are grouped into channel groups that include data with similar characteristics. For example, all Sentinel-2 RGB channels are grouped in one channel group.

### 0. Structure

Functionalities:
- Download pre-training data from Google Cloud (likely to be changed) to local dir: run ```train_flexipresto_mae.py``` with args.download == true
- Pre-train model: run ```train_flexipresto_mae.py```

Execution files:
- ```train_flexipresto_mae.py```: Presto pre-training
    - Setup (wandb, hyperparameters, etc.)
    - Download Dataset if necessary (needs to be executed once per collaborator to make pre-training possible)
    - Dataloader collate function: creates masks for pre-training
    - Pre-train model for e epochs
    - Evaluate model pre-training on validation task (encoder, with KNN)

- ```export.py```: Export data from Google Earth Engine to Google Cloud (likely to be changed)

Presto Model:
- ```src/flexipresto.py```
    - Encoder: 
        - divides images into patches
        - Projects patches to per-channel-group tokens
        - Adds embeddings (e.g., where is space is the token, or where in time)
        - removes masked tokens
        - Applies attention
        - adds masked tokens
    - Pixel Decoder:
        - gets embedded images
        - Applies attention
        - bring back into pixel space

### 1. Training the model from scratch

The main entrypoint to training a model from scratch is `train_flexipresto_mae.py`.
The hyperparameters of a training run are controlled by the configs in `config`.
For example, the following command trains a [medium](config/mae/medium.json) sized model:

```bash
python train_flexipresto_mae.py --config medium.json
```

Another option is to randomly select hyperparameters to train from, given a fixed encoder size.
Two encoder sizes are available - `tiny` (which has the same encoder size as `medium.json`) and `base`, which mirrors a ViT-B:

```bash
python train_flexipresto_mae.py --config random_tiny
```

Raw data is exported from EarthEngine as `.tif` files. This takes some processing to turn into an ML-ready format, so we save an interim data type (`.h5`). The `.h5` files are stored on WEKA under `/skylight-default/presto-h5pys`.

If you are only using the `.h5` files, then use the flag `--h5pys_only` - otherwise, the script will look for `tif` files as well. Use the `--h5py_folder` command to tell the script where the `.h5` files were mounted (by default, it will look at `data/h5pys`).

ESA contract number: ...