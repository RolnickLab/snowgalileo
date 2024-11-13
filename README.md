## A Pretrained Remote Sensing model

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
