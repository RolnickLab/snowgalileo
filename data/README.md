# Data Retrieval

## How to Export Pre-training Data

The export of pre-training data from Google Earth Engine can be kicked off with the script `export_for_pretrain.py` based on coordinates listed in a CSV file (stored in `pretraining_points/`). Exports are run locally via a download URL using the earthengine CLI and require authentication via a Google account.

Google Earth Engine authentication tokens expire after a limited time. The export workflow therefore supports restartable exports using a starting index.

Before starting an export session, authenticate with GEE:

`earthengine authenticate` (requires the earthengine-api to be installed)

Next, run the export script:

`python export_for_pretrain.py --export_start_idx <index> <folder>`

`--export_start_idx` (int) defines the index of the first row from the input CSV file to be processed. Use 0 for the initial run.

If exports stop due to expired authentication:

Re-authenticate using `earthengine authenticate` and restart the export using the index of the next unprocessed CSV row. 

Recommended procedure: Run the export script multiple times in parallel (starting from different row indeces) and store the data in individual folders (to be indicated with `--tifs_folder`). Then, combine all exported data using `scripts/copy_tifs.py`. We recommend using the directory `data/tifs_all_bands/` to combine all data, since this is the default option for all subsequent scripts. Both the export and the copy scripts automatically check whether a file for a given index already exists in the respective output directory.

We provide the shell scripts that we used for export kick-off in the `shell_scripts` folder.

Note: The export workflow allows export through Cloud Storage or Google Drive as alternative to the URL-Download option (not tested).

This results in ~149,496 files being stored in `data/tifs_all_bands/`.

## Computation of Normalization Values
The normalization values provided with this repository are computed using the entire pre-training dataset and can be used as is for downstream purposes.

We apply per-channel 2*std normalization based on the procedure in related work (Galileo, CROMA, and SatMAE). We don't apply normalization to naturally scaled variables (NDVI, NDSI, and one-hot encoded ESA Worldcover Map), or location. All normalization values were computed using the `scripts/compute_normalization.py` script, which computes running statistics over a given dataset using Welford's algorithm.

## Landsat Evaluation Data
1) run `run_eval_export.sh` (all possible input tifs are exported, the resulting number of files will be smaller than the number of masks bacause of export fails)
2) run `run_copy_eval.sh` (all masks that have a matching input will be copied into a new folder)
3) run `run_crop_eval.sh` (all inputs will be cropped to the shape of the masks)
4) run `scripts.train_test_split.py` to split the data into train and test
5) create a folder `landsat_eval_h5pys` and subfolders `train` and `test` within DATA_FOLDER
