# Data Retrieval

The following details how SnowGalileo's input data can be retrieved using Google Earth Engine.

## Pre-training Data

The export of pre-training data from Google Earth Engine can be started using the script `scripts/export_for_pretrain.py` based on coordinates listed in a CSV file (stored in `data/pretraining_points/`). Exports are run locally via a download URL using the earthengine CLI and require authentication via a Google account.

Google Earth Engine authentication tokens expire after a limited time. The export script therefore supports restartable exports using a starting index.

Before starting an export session, authenticate with GEE:

`earthengine authenticate` (requires the earthengine-api to be installed)

Next, run the export script:

`python - scripts.export_for_pretrain --start_export_from_idx <index>`

`--export_start_idx` (int) defines the index of the first row from the input CSV file to be processed. Use 0 for the initial run.

If exports stop due to expired authentication:

Re-authenticate using `earthengine authenticate` and restart the export using the index of the next unprocessed CSV row.

Recommended procedure: Run the export script multiple times in parallel (starting from different row indeces) and store the data in individual folders (to be indicated with `--tifs_folder`). Then, combine all exported data into the directory `data/tifs_all_bands/`, since this is the default option for all subsequent scripts. The export script automatically checks whether a file for a given index already exists in the respective output directory.

Note: The export workflow allows export through Cloud Storage or Google Drive as alternative to the URL-Download option.

## Computation of Normalization Values

The normalization values provided with this repository are computed using the entire pre-training dataset and can be used as is for downstream purposes.

We apply per-channel 2\*std normalization based on the procedure in related work (Galileo, CROMA, and SatMAE). We don't apply normalization to naturally scaled variables (NDVI, NDSI, and the one-hot encoded ESA Worldcover), or location. All normalization values were computed using the `scripts/compute_normalization.py` script, which computes running statistics over a given dataset using Welford's algorithm.

## Fine-tuning and Evaluation Data

Fine-tuning and evaluation data is based on labels (or "masks") provided with our data repository. The corresponding input data can also be retrieved from this data repository, or exported via Google Earth Engine based on the time and location of the labels.

1. Unzip the labels into `./data/landsat_eval_masks/`.
2. run `scripts/export_for_eval.py` (all possible input tifs are exported, but the resulting number of files will be smaller than the number of masks bacause of export fails). The same GEE procedure as for the pre-training files needs to be followed. To handle the large number of files to be exported, follow the recommended procedure above.
3. (if the filename coordinates of the labels are in UTM projection) run `scripts/developer_scripts/rename_utm_to_wgs84.py` to reproject the filename identifier to WGS.
4. run `scripts/developer_scripts/copy_matching_eval.py` (all masks that have a matching input will be copied into a new folder)
5. run `scripts/developer_scripts/eval_crop_bounds.py` (all inputs will be cropped to the shape of the masks)
6. Depending on the purpose of the data, follow one of the following steps:

- (for train/test data) run `scripts.train_test_split.py` to split the data into train and test
- (if data is used for evaluation only) move all files into a new subfolder called `test`

## Inference Data

(to-do)
