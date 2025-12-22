# Data Retrieval

## Pre-training Data

The export of pre-training data from Google Earth Engine can be started using the script `export_for_pretrain.py` based on coordinates listed in a CSV file (stored in `pretraining_points/`). Exports are run locally via a download URL using the earthengine CLI and require authentication via a Google account.

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

## Evaluation Data

The evaluation data consists of two parts: labels that are provided via... and need to be stored at the right location, and input data, which should be exported via Google Earth Engine based on the time and location identifier of the labels.

First, the labels need to be unzipped into `./data/landsat_eval_masks/`. To do so, for example, store the .zip file in the root directory, then run `unzip [file] -d "data/landsat_eval_masks/"`. All masks will now be stored in `./data/landsat_eval_masks/patches_UTM_5_95/`.

Since the coordinates will be processed in EPGS4326 format, the filenames need to be renamed from UTM to EPGS4326. To do so, run the script `scripts/rename_utm_to_wgs84.py`. This will modify the filenames of all files in `./data/landsat_eval_masks/patches_UTM_5_95/`.

## Landsat-based FSC Data
1) run `run_eval_export.sh` (all possible input tifs are exported, the resulting number of files will be smaller than the number of masks bacause of export fails). The same GEE procedure as for the pre-training files needs to be followed. To handle the large number of files to be exported, follow the recommended procedure above.
2) run `run_copy_eval.sh` (all masks that have a matching input will be copied into a new folder)
3) run `run_crop_eval.sh` (all inputs will be cropped to the shape of the masks)
4) run `scripts.train_test_split.py` to split the data into train and test (skip this step if the data is used for evaluation only)

### FSC Training Distribution
<img width="850" height="470" alt="fsc_train_mean" src="https://github.com/user-attachments/assets/9081bf2a-52db-438c-8919-8bb91a2b199f" />
<img width="857" height="451" alt="fsc_train_unique" src="https://github.com/user-attachments/assets/9a2a9c4c-67b0-49f3-9dd0-22195fe365e0" />

### FSC Test Distribution Rockies
<img width="850" height="470" alt="fsc_test_rockies_mean" src="https://github.com/user-attachments/assets/3f49d62c-a951-4d4b-b840-05dbcebb0500" />
<img width="848" height="451" alt="fsc_test_rockies_unique" src="https://github.com/user-attachments/assets/a382aaab-3883-4184-a932-5bd767b935a7" />

### FSC Test Distribution Switzerland
<img width="850" height="470" alt="fsc_test_switzerland_mean" src="https://github.com/user-attachments/assets/ede36ed8-a415-450c-867d-166fa76c6bfc" />
<img width="848" height="451" alt="fsc_test_switzerland_unique" src="https://github.com/user-attachments/assets/dbad80db-e7d4-473e-b8ea-fd1b3093bfc9" />

