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
2) (if filename locations are in UTM projection) run `run_rename_utm.sh` to reproject the filename identifier to WGS.
3) run `run_copy_eval.sh` (all masks that have a matching input will be copied into a new folder)
4) run `run_crop_eval.sh` (all inputs will be cropped to the shape of the masks)
5) Depending on the purpose of the data, follow one of the following steps:
- (for train/test data) run `scripts.train_test_split.py` to split the data into train and test 
- (if data is used for evaluation only) move all files into a new subfolder called `test`

## How to transfer new images from LRZ → Mila
(for example, to generate more patches, to generate a different patch size, or a different FSC distribution)
1) Log in to Terrabyte and create an interactive jupyter notebook session with a micromamba environment specified that has all the required packages installed.
2) Within the “uniform_extraction_points.ipynb” script, change your desired parameters (e.g., “patch_size”, “patches_per_bin”, “bin_labels” or “resample_dir”).
3) Run the script and store the outputs in the desired “patch_dir”.

## How to process large images
1) In the “uniform_extraction_points.ipynb” scripts, set “patch_size” to 20 (and optionally reduce the number of patches per bin).
2) Download the generated patches using Git Bash and scp.
3) Upload the patches to the Mila cluster using Ubuntu and scp.
4) Export the corresponding earthengine input images. We have to use the “drive” mode, since the files will exceed the file size limit of “url”. Files will be stored in your drive folder → download locally → upload to Mila and store in tifs folder.
5) Rename the patches and inputs from UTM to WGS84
6) (optionally subset the label patches if not all have been exported).
7) Crop the input images to a size (200, 200) with the eval_crop_bounds.py script.
Subset the input and label images into 4 using GDAL and the command line (using a different conda environment, since GDAL is not compatible with our numpy version).

- `gdalinfo LC09_20230216_FSC55_3138166.88837_446880.07876.tif | grep "Size is"`

- `gdal_retile.py -ps 100 100 -targetDir tiles LC09_20230216_FSC55_3138166.88837_446880.07876.tif` (for masks, use `-ps 10 10`)

9) Create a folder “inference” and store the resulting subsets here.
10) Run the generate_outputs script with an eval config that specifies the input and label folders.
11) Use GDAL (within gdal environment) to stitch the single components back together:

```
for base in $(ls *_1_1_with_preds.tif | sed 's/_1_1_with_preds.tif//'); do
  gdalbuildvrt "${base}_mosaic.vrt" \
       "${base}_1_1_with_preds.tif" \
       "${base}_1_2_with_preds.tif" \
       "${base}_2_1_with_preds.tif" \
       "${base}_2_2_with_preds.tif"
done
```

```
for vrt in *_mosaic.vrt; do
  gdal_translate "$vrt" "${vrt%.vrt}.tif"
done
```

  
### FSC Training Distribution Full Set Balanced
<img width="859" height="470" alt="balanced_train_mean" src="https://github.com/user-attachments/assets/439c722a-b7cc-437a-a5be-1ffe78a4923a" />
<img width="826" height="451" alt="balanced_train_unique" src="https://github.com/user-attachments/assets/26f1cb53-ec7f-440a-b2b7-1ffc01a23fb3" />
<img width="850" height="470" alt="balanced_test_mean" src="https://github.com/user-attachments/assets/b7723925-563a-49a6-8686-d5e1ae81f653" />
<img width="857" height="451" alt="balanced_test_unique" src="https://github.com/user-attachments/assets/4838f7bc-6c67-4639-a87b-65ff793a05a0" />

### FSC Training Distribution Full Set
<img width="859" height="470" alt="fsc_train_new_mean" src="https://github.com/user-attachments/assets/86edba8d-46bf-43ba-aa0c-6574851e5720" />
<img width="826" height="451" alt="fsc_train_new_unique" src="https://github.com/user-attachments/assets/38830850-42be-45ae-af6f-dc3c4fa27860" />

### FSC Training Distribution Subset
<img width="850" height="470" alt="fsc_train_mean" src="https://github.com/user-attachments/assets/9081bf2a-52db-438c-8919-8bb91a2b199f" />
<img width="857" height="451" alt="fsc_train_unique" src="https://github.com/user-attachments/assets/9a2a9c4c-67b0-49f3-9dd0-22195fe365e0" />

### FSC Test Distribution Rockies
<img width="850" height="470" alt="mean_in_rockies" src="https://github.com/user-attachments/assets/a3432bcb-8833-41da-8392-97090aa274e7" />
<img width="848" height="451" alt="unique_in_rockies" src="https://github.com/user-attachments/assets/0be610f0-ce9f-4fb2-a401-32cd1b6890f5" />

### FSC Test Distribution Switzerland
<img width="850" height="470" alt="fsc_test_switzerland_mean" src="https://github.com/user-attachments/assets/ede36ed8-a415-450c-867d-166fa76c6bfc" />
<img width="848" height="451" alt="fsc_test_switzerland_unique" src="https://github.com/user-attachments/assets/dbad80db-e7d4-473e-b8ea-fd1b3093bfc9" />

