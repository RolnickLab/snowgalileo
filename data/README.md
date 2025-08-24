# Data Retrieval

Note: all required shell scripts are stored in the `shell_scripts` folder.

## Pre-training Data
1) run the `run[number].sh` scripts 3x each
2) run `run_copy_files.sh` to copy individual tifs into one large tif folder
3) run `run_export_checks.sh` to perform some checks on the exports

## Landsat Evaluation Data
1) run `run_eval_export.sh` (all possible input tifs are exported, the resulting number of files will be smaller than the number of masks)
2) run `run_copy_eval.sh` (all masks that have a matching input will be copied into a new folder)
3) run `run_crop_eval.sh` (all inputs will be cropped to the shape of the masks)
4) run `scripts.train_test_split.py` to split the data into train and test
5) create a folder `landsat_eval_h5pys` and subfolders `train` and `test` within DATA_FOLDER
