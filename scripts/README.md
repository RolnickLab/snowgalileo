### File Structure

Here, starter scripts for pre-training, fine-tuning, evaluation, and inference are stored.

Pre-training:

- `scripts/export_for_pretrain.py`: Export pre-training data from Google Earth Engine based on specified sampling points (stored in `data/pretraining_points`). More information can be found in `data/README.md`.
- `scripts/compute_normalization.py`: Compute ML normalization values based on the pre-training dataset. The same values are used for all subsequent training and evaluation steps.
- `scripts/pretrain.py`: SnowGalileo pre-training.

Fine-tuning & Evaluation:

- `scripts/export_for_eval.py`: Export data from Google Earth Engine for evaluation purposes. More information can be found in `data/README.md`.
- `scripts/finetune.py`: SnowGalileo fine-tuning.
- `scripts/finetune_with_clouds.py`: SnowGalileo fine-tuning with generated clouds.
- `scripts/finetune_sweeps.py`: Hyperparameter sweeps for SnowGalileo fine-tuning.
- `scripts/train_sklearn_baseline.py`: Training of random forest, MLP, and support vector regressor.
- `scripts/sklearn_sweeps.py`: Hyperparameter sweeps for training the baseline models.
- `scripts/eval_only.py`: Evaluates finetuned model from checkpoint.
- `scripts/eval_with_clouds.py`: Evaluates finetuned model from checkpoint with generated clouds.
- `scripts/eval_individual_patches.py`: Evaluates the performance per individual tile.
- `scripts/visualize.py`: Used to plot qualitative predictions.
- `scripts/test_sklearn_baseline.py`: Evaluation of random forest, MLP, and support vector regressor.

Inference Execution:

- `scripts/export_for_inference.py`: Export data from Google Earth Engine for inference purposes. More information in `data/README.md`.
- `scripts/run_inference`: Generates output GeoTIFFs including model input and predictions.