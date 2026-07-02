### File Structure

Pre-training Execution:

- `scripts/export_for_pretrain.py`: Export pre-training data from Google Earth Engine based on specified sampling points (stored in `data/pretraining_points`).
- `scripts/pretrain.py`: Snowgalileo pre-training
  - Setup (wandb, hyperparameters, etc.)
  - Dataloader collate function: creates masks for pre-training
  - Pre-train model for e epochs
  - Evaluate model pre-training on validation task (encoder, with KNN)

Evaluation Execution:

- `scripts/export_for_eval.py`: Export data from Google Earth Engine for evaluation purposes. More post-processing is necessary (TODO: document what exactly)
- `scripts/run_inference`: Generates output GeoTIFFs including model input and predictions. Currently only works with already exported data (data paths are specified in the eval config to be passed as argument)
- `scripts/finetune.py`: Main entrypoint for finetuning
- `scripts/finetune_sweeps.py`: Hyperparameter sweeps for finetuning
- `scripts/eval_only.py`: Evaluates finetuned model from checkpoint. Will be main entrypoint for analyzes experiments
- `scripts/visualize.py`: Used to plot qualitative predictions