#!/bin/bash
#SBATCH --cpus-per-task=4  # request cpus
#SBATCH --mem=24GB  # RAM memory, NOT gpu memory
#SBATCH -p long
#SBATCH --time=3-00:00:00
#SBATCH --error=error_vis.txt

module load anaconda/3
conda activate [YOUR_ENVIRONMENT_NAME]

python -m scripts.predict_and_generate_output --checkpoint_name "hdh8g195.pth" --id "clear_pretrained_42_rockies" \
--eval_config_name "fsc_test_rockies_tiny.json" --decoding_strategy "finetune"