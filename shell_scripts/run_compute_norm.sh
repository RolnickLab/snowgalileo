#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB

module load anaconda/3

conda activate presto-v3
python -m scripts.compute_norm --tifs_folder "all_test" --estimate_from 5
