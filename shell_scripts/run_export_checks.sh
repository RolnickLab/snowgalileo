#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt

module load anaconda/3

conda activate presto-v3
python -m scripts.check_exports --tif_folder "tifs_all"