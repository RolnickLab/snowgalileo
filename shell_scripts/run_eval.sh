#!/bin/bash
#SBATCH --cpus-per-task=4  # request cpus
#SBATCH --mem=24GB  # RAM memory, NOT gpu memory
#SBATCH -p long
#SBATCH --time=5-00:00:00
#SBATCH --error=error_eval_resampled.txt

module load anaconda/3
conda activate presto-v3

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81
export GEO_BENCH_DIR="data/geobench/"

python eval.py --output_folder ""