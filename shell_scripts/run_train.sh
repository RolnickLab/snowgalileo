#!/bin/bash
#SBATCH --cpus-per-task=4  # request cpus
#SBATCH --mem=24GB  # RAM memory, NOT gpu memory
#SBATCH -p long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=5-00:00:00
#SBATCH --error=error_train_500m_50k_with_eval.txt

module load anaconda/3
conda activate presto-v3

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81
export GEO_BENCH_DIR="data/geobench/"

python train.py --run_name_prefix "500m_50k_with_eval" --config_file "ai4snow.json" --h5py_folder "h5pys" --tifs_folder "tifs_all_bands_500m_50k"