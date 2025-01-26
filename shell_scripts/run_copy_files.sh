#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt

module load anaconda/3

conda activate presto-v3

python -m scripts.copy_tifs --src_folder "tifs3000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs6000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs9000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs12000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs15000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs18000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs21000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs24000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs27000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs30000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs33000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs36000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs39000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs42000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs45000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs48000" --dest_folder "tifs_all_bands_500m"
python -m scripts.copy_tifs --src_folder "tifs51000" --dest_folder "tifs_all_bands_500m"