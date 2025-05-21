#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error15000.txt

module load anaconda/3

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/home/mila/m/marlena.reil/google-cloud-sdk/path.bash.inc' ]; then . '/home/mila/m/marlena.reil/google-cloud-sdk/path.bash.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/home/mila/m/marlena.reil/google-cloud-sdk/completion.bash.inc' ]; then . '/home/mila/m/marlena.reil/google-cloud-sdk/completion.bash.inc'; fi

conda activate presto-v3

python export.py --mode "url" --start_export_from_idx 15000 --tifs_folder "tifs15000"
#python export.py --mode "url" --start_export_from_idx 16000 --tifs_folder "tifs15000"
#python export.py --mode "url" --start_export_from_idx 17000 --tifs_folder "tifs15000"