#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error3000.txt

module load anaconda/3

# The next line updates PATH for the Google Cloud SDK.
if [ -f '/home/mila/m/marlena.reil/google-cloud-sdk/path.bash.inc' ]; then . '/home/mila/m/marlena.reil/google-cloud-sdk/path.bash.inc'; fi

# The next line enables shell command completion for gcloud.
if [ -f '/home/mila/m/marlena.reil/google-cloud-sdk/completion.bash.inc' ]; then . '/home/mila/m/marlena.reil/google-cloud-sdk/completion.bash.inc'; fi

conda activate presto-v3
export GOOGLE_APPLICATION_CREDENTIALS="/home/mila/m/marlena.reil/scratch/ai4snow/presto-v3/ee-marlena-credentials.json"

python export.py --mode "url" --start_export_from_idx 0 --tifs_folder "tifs3000"
#python export.py --mode "url" --start_export_from_idx 1000 --tifs_folder "tifs3000"
#python export.py --mode "url" --start_export_from_idx 2000 --tifs_folder "tifs3000"