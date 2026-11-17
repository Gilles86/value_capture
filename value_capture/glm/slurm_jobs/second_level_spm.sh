#!/bin/bash -l
#SBATCH --job-name=vc_glm_spm
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/second_level_spm_%j.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# SPM12 one-sample t on the nilearn reward-GLM effect maps: RFT voxel/cluster FWE, whole brain and reward SVC.
set -eo pipefail
module load matlab/r2023b
MATLAB=$(command -v matlab)
test -n "$MATLAB" || { echo "matlab not on PATH after module load"; exit 1; }
export APPTAINER_BIND=/shares/zne.uzh
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/second_level_spm.py --matlab "$MATLAB" "$@"
