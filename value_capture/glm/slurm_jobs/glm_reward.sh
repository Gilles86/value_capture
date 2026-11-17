#!/bin/bash
#SBATCH --job-name=vc_glm_reward
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/glm_reward_%A_%a.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# First-level reward GLM (fit_nilearn_glm_reward.py), MNI 2 mm, 6 mm smoothing.  sbatch --array=4-15 ...
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/fit_nilearn_glm_reward.py "$SUB" \
    --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture --n-jobs 8
