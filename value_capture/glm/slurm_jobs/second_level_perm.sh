#!/bin/bash
#SBATCH --job-name=vc_glm_perm
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/second_level_perm_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
# Permutation group inference (cluster mass + TFCE FWE, reward ROIs) on the reward GLM.
set -eo pipefail
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/fit_second_level_perm.py \
    --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture --n-jobs 16 "$@"
