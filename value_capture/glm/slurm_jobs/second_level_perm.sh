#!/bin/bash
#SBATCH --job-name=vc_glm_perm
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/second_level_perm_%A_%a.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --qos=normal
# Permutation group inference (cluster mass + TFCE FWE, reward ROIs) on the reward GLM, one contrast per array
# task (TFCE whole brain ~1-1.5 h per 5000 permutations on 16 CPUs):
#   sbatch --array=0-7 second_level_perm.sh
set -eo pipefail
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/fit_second_level_perm.py \
    --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture --n-jobs 16 --design "${DESIGN:-logreward}" \
    ${SLURM_ARRAY_TASK_ID:+--contrasts $($HOME/data/conda/envs/value_capture/bin/python -c "from value_capture.glm.fit_second_level_perm import contrasts_for; print(contrasts_for(\"${DESIGN:-logreward}\")[$SLURM_ARRAY_TASK_ID])")} "$@"
