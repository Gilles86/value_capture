#!/bin/bash
#SBATCH --job-name=vc_glmsingle_fbreg
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --output=/home/%u/logs/value_capture/glmsingle_fbreg_%A_%a.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# GLMsingle with the points displays as nuisance regressors (fit_glmsingle.py --feedback-regressors).
# Timing/memory as the Snakemake glmsingle rule: ~8-11 h; sub-13/14 need 200G (submit them with --mem=200G).
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/fit_glmsingle.py "$SUB" \
    --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture --feedback-regressors
