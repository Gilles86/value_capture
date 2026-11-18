#!/bin/bash
#SBATCH --job-name=vc_glmsingle_fbreg
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --output=/home/%u/logs/value_capture/glmsingle_fbreg_%A_%a.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# GLMsingle with the points displays modelled, GLMSINGLE_FLAG required: --feedback-regressors (canonical-HRF
# nuisance regressors -> glmsingle_fbreg) or --feedback-events (own single-trial betas, per-voxel HRF ->
# glmsingle_fbevents).   GLMSINGLE_FLAG=--feedback-events sbatch --array=4-12,15 glmsingle_fbreg.sh
# Timing/memory as the Snakemake glmsingle rule: ~8-11 h; sub-13/14 need 200G (submit them with --mem=200G).
set -eo pipefail
FLAG="${GLMSINGLE_FLAG:-}"
# Never run without a feedback flag: that would write to (and overwrite) the plain glmsingle derivative.
case "$FLAG" in --feedback-regressors|--feedback-events) ;; *) echo "set GLMSINGLE_FLAG"; exit 1;; esac
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/glm/fit_glmsingle.py "$SUB" \
    --bids-folder /shares/zne.uzh/gdehol/ds-valuecapture "$FLAG"
