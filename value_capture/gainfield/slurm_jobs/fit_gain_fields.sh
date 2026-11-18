#!/bin/bash
#SBATCH --job-name=vc_gain_fields
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/gain_fields_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Gain fields at the target and at low- / high-value distractors, per subject x ROI, with per-run gains.
#   sbatch --array=4-15 fit_gain_fields.sh     TEST: ROIS=V1 sbatch --array=7 fit_gain_fields.sh
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/gainfield/fit_gain_fields.py "$SUB" --per-run \
    ${ROIS:+--rois $ROIS} ${EXTRA_ARGS}
