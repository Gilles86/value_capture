#!/bin/bash
#SBATCH --job-name=vc_gain_cpu
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/gain_fields_cpu_%A_%a.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# CPU version of fit_gain_fields.sh (about 1.7x slower than an A100, but no GPU queue).   sbatch --array=4-15 fit_gain_fields_cpu.sh
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/gainfield/fit_gain_fields.py "$SUB" --per-run \
    ${ROIS:+--rois $ROIS} ${EXTRA_ARGS}
