#!/bin/bash
#SBATCH --job-name=vc_valuerois
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/value_rois_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Value / colour in MTL, striatum, OFC and visual controls (univariate + decoding).   sbatch --array=4-15 value_rois.sh
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/decode/value_rois.py "$SUB" ${EXTRA_ARGS}
