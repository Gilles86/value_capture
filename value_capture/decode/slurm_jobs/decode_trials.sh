#!/bin/bash
#SBATCH --job-name=vc_decode_trials
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/decode_trials_%A_%a.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Decode held-out-session trials (model-free gain readout). Array index = subject*10 + test session,
# e.g. --array=41,42,51,52,...   TEST: sbatch --array=71 ...
set -eo pipefail
SUB=$(printf "%02d" $((SLURM_ARRAY_TASK_ID / 10)))
SES=$((SLURM_ARRAY_TASK_ID % 10))
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/decode/decode_trials.py "$SUB" \
    --test-session "$SES" ${ROIS:+--rois $ROIS} ${EXTRA_ARGS}
