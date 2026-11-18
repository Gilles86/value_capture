#!/bin/bash
#SBATCH --job-name=vc_decode_cpu
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/decode_trials_cpu_%A_%a.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# CPU version of decode_trials.sh (about 2x slower than an A100, but no GPU queue; 16 cores are no faster than 8).
# Array index = subject*10 + test session, e.g. --array=41,42,51,52
set -eo pipefail
SUB=$(printf "%02d" $((SLURM_ARRAY_TASK_ID / 10)))
SES=$((SLURM_ARRAY_TASK_ID % 10))
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/decode/decode_trials.py "$SUB" \
    --test-session "$SES" ${ROIS:+--rois $ROIS} ${EXTRA_ARGS}
