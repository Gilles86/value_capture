#!/bin/bash
#SBATCH --job-name=vc_decode_agg
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/aggregate_decoded_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
# Reduce all decoded trials to gain summaries and the trial-wise gain table: <bids>/derivatives/decoded_gain/
set -eo pipefail
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/decode/aggregate_decoded.py \
    --out-dir /shares/zne.uzh/gdehol/ds-valuecapture/derivatives/decoded_gain
