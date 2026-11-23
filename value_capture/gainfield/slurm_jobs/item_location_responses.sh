#!/bin/bash
#SBATCH --job-name=vc_itemloc
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/value_capture/item_location_%A_%a.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
# Single-trial responses at the item locations.   sbatch --array=4-15 item_location_responses.sh
set -eo pipefail
SUB=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
export TMPDIR="/scratch/$USER/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1 KERAS_BACKEND=jax JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
cd "$HOME/git/value_capture"
exec $HOME/data/conda/envs/value_capture/bin/python -u value_capture/gainfield/item_location_responses.py "$SUB" ${EXTRA_ARGS}
