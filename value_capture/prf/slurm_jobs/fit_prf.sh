#!/bin/bash
#SBATCH --job-name=fit_prf
#SBATCH --account=zne.uzh
#SBATCH --output=/home/%u/logs/value_capture/fit_prf_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100   # L4 (24 GB) runs out of GPU memory in the grid search on the 10k-pixel grid
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Fit GaussianPRF2D model to GLMsingle single-trial betas.
#
# Usage:
#   PARTICIPANT_LABEL=01 sbatch fit_prf.sh
#   PARTICIPANT_LABEL=01 SESSION=2 sbatch fit_prf.sh
#   PARTICIPANT_LABEL=01 GM_THRESHOLD=0.5 sbatch fit_prf.sh
#
# Run as a SLURM array (one job per subject) by omitting PARTICIPANT_LABEL:
#   sbatch --array=1-2 fit_prf.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
GM_THRESHOLD="${GM_THRESHOLD:-0.1}"   # liberal: keeps ~all cortex incl. sulcal depths
GD_ITERATIONS="${GD_ITERATIONS:-1000}"
GLMSINGLE_DERIV="${GLMSINGLE_DERIV:-glmsingle}"
VOXEL_CHUNK_SIZE=""  # A100 40GB handles 25k voxels easily (10k-pixel grid: ~1 GB per chunk)
PARADIGM="${PARADIGM:-bar_items}"
TRAIN_SESSIONS="${TRAIN_SESSIONS:-}"   # e.g. 1 -> fit on session 1 only (cross-validation)   # bar + search items on a ±5° grid -> prf_<deriv>_items

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --gd-iterations "$GD_ITERATIONS"
    --glmsingle-deriv "$GLMSINGLE_DERIV"
    --paradigm "$PARADIGM"
)

[ -n "$SESSION" ]          && ARGS+=(--sessions $SESSION)
[ -n "$GM_THRESHOLD" ]     && ARGS+=(--gm-threshold "$GM_THRESHOLD")
[ -n "$TRAIN_SESSIONS" ]   && ARGS+=(--train-sessions $TRAIN_SESSIONS)
[ -n "$VOXEL_CHUNK_SIZE" ] && ARGS+=(--voxel-chunk-size "$VOXEL_CHUNK_SIZE")

echo "fit_prf: sub-${PARTICIPANT_LABEL}  glmsingle_deriv=${GLMSINGLE_DERIV}  gm_threshold=${GM_THRESHOLD:-none}  gd_iter=${GD_ITERATIONS}"
echo "Args: ${ARGS[*]}"

source $HOME/data/miniforge3/etc/profile.d/conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

# jax[cuda12] ships its own CUDA libraries — no module load needed.
# Print devices so the log confirms GPU is visible.
python -c "import jax; print('JAX devices:', jax.devices())"

python -u "$REPO/value_capture/prf/fit_prf.py" "${ARGS[@]}"
