#!/bin/bash
#SBATCH --job-name=fit_prf
#SBATCH --output=/home/gdehol/logs/fit_prf_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fit GaussianPRF2D model to GLMsingle single-trial betas.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=01 fit_prf.sh
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=2 fit_prf.sh
#   sbatch --export=PARTICIPANT_LABEL=01,GM_THRESHOLD=0.5 fit_prf.sh
#
# Run as a SLURM array (one job per subject) by omitting PARTICIPANT_LABEL:
#   sbatch --array=1-2 fit_prf.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
GM_THRESHOLD="${GM_THRESHOLD:-}"
GD_ITERATIONS="${GD_ITERATIONS:-1000}"
GLMSINGLE_DERIV="${GLMSINGLE_DERIV:-glmsingle}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --gd-iterations "$GD_ITERATIONS"
    --glmsingle-deriv "$GLMSINGLE_DERIV"
)

[ -n "$SESSION" ]       && ARGS+=(--sessions $SESSION)
[ -n "$GM_THRESHOLD" ]  && ARGS+=(--gm-threshold "$GM_THRESHOLD")

echo "fit_prf: sub-${PARTICIPANT_LABEL}  glmsingle_deriv=${GLMSINGLE_DERIV}  gm_threshold=${GM_THRESHOLD:-none}  gd_iter=${GD_ITERATIONS}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
module load gpu

conda run -n value_capture python -u \
    "$REPO/value_capture/prf/fit_prf.py" \
    "${ARGS[@]}"
