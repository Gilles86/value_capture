#!/bin/bash
#SBATCH --job-name=fit_glmsingle
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_glmsingle_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# Fit GLMsingle single-trial betas for the value capture fMRI task.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=01 fit_glmsingle.sh
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=2 fit_glmsingle.sh
#
# Optional overrides (--export key=value):
#   SESSION         session number(s), space-separated (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#   DEBUG           set to "1" to write all 4 model steps + figures (default: off)
#
# Run as a SLURM array (one job per subject) by omitting PARTICIPANT_LABEL:
#   sbatch --array=1-2 fit_glmsingle.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
FWHM="${FWHM:-0}"
DEBUG="${DEBUG:-0}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --fwhm "$FWHM"
)

[ -n "$SESSION" ] && ARGS+=(--sessions $SESSION)
[ "$DEBUG" = "1" ] && ARGS+=(--debug)

echo "fit_glmsingle: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  fwhm=${FWHM}mm  debug=${DEBUG}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/value_capture/glm/fit_glmsingle.py" "${ARGS[@]}"
