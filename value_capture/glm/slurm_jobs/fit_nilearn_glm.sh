#!/bin/bash
#SBATCH --job-name=fit_nilearn_glm
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_nilearn_glm_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Fit nilearn first-level GLM for the value capture fMRI task.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=01 fit_nilearn_glm.sh
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=2 fit_nilearn_glm.sh
#
# Optional overrides (--export key=value):
#   SESSION         session number(s), space-separated (default: all sessions)
#   FMRIPREP_DERIV  fmriprep derivative label (default: fmriprep)
#
# Run as a SLURM array (one job per subject) by omitting PARTICIPANT_LABEL:
#   sbatch --array=1-3 fit_nilearn_glm.sh

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SESSION="${SESSION:-}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SPACE="${SPACE:-T1w}"
RES="${RES:-}"
SMOOTHING_FWHM="${SMOOTHING_FWHM:-6}"
N_JOBS="${N_JOBS:-16}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --space "$SPACE"
)

[ -n "$SESSION" ]        && ARGS+=(--sessions $SESSION)
[ -n "$RES" ]            && ARGS+=(--res "$RES")
[ -n "$SMOOTHING_FWHM" ] && ARGS+=(--smoothing-fwhm "$SMOOTHING_FWHM")
[ -n "$N_JOBS" ]         && ARGS+=(--n-jobs "$N_JOBS")

echo "fit_nilearn_glm: sub-${PARTICIPANT_LABEL}  deriv=${FMRIPREP_DERIV}  space=${SPACE}  res=${RES}  smoothing=${SMOOTHING_FWHM}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/value_capture/glm/fit_nilearn_glm.py" "${ARGS[@]}"
