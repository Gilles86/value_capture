#!/bin/bash
# Submit the full nilearn GLM pipeline:
#   1. First-level GLM per subject (array job)
#   2. Second-level (group) GLM — depends on all first-level jobs
#   3. Per-subject PDF reports (array job) — depends on second-level
#
# Usage:
#   bash submit_nilearn_glm.sh
#
# Optional env overrides (export before calling):
#   SPACE, RES, SMOOTHING_FWHM, FMRIPREP_DERIV, FDR_ALPHA, CLUSTER_THRESHOLD
#
# Defaults to MNI152NLin2009cAsymm res-2 to match the second-level GLM.
#
# Examples:
#   bash submit_nilearn_glm.sh
#   SPACE=T1w RES= bash submit_nilearn_glm.sh  # native space only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Subjects array index range (matches subjects.yml: 01 02 03)
SUBJECT_ARRAY="1-3"

# Defaults to match fit_second_level.sh (MNI152NLin2009cAsymm res-2)
SPACE="${SPACE:-MNI152NLin2009cAsymm}"
RES="${RES:-2}"

# --- Optional export vars passed through to each job ---
EXPORT_VARS=""
[ -n "$SPACE" ]             && EXPORT_VARS="${EXPORT_VARS},SPACE=$SPACE"
[ -n "$RES" ]               && EXPORT_VARS="${EXPORT_VARS},RES=$RES"
[ -n "$SMOOTHING_FWHM" ]    && EXPORT_VARS="${EXPORT_VARS},SMOOTHING_FWHM=$SMOOTHING_FWHM"
[ -n "$FMRIPREP_DERIV" ]    && EXPORT_VARS="${EXPORT_VARS},FMRIPREP_DERIV=$FMRIPREP_DERIV"
[ -n "$FDR_ALPHA" ]         && EXPORT_VARS="${EXPORT_VARS},FDR_ALPHA=$FDR_ALPHA"
[ -n "$CLUSTER_THRESHOLD" ] && EXPORT_VARS="${EXPORT_VARS},CLUSTER_THRESHOLD=$CLUSTER_THRESHOLD"
EXPORT_VARS="${EXPORT_VARS#,}"  # strip leading comma

echo "=== 1. Submitting first-level GLM (array ${SUBJECT_ARRAY}) ==="
FIRST_LEVEL_SUBMIT="sbatch --array=${SUBJECT_ARRAY} --parsable"
[ -n "$EXPORT_VARS" ] && FIRST_LEVEL_SUBMIT="$FIRST_LEVEL_SUBMIT --export=${EXPORT_VARS}"
FIRST_LEVEL_ID=$($FIRST_LEVEL_SUBMIT "$SCRIPT_DIR/fit_nilearn_glm.sh")
echo "  Job ID: ${FIRST_LEVEL_ID}"

echo ""
echo "=== 2. Submitting second-level GLM (depends on all first-level) ==="
SECOND_LEVEL_SUBMIT="sbatch --dependency=afterok:${FIRST_LEVEL_ID} --parsable"
[ -n "$EXPORT_VARS" ] && SECOND_LEVEL_SUBMIT="$SECOND_LEVEL_SUBMIT --export=${EXPORT_VARS}"
SECOND_LEVEL_ID=$($SECOND_LEVEL_SUBMIT "$SCRIPT_DIR/fit_second_level.sh")
echo "  Job ID: ${SECOND_LEVEL_ID}"

echo ""
echo "=== 3. Submitting per-subject reports (array ${SUBJECT_ARRAY}, depends on second-level) ==="
REPORT_SUBMIT="sbatch --array=${SUBJECT_ARRAY} --dependency=afterok:${SECOND_LEVEL_ID} --parsable"
[ -n "$EXPORT_VARS" ] && REPORT_SUBMIT="$REPORT_SUBMIT --export=${EXPORT_VARS}"
REPORT_ID=$($REPORT_SUBMIT "$SCRIPT_DIR/report_nilearn_glm.sh")
echo "  Job ID: ${REPORT_ID}"

echo ""
echo "Pipeline submitted."
echo "  First-level:  ${FIRST_LEVEL_ID}"
echo "  Second-level: ${SECOND_LEVEL_ID}"
echo "  Reports:      ${REPORT_ID}"
echo ""
echo "Monitor with: squeue -u gdehol"
