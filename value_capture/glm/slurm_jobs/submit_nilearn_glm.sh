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
# Defaults to MNI152NLin2009cAsym res-2 to match the second-level GLM.
#
# Examples:
#   bash submit_nilearn_glm.sh
#   SPACE=T1w RES= bash submit_nilearn_glm.sh  # native space only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Subjects array index range (matches subjects.yml: 01 02 03)
SUBJECT_ARRAY="1-3"

# Defaults to match fit_second_level.sh (MNI152NLin2009cAsym res-2)
SPACE="${SPACE:-MNI152NLin2009cAsym}"
RES="${RES:-2}"

# --- Optional env vars passed through to each job (in sbatch's environment,
# not as an sbatch export list: lists trigger "user env retrieval failed" holds) ---
ENV_VARS=()
[ -n "$SPACE" ]             && ENV_VARS+=("SPACE=$SPACE")
[ -n "$RES" ]               && ENV_VARS+=("RES=$RES")
[ -n "$SMOOTHING_FWHM" ]    && ENV_VARS+=("SMOOTHING_FWHM=$SMOOTHING_FWHM")
[ -n "$FMRIPREP_DERIV" ]    && ENV_VARS+=("FMRIPREP_DERIV=$FMRIPREP_DERIV")
[ -n "$FDR_ALPHA" ]         && ENV_VARS+=("FDR_ALPHA=$FDR_ALPHA")
[ -n "$CLUSTER_THRESHOLD" ] && ENV_VARS+=("CLUSTER_THRESHOLD=$CLUSTER_THRESHOLD")

echo "=== 1. Submitting first-level GLM (array ${SUBJECT_ARRAY}) ==="
FIRST_LEVEL_ID=$(env "${ENV_VARS[@]}" sbatch --array=${SUBJECT_ARRAY} --parsable \
    "$SCRIPT_DIR/fit_nilearn_glm.sh")
echo "  Job ID: ${FIRST_LEVEL_ID}"

echo ""
echo "=== 2. Submitting second-level GLM (depends on all first-level) ==="
SECOND_LEVEL_ID=$(env "${ENV_VARS[@]}" sbatch --dependency=afterok:${FIRST_LEVEL_ID} --parsable \
    "$SCRIPT_DIR/fit_second_level.sh")
echo "  Job ID: ${SECOND_LEVEL_ID}"

echo ""
echo "=== 3. Submitting per-subject reports (array ${SUBJECT_ARRAY}, depends on second-level) ==="
REPORT_ID=$(env "${ENV_VARS[@]}" sbatch --array=${SUBJECT_ARRAY} --dependency=afterok:${SECOND_LEVEL_ID} --parsable \
    "$SCRIPT_DIR/report_nilearn_glm.sh")
echo "  Job ID: ${REPORT_ID}"

echo ""
echo "Pipeline submitted."
echo "  First-level:  ${FIRST_LEVEL_ID}"
echo "  Second-level: ${SECOND_LEVEL_ID}"
echo "  Reports:      ${REPORT_ID}"
echo ""
echo "Monitor with: squeue -u gdehol"
