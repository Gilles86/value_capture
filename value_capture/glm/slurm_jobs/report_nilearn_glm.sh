#!/bin/bash
#SBATCH --job-name=report_nilearn_glm
#SBATCH --output=/home/gdehol/logs/report_nilearn_glm_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Generate per-subject FDR-thresholded GLM contrast report PDFs.
#
# Usage:
#   sbatch --array=1-3 report_nilearn_glm.sh
#   sbatch --export=PARTICIPANT_LABEL=01 report_nilearn_glm.sh
#
# Optional overrides (--export key=value):
#   SPACE             BIDS space entity (default: T1w)
#   RES               BIDS res entity (default: none)
#   FDR_ALPHA         FDR significance level (default: 0.05)
#   CLUSTER_THRESHOLD minimum cluster size in voxels (default: 10)

if [ -z "$PARTICIPANT_LABEL" ]; then
    PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
fi

SPACE="${SPACE:-T1w}"
RES="${RES:-}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
CLUSTER_THRESHOLD="${CLUSTER_THRESHOLD:-10}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-valuecapture
REPO=$HOME/git/value_capture

ARGS=(
    "$PARTICIPANT_LABEL"
    --bids-folder "$BIDS_FOLDER"
    --space "$SPACE"
    --fdr-alpha "$FDR_ALPHA"
    --cluster-threshold "$CLUSTER_THRESHOLD"
)

[ -n "$RES" ] && ARGS+=(--res "$RES")

echo "report_nilearn_glm: sub-${PARTICIPANT_LABEL}  space=${SPACE}  fdr_alpha=${FDR_ALPHA}  cluster_threshold=${CLUSTER_THRESHOLD}"
echo "Args: ${ARGS[*]}"

. $HOME/init_conda.sh
conda activate value_capture
export PYTHONUNBUFFERED=1

python -u "$REPO/value_capture/glm/report_nilearn_glm.py" "${ARGS[@]}"
