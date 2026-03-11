#!/bin/bash
# Submit the full preprocessing pipeline for one or more subjects/sessions.
# NORDIC runs first; fmriprep is submitted with a dependency on all NORDIC jobs.
#
# Usage:
#   bash submit_pipeline.sh SUBJECT SESSION [SUBJECT SESSION ...]
#
# Examples:
#   bash submit_pipeline.sh 01 1          # single subject/session
#   bash submit_pipeline.sh 01 1 01 2    # both sessions of sub-01
#   bash submit_pipeline.sh 01 1 02 1    # ses-1 of two subjects

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NORDIC_DIR="$SCRIPT_DIR/nordic"
FMRIPREP_DIR="$SCRIPT_DIR/cluster_preproc"

if [ $# -lt 2 ] || [ $(( $# % 2 )) -ne 0 ]; then
    echo "Usage: $0 SUBJECT SESSION [SUBJECT SESSION ...]"
    exit 1
fi

ALL_NORDIC_IDS=""
SUBJECTS_SUBMITTED=""

while [ $# -gt 0 ]; do
    SUBJECT=$1
    SESSION=$2
    shift 2

    echo "=== Submitting NORDIC for sub-${SUBJECT} ses-${SESSION} ==="
    NORDIC_IDS=$(bash "$NORDIC_DIR/submit_all_runs.sh" "$SUBJECT" "$SESSION")
    echo "  Job IDs: $NORDIC_IDS"

    # Accumulate job IDs for fmriprep dependency
    for ID in $(echo "$NORDIC_IDS" | tr ':' ' '); do
        ALL_NORDIC_IDS="${ALL_NORDIC_IDS}:${ID}"
    done

    # Track unique subject numbers for the SLURM array
    SUB_NUM=$(echo "$SUBJECT" | sed 's/^0*//')  # strip leading zeros for array index
    if ! echo "$SUBJECTS_SUBMITTED" | grep -qw "$SUB_NUM"; then
        SUBJECTS_SUBMITTED="${SUBJECTS_SUBMITTED},${SUB_NUM}"
    fi
done

# Strip leading separators
ALL_NORDIC_IDS="${ALL_NORDIC_IDS#:}"
SUBJECTS_SUBMITTED="${SUBJECTS_SUBMITTED#,}"

echo ""
echo "=== Submitting fmriprep for subjects: ${SUBJECTS_SUBMITTED} ==="
echo "    Dependency: afterok:${ALL_NORDIC_IDS}"

cd "$FMRIPREP_DIR"
sbatch \
    --dependency="afterok:${ALL_NORDIC_IDS}" \
    --array="${SUBJECTS_SUBMITTED}" \
    fmriprep.sh

echo "Pipeline submitted."
